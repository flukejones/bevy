//! GPU mesh preprocessing.
//!
//! This is an optional pass that uses a compute shader to reduce the amount of
//! data that has to be transferred from the CPU to the GPU. When enabled,
//! instead of transferring [`MeshUniform`]s to the GPU, we transfer the smaller
//! [`MeshInputUniform`]s instead and use the GPU to calculate the remaining
//! derived fields in [`MeshUniform`].

use core::num::{NonZero, NonZeroU64};

use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, Handle};
use bevy_core_pipeline::{
    deferred::node::late_deferred_prepass,
    mip_generation::experimental::depth::{early_downsample_depth, ViewDepthPyramid},
    prepass::{
        node::{early_prepass, late_prepass},
        DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass, PreviousViewData,
        PreviousViewUniformOffset, PreviousViewUniforms,
    },
    schedule::{Core3d, Core3dSystems},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    prelude::resource_exists,
    query::{Has, Or, With, Without},
    resource::Resource,
    schedule::{common_conditions::any_match_filter, IntoScheduleConfigs as _},
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_log::warn_once;
use bevy_math::Vec4;
use bevy_platform::collections::HashMap;
use bevy_render::{
    batching::gpu_preprocessing::{
        clear_scene_unpacking_buffers, BatchedInstanceBuffers, BuildIndirectParametersMetadata,
        GpuBinMetadata, GpuOcclusionCullingWorkItemBuffers, GpuPreprocessingMode,
        GpuPreprocessingSupport, GpuSceneUnpackingMetadata, IndirectBatchSet,
        IndirectParametersBuffers, IndirectParametersBuildJob, IndirectParametersBuildJobs,
        IndirectParametersIndexed, IndirectParametersMetadata, IndirectParametersNonIndexed,
        LatePreprocessWorkItemIndirectParameters, PreprocessWorkItem, PreprocessWorkItemBuffers,
        SceneUnpackingBuffers, SceneUnpackingBuffersKey, SceneUnpackingMetadataIndex,
        UntypedPhaseBatchedInstanceBuffers, UntypedPhaseIndirectParametersBuffers,
    },
    diagnostic::RecordDiagnostics as _,
    occlusion_culling::OcclusionCulling,
    render_phase::{GpuRenderBinnedMeshInstance, UNIFORM_ALLOCATION_WORKGROUP_SIZE},
    render_resource::{
        binding_types::{storage_buffer, storage_buffer_read_only, texture_2d, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        BindingResource, Buffer, BufferBinding, BufferVec, CachedComputePipelineId, ComputePass,
        ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
        DynamicBindGroupLayoutEntries, PartialBufferVec, PipelineCache, RawBufferVec, ShaderStages,
        ShaderType, SparseBufferUpdateBindGroups, SparseBufferUpdateJobs,
        SparseBufferUpdatePipelines, SpecializedComputePipeline, SpecializedComputePipelines,
        TextureSampleType, UninitBufferVec,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    settings::WgpuFeatures,
    view::{
        ExtractedView, NoIndirectDrawing, RenderVisibilityRanges, RetainedViewEntity, ViewUniform,
        ViewUniformOffset, ViewUniforms,
    },
    GpuResourceAppExt, Render, RenderApp, RenderSystems,
};
use bevy_shader::Shader;
use bevy_utils::{default, TypeIdHashMap};
use bitflags::bitflags;
use smallvec::{smallvec, SmallVec};
use tracing::warn;

use crate::{
    LightEntity, MeshCullingData, MeshCullingDataBuffer, MeshInputUniform, MeshUniform,
    PreviousMeshInputUniform,
};

use super::{ShadowView, ViewLightEntities};

/// The GPU workgroup size.
const WORKGROUP_SIZE: usize = 64;

/// A plugin that builds mesh uniforms on GPU.
///
/// This will only be added if the platform supports compute shaders (e.g. not
/// on WebGL 2).
pub struct GpuMeshPreprocessPlugin {
    /// Whether we're building [`MeshUniform`]s on GPU.
    ///
    /// This requires compute shader support and so will be forcibly disabled if
    /// the platform doesn't support those.
    pub use_gpu_instance_buffer_builder: bool,
}

/// The compute shader pipelines for the GPU mesh preprocessing and indirect
/// parameter building passes.
#[derive(Resource)]
pub struct PreprocessPipelines {
    /// The pipeline used for CPU culling. This pipeline doesn't populate
    /// indirect parameter metadata.
    pub direct_preprocess: PreprocessPipeline,
    /// The pipeline used for mesh preprocessing when GPU frustum culling is in
    /// use, but occlusion culling isn't.
    ///
    /// This pipeline populates indirect parameter metadata.
    pub gpu_frustum_culling_preprocess: PreprocessPipeline,
    /// The pipeline used for the first phase of occlusion culling.
    ///
    /// This pipeline culls, transforms meshes, and populates indirect parameter
    /// metadata.
    pub early_gpu_occlusion_culling_preprocess: PreprocessPipeline,
    /// The pipeline used for the second phase of occlusion culling.
    ///
    /// This pipeline culls, transforms meshes, and populates indirect parameter
    /// metadata.
    pub late_gpu_occlusion_culling_preprocess: PreprocessPipeline,
    /// The pipeline that builds indirect draw parameters for indexed meshes,
    /// when frustum culling is enabled but occlusion culling *isn't* enabled.
    pub gpu_frustum_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline,
    /// The pipeline that builds indirect draw parameters for non-indexed
    /// meshes, when frustum culling is enabled but occlusion culling *isn't*
    /// enabled.
    pub gpu_frustum_culling_build_non_indexed_indirect_params: BuildIndirectParametersPipeline,
    /// Compute shader pipelines for the early prepass phase that draws meshes
    /// visible in the previous frame.
    pub early_phase: PreprocessPhasePipelines,
    /// Compute shader pipelines for the late prepass phase that draws meshes
    /// that weren't visible in the previous frame, but became visible this
    /// frame.
    pub late_phase: PreprocessPhasePipelines,
    /// Compute shader pipelines for the main color phase.
    pub main_phase: PreprocessPhasePipelines,
    /// Compute shader pipelines for the uniform allocation step.
    pub scene_unpacking: SceneUnpackingPipelines,
}

/// Compute shader pipelines for a specific phase: early, late, or main.
///
/// The distinction between these phases is relevant for occlusion culling.
#[derive(Clone)]
pub struct PreprocessPhasePipelines {
    /// The pipeline that resets the indirect draw counts used in
    /// `multi_draw_indirect_count` to 0 in preparation for a new pass.
    pub reset_indirect_batch_sets: ResetIndirectBatchSetsPipeline,
    /// The pipeline used for indexed indirect parameter building.
    ///
    /// This pipeline converts indirect parameter metadata into indexed indirect
    /// parameters.
    pub gpu_occlusion_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline,
    /// The pipeline used for non-indexed indirect parameter building.
    ///
    /// This pipeline converts indirect parameter metadata into non-indexed
    /// indirect parameters.
    pub gpu_occlusion_culling_build_non_indexed_indirect_params: BuildIndirectParametersPipeline,
}

/// The pipeline for the GPU mesh preprocessing shader.
pub struct PreprocessPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The shader asset handle.
    pub shader: Handle<Shader>,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

/// The pipeline for the batch set count reset shader.
///
/// This shader resets the indirect batch set count to 0 for each view. It runs
/// in between every phase (early, late, and main).
#[derive(Clone)]
pub struct ResetIndirectBatchSetsPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The shader asset handle.
    pub shader: Handle<Shader>,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

/// The pipeline for the indirect parameter building shader.
#[derive(Clone)]
pub struct BuildIndirectParametersPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The shader asset handle.
    pub shader: Handle<Shader>,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

/// Pipelines that allocate mesh uniforms and unpack cached mesh instances.
///
/// Local allocation and unpacking share a dispatch; larger allocations also need global scan and fan passes.
///
/// Although the `Handle<Shader>` is the same among these three pipelines, they
/// have to be separate so that the `SpecializedComputePipeline` implementation
/// on each sub-pipeline can access it.
#[derive(Clone)]
pub struct SceneUnpackingPipelines {
    /// The pipeline for step 1: local scan.
    pub local_scan: SceneUnpackingPipeline,
    /// The pipeline for step 2: global scan.
    pub global_scan: UniformAllocationGlobalScanPipeline,
    /// The pipeline for step 3: fan.
    pub fan: UniformAllocationFanPipeline,
}

/// The combined local allocation scan and instance-unpacking pipeline.
#[derive(Clone)]
pub struct SceneUnpackingPipeline {
    /// The bind group layout, shared among all the uniform allocation
    /// pipelines.
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The shader, also shared among all uniform allocation pipelines.
    pub shader: Handle<Shader>,
    /// The pipeline ID for the first step of the `unpack_scene` shader.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

/// The global prefix scan for allocations spanning multiple workgroups.
///
/// This step is skipped if the number of bins in the batch set is 256 or fewer.
#[derive(Clone)]
pub struct UniformAllocationGlobalScanPipeline {
    /// The bind group layout, shared among all the uniform allocation
    /// pipelines.
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The shader, also shared among all uniform allocation pipelines.
    pub shader: Handle<Shader>,
    /// The pipeline ID for the second step of the `unpack_scene` shader.
    pub pipeline_id_global_scan: Option<CachedComputePipelineId>,
}

/// The fan pass that adds global prefix offsets to later workgroups.
///
/// This step is skipped if the number of bins in the batch set is 256 or fewer.
#[derive(Clone)]
pub struct UniformAllocationFanPipeline {
    /// The bind group layout, shared among all the uniform allocation
    /// pipelines.
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The shader, also shared among all uniform allocation pipelines.
    pub shader: Handle<Shader>,
    /// The pipeline ID for the third step of the `unpack_scene` shader.
    pub pipeline_id_fan: Option<CachedComputePipelineId>,
}

bitflags! {
    /// Specifies variants of the mesh preprocessing shader.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PreprocessPipelineKey: u8 {
        /// Whether GPU frustum culling is in use.
        ///
        /// This `#define`'s `FRUSTUM_CULLING` in the shader.
        const FRUSTUM_CULLING = 1;
        /// Whether GPU two-phase occlusion culling is in use.
        ///
        /// This `#define`'s `OCCLUSION_CULLING` in the shader.
        const OCCLUSION_CULLING = 2;
        /// Whether this is the early phase of GPU two-phase occlusion culling.
        ///
        /// This `#define`'s `EARLY_PHASE` in the shader.
        const EARLY_PHASE = 4;
    }

    /// Specifies variants of the indirect parameter building shader.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BuildIndirectParametersPipelineKey: u8 {
        /// Whether the indirect parameter building shader is processing indexed
        /// meshes (those that have index buffers).
        ///
        /// This defines `INDEXED` in the shader.
        const INDEXED = 1;
        /// Whether the GPU and driver supports `multi_draw_indirect_count`.
        ///
        /// This defines `MULTI_DRAW_INDIRECT_COUNT_SUPPORTED` in the shader.
        const MULTI_DRAW_INDIRECT_COUNT_SUPPORTED = 2;
        /// Whether GPU two-phase occlusion culling is in use.
        ///
        /// This `#define`'s `OCCLUSION_CULLING` in the shader.
        const OCCLUSION_CULLING = 4;
        /// Whether this is the early phase of GPU two-phase occlusion culling.
        ///
        /// This `#define`'s `EARLY_PHASE` in the shader.
        const EARLY_PHASE = 8;
        /// Whether this is the late phase of GPU two-phase occlusion culling.
        ///
        /// This `#define`'s `LATE_PHASE` in the shader.
        const LATE_PHASE = 16;
        /// Whether this is the phase that runs after the early and late phases,
        /// and right before the main drawing logic, when GPU two-phase
        /// occlusion culling is in use.
        ///
        /// This `#define`'s `MAIN_PHASE` in the shader.
        const MAIN_PHASE = 32;
    }
}

/// The compute shader bind group for the mesh preprocessing pass for each
/// render phase.
///
/// This goes on the view. It maps the [`core::any::TypeId`] of a render phase
/// (e.g.  [`bevy_core_pipeline::core_3d::Opaque3d`]) to the
/// [`PhasePreprocessBindGroups`] for that phase.
#[derive(Component, Clone, Deref, DerefMut)]
pub struct PreprocessBindGroups(pub TypeIdHashMap<PhasePreprocessBindGroups>);

/// The compute shader bind group for the mesh preprocessing step for a single
/// render phase on a single view.
#[derive(Clone)]
pub enum PhasePreprocessBindGroups {
    /// The bind group used for the single invocation of the compute shader when
    /// indirect drawing is *not* being used.
    ///
    /// Because direct drawing doesn't require splitting the meshes into indexed
    /// and non-indexed meshes, there's only one bind group in this case.
    Direct(BindGroup),

    /// The bind groups used for the compute shader when indirect drawing is
    /// being used, but occlusion culling isn't being used.
    ///
    /// Because indirect drawing requires splitting the meshes into indexed and
    /// non-indexed meshes, there are two bind groups here.
    IndirectFrustumCulling {
        /// The bind group for indexed meshes.
        indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes.
        non_indexed: Option<BindGroup>,
    },

    /// The bind groups used for the compute shader when indirect drawing is
    /// being used, but occlusion culling isn't being used.
    ///
    /// Because indirect drawing requires splitting the meshes into indexed and
    /// non-indexed meshes, and because occlusion culling requires splitting
    /// this phase into early and late versions, there are four bind groups
    /// here.
    IndirectOcclusionCulling {
        /// The bind group for indexed meshes during the early mesh
        /// preprocessing phase.
        early_indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes during the early mesh
        /// preprocessing phase.
        early_non_indexed: Option<BindGroup>,
        /// The bind group for indexed meshes during the late mesh preprocessing
        /// phase.
        late_indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes during the late mesh
        /// preprocessing phase.
        late_non_indexed: Option<BindGroup>,
    },
}

/// The bind groups for the compute shaders that reset indirect draw counts and
/// build indirect parameters.
///
/// There's one set of bind group for each phase. Phases are keyed off their
/// [`core::any::TypeId`].
#[derive(Resource, Default, Deref, DerefMut)]
pub struct BuildIndirectParametersBindGroups(
    pub TypeIdHashMap<PhaseBuildIndirectParametersBindGroups>,
);

impl BuildIndirectParametersBindGroups {
    /// Creates a new, empty [`BuildIndirectParametersBindGroups`] table.
    pub fn new() -> BuildIndirectParametersBindGroups {
        Self::default()
    }
}

/// The per-phase set of bind groups for the compute shaders that reset indirect
/// draw counts and build indirect parameters.
pub struct PhaseBuildIndirectParametersBindGroups {
    /// The bind group for the `reset_indirect_batch_sets.wesl` shader, for
    /// indexed meshes.
    reset_indexed_indirect_batch_sets: Option<BindGroup>,
    /// The bind group for the `reset_indirect_batch_sets.wesl` shader, for
    /// non-indexed meshes.
    reset_non_indexed_indirect_batch_sets: Option<BindGroup>,
    /// The bind group for the `build_indirect_params.wesl` shader, for indexed
    /// meshes.
    build_indexed_indirect: Option<BindGroup>,
    /// The bind group for the `build_indirect_params.wesl` shader, for
    /// non-indexed meshes.
    build_non_indexed_indirect: Option<BindGroup>,
}

/// A resource, part of the render world, that stores all the bind groups for
/// scene unpacking. Bindings must use the current frame's metadata offsets.
///
/// There will be one such bind group for each combination of view, phase, and
/// mesh indexed-ness.
#[derive(Clone, Resource, Default, Deref, DerefMut)]
pub struct SceneUnpackingBindGroups(
    pub HashMap<SceneUnpackingBuffersKey, ViewPhaseSceneUnpackingBindGroups>,
);

/// Scene-unpacking bindings for the indexed and non-indexed batch sets of one view/phase.
#[derive(Clone)]
pub struct ViewPhaseSceneUnpackingBindGroups {
    /// The bind groups for the indexed meshes, one for each batch set.
    indexed: Vec<ViewPhaseSceneUnpackingBindGroup>,
    /// The bind groups for the non-indexed meshes, one for each batch set.
    non_indexed: Vec<ViewPhaseSceneUnpackingBindGroup>,
}

/// Scene-unpacking bindings and dispatch sizes for one cached batch set.
#[derive(Clone)]
pub struct ViewPhaseSceneUnpackingBindGroup {
    /// The index of the metadata in the
    /// [`SceneUnpackingBuffers::metadata`] buffer.
    pub metadata_index: SceneUnpackingMetadataIndex,
    /// The actual shader bind group.
    pub bind_group: BindGroup,
    /// The total number of bins in this batch set.
    pub bin_count: u32,
    /// The number of cached instances; it can exceed the bin count.
    pub mesh_instance_count: u32,
}

/// Stops the `GpuPreprocessNode` attempting to generate the buffer for this view
/// useful to avoid duplicating effort if the bind group is shared between views
#[derive(Component, Default)]
pub struct SkipGpuPreprocess;

type WithAnyPrepass = Or<(
    With<DepthPrepass>,
    With<NormalPrepass>,
    With<MotionVectorPrepass>,
    With<DeferredPrepass>,
)>;

impl Plugin for GpuMeshPreprocessPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "mesh_preprocess.wesl");
        embedded_asset!(app, "reset_indirect_batch_sets.wesl");
        embedded_asset!(app, "build_indirect_params.wesl");
        embedded_asset!(app, "unpack_scene.wesl");
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // This plugin does nothing if GPU instance buffer building isn't in
        // use.
        let gpu_preprocessing_support = render_app.world().resource::<GpuPreprocessingSupport>();
        if !self.use_gpu_instance_buffer_builder || !gpu_preprocessing_support.is_available() {
            return;
        }

        render_app
            .init_gpu_resource::<SceneUnpackingBindGroups>()
            .init_gpu_resource::<PreprocessPipelines>()
            .init_gpu_resource::<SpecializedComputePipelines<PreprocessPipeline>>()
            .init_gpu_resource::<SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>>()
            .init_gpu_resource::<SpecializedComputePipelines<BuildIndirectParametersPipeline>>()
            .init_gpu_resource::<SpecializedComputePipelines<SceneUnpackingPipeline>>()
            .init_gpu_resource::<SpecializedComputePipelines<UniformAllocationGlobalScanPipeline>>()
            .init_gpu_resource::<SpecializedComputePipelines<UniformAllocationFanPipeline>>()
            .add_systems(
                Render,
                (
                    clear_scene_unpacking_buffers.in_set(RenderSystems::PrepareResources),
                    prepare_preprocess_pipelines.in_set(RenderSystems::Prepare),
                    prepare_preprocess_bind_groups
                        .run_if(resource_exists::<BatchedInstanceBuffers<
                            MeshUniform,
                            MeshInputUniform
                        >>)
                        .in_set(RenderSystems::PrepareBindGroups)
                        .after(prepare_preprocess_pipelines),
                    write_mesh_culling_data_buffer.in_set(RenderSystems::PrepareResourcesFlush),
                ),
            )
            .add_systems(
                Core3d,
                (
                    (
                        unpack_scene,
                        early_gpu_preprocess,
                        early_prepass_build_indirect_parameters.run_if(any_match_filter::<(
                            With<PreprocessBindGroups>,
                            Without<SkipGpuPreprocess>,
                            Without<NoIndirectDrawing>,
                            Or<(WithAnyPrepass, With<ShadowView>)>,
                        )>),
                    )
                        .chain()
                        .before(early_prepass),
                    (
                        late_gpu_preprocess,
                        late_prepass_build_indirect_parameters.run_if(any_match_filter::<(
                            With<PreprocessBindGroups>,
                            Without<SkipGpuPreprocess>,
                            Without<NoIndirectDrawing>,
                            Or<(WithAnyPrepass, With<ShadowView>)>,
                            With<OcclusionCulling>,
                        )>),
                    )
                        .chain()
                        .after(early_downsample_depth)
                        .before(late_prepass),
                    main_build_indirect_parameters
                        .run_if(any_match_filter::<(
                            With<PreprocessBindGroups>,
                            Without<SkipGpuPreprocess>,
                            Without<NoIndirectDrawing>,
                        )>)
                        .after(late_prepass_build_indirect_parameters)
                        .after(late_deferred_prepass)
                        .before(Core3dSystems::MainPass),
                ),
            );
    }
}

/// Allocates mesh uniform ranges and unpacks cached instances for each batch set.
/// Global scan and fan passes finish allocations spanning multiple workgroups.
pub fn unpack_scene(
    current_view: ViewQuery<Option<&ViewLightEntities>, Without<SkipGpuPreprocess>>,
    view_query: Query<&ExtractedView, Without<SkipGpuPreprocess>>,
    light_query: Query<&LightEntity>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    pipeline_cache: Res<PipelineCache>,
    preprocess_pipelines: Res<PreprocessPipelines>,
    scene_unpacking_bind_groups: Res<SceneUnpackingBindGroups>,
    mut render_context: RenderContext,
) {
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    // Don't run if the shaders haven't been compiled yet.

    let (
        Some(scene_unpacking_pipeline_id),
        Some(uniform_allocation_global_scan_pipeline_id),
        Some(uniform_allocation_fan_pipeline_id),
    ) = (
        preprocess_pipelines.scene_unpacking.local_scan.pipeline_id,
        preprocess_pipelines
            .scene_unpacking
            .global_scan
            .pipeline_id_global_scan,
        preprocess_pipelines.scene_unpacking.fan.pipeline_id_fan,
    )
    else {
        return;
    };

    let (
        Some(scene_unpacking_pipeline),
        Some(uniform_allocation_global_scan_pipeline),
        Some(uniform_allocation_fan_pipeline),
    ) = (
        pipeline_cache.get_compute_pipeline(scene_unpacking_pipeline_id),
        pipeline_cache.get_compute_pipeline(uniform_allocation_global_scan_pipeline_id),
        pipeline_cache.get_compute_pipeline(uniform_allocation_fan_pipeline_id),
    )
    else {
        return;
    };

    let command_encoder = render_context.command_encoder();
    let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("scene unpacking"),
        timestamp_writes: None,
    });

    let pass_span = diagnostics.pass_span(&mut compute_pass, "scene_unpacking");

    // Gather up all views.
    let view_entity = current_view.entity();
    let shadow_cascade_views = current_view.into_inner();
    let all_views =
        gather_shadow_cascades_for_view(view_entity, shadow_cascade_views, &light_query);

    // Loop over each view…
    for view_entity in all_views {
        let Ok(view) = view_query.get(view_entity) else {
            continue;
        };

        // …and each phase within each view.
        for phase_type_id in batched_instance_buffers.phase_instance_buffers.keys() {
            let uniform_allocation_buffers_key = SceneUnpackingBuffersKey {
                phase: *phase_type_id,
                view: view.retained_view_entity,
            };

            // Fetch the bind groups for this (view, phase) combination.
            let Some(phase_scene_unpacking_bind_groups) =
                scene_unpacking_bind_groups.get(&uniform_allocation_buffers_key)
            else {
                continue;
            };

            // Invoke the shader for all batch sets corresponding to indexed
            // meshes and then for all batch sets corresponding to
            // non-indexed meshes.
            for scene_unpacking_bind_group in phase_scene_unpacking_bind_groups
                .indexed
                .iter()
                .chain(phase_scene_unpacking_bind_groups.non_indexed.iter())
            {
                dispatch_scene_unpacking(
                    &mut compute_pass,
                    scene_unpacking_pipeline,
                    uniform_allocation_global_scan_pipeline,
                    uniform_allocation_fan_pipeline,
                    scene_unpacking_bind_group,
                );
            }
        }
    }

    pass_span.end(&mut compute_pass);
}

fn dispatch_scene_unpacking(
    pass: &mut ComputePass<'_>,
    local_scan: &ComputePipeline,
    global_scan: &ComputePipeline,
    fan: &ComputePipeline,
    bindings: &ViewPhaseSceneUnpackingBindGroup,
) {
    let bin_workgroups = bindings
        .bin_count
        .div_ceil(UNIFORM_ALLOCATION_WORKGROUP_SIZE);
    let workgroups = bin_workgroups.max(
        bindings
            .mesh_instance_count
            .div_ceil(UNIFORM_ALLOCATION_WORKGROUP_SIZE),
    );
    if workgroups == 0 {
        return;
    }

    pass.set_pipeline(local_scan);
    pass.set_bind_group(
        0,
        &bindings.bind_group,
        &[bindings.metadata_index.uniform_offset()],
    );
    pass.dispatch_workgroups(workgroups, 1, 1);

    if bin_workgroups > 1 {
        pass.set_pipeline(global_scan);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(fan);
        pass.dispatch_workgroups(bin_workgroups - 1, 1, 1);
    }
}

pub fn early_gpu_preprocess(
    current_view: ViewQuery<Option<&ViewLightEntities>, Without<SkipGpuPreprocess>>,
    view_query: Query<
        (
            &ExtractedView,
            Option<&PreprocessBindGroups>,
            Option<&ViewUniformOffset>,
            Has<NoIndirectDrawing>,
            Has<OcclusionCulling>,
        ),
        Without<SkipGpuPreprocess>,
    >,
    light_query: Query<&LightEntity>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    pipeline_cache: Res<PipelineCache>,
    preprocess_pipelines: Res<PreprocessPipelines>,
    mut ctx: RenderContext,
) {
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let command_encoder = ctx.command_encoder();

    let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("early_mesh_preprocessing"),
        timestamp_writes: None,
    });

    let pass_span = diagnostics.pass_span(&mut compute_pass, "early_mesh_preprocessing");

    let view_entity = current_view.entity();
    let shadow_cascade_views = current_view.into_inner();
    let all_views =
        gather_shadow_cascades_for_view(view_entity, shadow_cascade_views, &light_query);

    // Run the compute passes.
    for view_entity in all_views {
        let Ok((view, bind_groups, view_uniform_offset, no_indirect_drawing, occlusion_culling)) =
            view_query.get(view_entity)
        else {
            continue;
        };

        let Some(bind_groups) = bind_groups else {
            continue;
        };
        let Some(view_uniform_offset) = view_uniform_offset else {
            continue;
        };

        // Select the right pipeline, depending on whether GPU culling is in
        // use.
        let maybe_pipeline_id = if no_indirect_drawing {
            preprocess_pipelines.direct_preprocess.pipeline_id
        } else if occlusion_culling {
            preprocess_pipelines
                .early_gpu_occlusion_culling_preprocess
                .pipeline_id
        } else {
            preprocess_pipelines
                .gpu_frustum_culling_preprocess
                .pipeline_id
        };

        // Fetch the pipeline.
        let Some(preprocess_pipeline_id) = maybe_pipeline_id else {
            warn!("The build mesh uniforms pipeline wasn't ready");
            continue;
        };

        let Some(preprocess_pipeline) = pipeline_cache.get_compute_pipeline(preprocess_pipeline_id)
        else {
            // This will happen while the pipeline is being compiled and is fine.
            continue;
        };

        compute_pass.set_pipeline(preprocess_pipeline);

        // Loop over each render phase.
        for (phase_type_id, batched_phase_instance_buffers) in
            &batched_instance_buffers.phase_instance_buffers
        {
            // Grab the work item buffers for this view.
            let Some(work_item_buffers) = batched_phase_instance_buffers
                .work_item_buffers
                .get(&view.retained_view_entity)
            else {
                continue;
            };

            // Fetch the bind group for the render phase.
            let Some(phase_bind_groups) = bind_groups.get(phase_type_id) else {
                continue;
            };

            // Make sure the mesh preprocessing shader has access to the
            // view info it needs to do culling and motion vector
            // computation.
            let dynamic_offsets = [view_uniform_offset.offset];

            // Are we drawing directly or indirectly?
            match *phase_bind_groups {
                PhasePreprocessBindGroups::Direct(ref bind_group) => {
                    // Invoke the mesh preprocessing shader to transform
                    // meshes only, but not cull.
                    let PreprocessWorkItemBuffers::Direct(work_item_buffer) = work_item_buffers
                    else {
                        continue;
                    };
                    compute_pass.set_bind_group(0, bind_group, &dynamic_offsets);
                    let workgroup_count = work_item_buffer.len().div_ceil(WORKGROUP_SIZE);
                    if workgroup_count > 0 {
                        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                    }
                }

                PhasePreprocessBindGroups::IndirectFrustumCulling {
                    indexed: ref maybe_indexed_bind_group,
                    non_indexed: ref maybe_non_indexed_bind_group,
                }
                | PhasePreprocessBindGroups::IndirectOcclusionCulling {
                    early_indexed: ref maybe_indexed_bind_group,
                    early_non_indexed: ref maybe_non_indexed_bind_group,
                    ..
                } => {
                    // Invoke the mesh preprocessing shader to transform and
                    // cull the meshes.
                    let PreprocessWorkItemBuffers::Indirect {
                        indexed: indexed_buffer,
                        non_indexed: non_indexed_buffer,
                        ..
                    } = work_item_buffers
                    else {
                        continue;
                    };

                    // Transform and cull indexed meshes if there are any.
                    if let Some(indexed_bind_group) = maybe_indexed_bind_group {
                        if let PreprocessWorkItemBuffers::Indirect {
                            gpu_occlusion_culling:
                                Some(GpuOcclusionCullingWorkItemBuffers {
                                    late_indirect_parameters_indexed_offset,
                                    ..
                                }),
                            ..
                        } = *work_item_buffers
                        {
                            compute_pass.set_immediates(
                                0,
                                bytemuck::bytes_of(&late_indirect_parameters_indexed_offset),
                            );
                        }

                        compute_pass.set_bind_group(0, indexed_bind_group, &dynamic_offsets);
                        let workgroup_count = indexed_buffer.len().div_ceil(WORKGROUP_SIZE);
                        if workgroup_count > 0 {
                            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                        }
                    }

                    // Transform and cull non-indexed meshes if there are any.
                    if let Some(non_indexed_bind_group) = maybe_non_indexed_bind_group {
                        if let PreprocessWorkItemBuffers::Indirect {
                            gpu_occlusion_culling:
                                Some(GpuOcclusionCullingWorkItemBuffers {
                                    late_indirect_parameters_non_indexed_offset,
                                    ..
                                }),
                            ..
                        } = *work_item_buffers
                        {
                            compute_pass.set_immediates(
                                0,
                                bytemuck::bytes_of(&late_indirect_parameters_non_indexed_offset),
                            );
                        }

                        compute_pass.set_bind_group(0, non_indexed_bind_group, &dynamic_offsets);
                        let workgroup_count = non_indexed_buffer.len().div_ceil(WORKGROUP_SIZE);
                        if workgroup_count > 0 {
                            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                        }
                    }
                }
            }
        }
    }

    pass_span.end(&mut compute_pass);
}

/// A helper function that returns all the shadow cascades that need to be
/// rendered for the given view, as well as the view itself.
fn gather_shadow_cascades_for_view(
    view_entity: Entity,
    shadow_cascade_views: Option<&ViewLightEntities>,
    light_query: &Query<&LightEntity>,
) -> SmallVec<[Entity; 8]> {
    let mut all_views: SmallVec<[_; 8]> = SmallVec::new();
    all_views.push(view_entity);
    if let Some(shadow_cascade_views) = shadow_cascade_views {
        all_views.extend(
            shadow_cascade_views
                .lights
                .iter()
                .filter(|light_entity| {
                    light_query.get(**light_entity).is_ok_and(|light_entity| {
                        matches!(*light_entity, LightEntity::Directional { .. })
                    })
                })
                .copied(),
        );
    }
    all_views
}

pub fn late_gpu_preprocess(
    current_view: ViewQuery<
        (&ExtractedView, &PreprocessBindGroups, &ViewUniformOffset),
        (
            Without<SkipGpuPreprocess>,
            Without<NoIndirectDrawing>,
            With<OcclusionCulling>,
            With<DepthPrepass>,
        ),
    >,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    pipeline_cache: Res<PipelineCache>,
    preprocess_pipelines: Res<PreprocessPipelines>,
    mut ctx: RenderContext,
) {
    let (view, bind_groups, view_uniform_offset) = current_view.into_inner();

    // Fetch the pipeline BEFORE starting diagnostic spans to avoid panic on early return
    let maybe_pipeline_id = preprocess_pipelines
        .late_gpu_occlusion_culling_preprocess
        .pipeline_id;

    let Some(preprocess_pipeline_id) = maybe_pipeline_id else {
        warn_once!("The build mesh uniforms pipeline wasn't ready");
        return;
    };

    let Some(preprocess_pipeline) = pipeline_cache.get_compute_pipeline(preprocess_pipeline_id)
    else {
        // This will happen while the pipeline is being compiled and is fine.
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let command_encoder = ctx.command_encoder();

    let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("late_mesh_preprocessing"),
        timestamp_writes: None,
    });

    let pass_span = diagnostics.pass_span(&mut compute_pass, "late_mesh_preprocessing");

    compute_pass.set_pipeline(preprocess_pipeline);

    // Loop over each phase. Because we built the phases in parallel,
    // each phase has a separate set of instance buffers.
    for (phase_type_id, batched_phase_instance_buffers) in
        &batched_instance_buffers.phase_instance_buffers
    {
        let UntypedPhaseBatchedInstanceBuffers {
            ref work_item_buffers,
            ref late_indexed_indirect_parameters_buffer,
            ref late_non_indexed_indirect_parameters_buffer,
            ..
        } = *batched_phase_instance_buffers;

        // Grab the work item buffers for this view.
        let Some(phase_work_item_buffers) = work_item_buffers.get(&view.retained_view_entity)
        else {
            continue;
        };

        let (
            PreprocessWorkItemBuffers::Indirect {
                gpu_occlusion_culling:
                    Some(GpuOcclusionCullingWorkItemBuffers {
                        late_indirect_parameters_indexed_offset,
                        late_indirect_parameters_non_indexed_offset,
                        ..
                    }),
                ..
            },
            Some(PhasePreprocessBindGroups::IndirectOcclusionCulling {
                late_indexed: maybe_late_indexed_bind_group,
                late_non_indexed: maybe_late_non_indexed_bind_group,
                ..
            }),
            Some(late_indexed_indirect_parameters_buffer),
            Some(late_non_indexed_indirect_parameters_buffer),
        ) = (
            phase_work_item_buffers,
            bind_groups.get(phase_type_id),
            late_indexed_indirect_parameters_buffer.buffer(),
            late_non_indexed_indirect_parameters_buffer.buffer(),
        )
        else {
            continue;
        };

        let mut dynamic_offsets: SmallVec<[u32; 1]> = smallvec![];
        dynamic_offsets.push(view_uniform_offset.offset);

        // If there's no space reserved for work items, then don't
        // bother doing the dispatch, as there can't possibly be any
        // meshes of the given class (indexed or non-indexed) in this
        // phase.

        // Transform and cull indexed meshes if there are any.
        if let Some(late_indexed_bind_group) = maybe_late_indexed_bind_group {
            compute_pass.set_immediates(
                0,
                bytemuck::bytes_of(late_indirect_parameters_indexed_offset),
            );

            compute_pass.set_bind_group(0, late_indexed_bind_group, &dynamic_offsets);
            compute_pass.dispatch_workgroups_indirect(
                late_indexed_indirect_parameters_buffer,
                (*late_indirect_parameters_indexed_offset as u64)
                    * (size_of::<LatePreprocessWorkItemIndirectParameters>() as u64),
            );
        }

        // Transform and cull non-indexed meshes if there are any.
        if let Some(late_non_indexed_bind_group) = maybe_late_non_indexed_bind_group {
            compute_pass.set_immediates(
                0,
                bytemuck::bytes_of(late_indirect_parameters_non_indexed_offset),
            );

            compute_pass.set_bind_group(0, late_non_indexed_bind_group, &dynamic_offsets);
            compute_pass.dispatch_workgroups_indirect(
                late_non_indexed_indirect_parameters_buffer,
                (*late_indirect_parameters_non_indexed_offset as u64)
                    * (size_of::<LatePreprocessWorkItemIndirectParameters>() as u64),
            );
        }
    }

    pass_span.end(&mut compute_pass);
}

/// A render graph system, run for each view, that builds the indirect
/// parameters for multi-draw indirect calls for the early prepass.
///
/// The early prepass is the prepass that draws objects that were visible in the
/// previous frame.
pub fn early_prepass_build_indirect_parameters(
    current_view: ViewQuery<&ExtractedView>,
    preprocess_pipelines: Res<PreprocessPipelines>,
    build_indirect_params_bind_groups: Option<Res<BuildIndirectParametersBindGroups>>,
    pipeline_cache: Res<PipelineCache>,
    indirect_parameters_buffers: Option<Res<IndirectParametersBuffers>>,
    build_indirect_parameters_uniform_indices: Res<BuildIndirectParametersMetadata>,
    mut ctx: RenderContext,
) {
    run_build_indirect_parameters(
        &mut ctx,
        current_view.into_inner().retained_view_entity,
        build_indirect_params_bind_groups.as_deref(),
        &pipeline_cache,
        indirect_parameters_buffers.as_deref(),
        &build_indirect_parameters_uniform_indices,
        &preprocess_pipelines.early_phase,
        "early_prepass_indirect_parameters_building",
    );
}

/// A render graph system, run for each view, that builds the indirect
/// parameters for multi-draw indirect calls for the late prepass.
///
/// The late prepass is the prepass that draws objects that weren't visible in
/// the previous frame but became visible this frame (disocclusions). It'll be
/// skipped if occlusion culling is disabled.
pub fn late_prepass_build_indirect_parameters(
    current_view: ViewQuery<&ExtractedView>,
    preprocess_pipelines: Res<PreprocessPipelines>,
    build_indirect_params_bind_groups: Option<Res<BuildIndirectParametersBindGroups>>,
    pipeline_cache: Res<PipelineCache>,
    indirect_parameters_buffers: Option<Res<IndirectParametersBuffers>>,
    build_indirect_parameters_uniform_indices: Res<BuildIndirectParametersMetadata>,
    mut ctx: RenderContext,
) {
    run_build_indirect_parameters(
        &mut ctx,
        current_view.into_inner().retained_view_entity,
        build_indirect_params_bind_groups.as_deref(),
        &pipeline_cache,
        indirect_parameters_buffers.as_deref(),
        &build_indirect_parameters_uniform_indices,
        &preprocess_pipelines.late_phase,
        "late_prepass_indirect_parameters_building",
    );
}

/// A render graph system, run for each view, that builds the indirect
/// parameters for multi-draw indirect calls for the main opaque and transparent
/// passes.
pub fn main_build_indirect_parameters(
    current_view: ViewQuery<&ExtractedView, Without<ShadowView>>,
    preprocess_pipelines: Res<PreprocessPipelines>,
    build_indirect_params_bind_groups: Option<Res<BuildIndirectParametersBindGroups>>,
    pipeline_cache: Res<PipelineCache>,
    indirect_parameters_buffers: Option<Res<IndirectParametersBuffers>>,
    build_indirect_parameters_uniform_indices: Res<BuildIndirectParametersMetadata>,
    mut ctx: RenderContext,
) {
    run_build_indirect_parameters(
        &mut ctx,
        current_view.into_inner().retained_view_entity,
        build_indirect_params_bind_groups.as_deref(),
        &pipeline_cache,
        indirect_parameters_buffers.as_deref(),
        &build_indirect_parameters_uniform_indices,
        &preprocess_pipelines.main_phase,
        "main_indirect_parameters_building",
    );
}

/// Shared logic common to all render graph systems that build indirect
/// parameters for multi-draw indirect calls.
pub(crate) fn run_build_indirect_parameters(
    ctx: &mut RenderContext,
    retained_view_entity: RetainedViewEntity,
    build_indirect_params_bind_groups: Option<&BuildIndirectParametersBindGroups>,
    pipeline_cache: &PipelineCache,
    indirect_parameters_buffers: Option<&IndirectParametersBuffers>,
    build_indirect_parameters_uniform_indices: &BuildIndirectParametersMetadata,
    preprocess_phase_pipelines: &PreprocessPhasePipelines,
    label: &'static str,
) {
    let Some(build_indirect_params_bind_groups) = build_indirect_params_bind_groups else {
        return;
    };
    let Some(indirect_parameters_buffers) = indirect_parameters_buffers else {
        return;
    };
    let Some(view_build_indirect_parameters_uniform_indices) =
        build_indirect_parameters_uniform_indices.get(&retained_view_entity)
    else {
        return;
    };

    let command_encoder = ctx.command_encoder();

    let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });

    // Fetch the pipeline.
    let (
        Some(reset_indirect_batch_sets_pipeline_id),
        Some(build_indexed_indirect_params_pipeline_id),
        Some(build_non_indexed_indirect_params_pipeline_id),
    ) = (
        preprocess_phase_pipelines
            .reset_indirect_batch_sets
            .pipeline_id,
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_indexed_indirect_params
            .pipeline_id,
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_non_indexed_indirect_params
            .pipeline_id,
    )
    else {
        warn!("The build indirect parameters pipelines weren't ready");
        return;
    };

    let (
        Some(reset_indirect_batch_sets_pipeline),
        Some(build_indexed_indirect_params_pipeline),
        Some(build_non_indexed_indirect_params_pipeline),
    ) = (
        pipeline_cache.get_compute_pipeline(reset_indirect_batch_sets_pipeline_id),
        pipeline_cache.get_compute_pipeline(build_indexed_indirect_params_pipeline_id),
        pipeline_cache.get_compute_pipeline(build_non_indexed_indirect_params_pipeline_id),
    )
    else {
        // This will happen while the pipeline is being compiled and is fine.
        return;
    };

    // Loop over each phase. As each has as separate set of buffers, we need to
    // build indirect parameters individually for each phase.
    for (phase_type_id, phase_build_indirect_params_bind_groups) in
        build_indirect_params_bind_groups.iter()
    {
        let Some(phase_indirect_parameters_buffers) =
            indirect_parameters_buffers.get(phase_type_id)
        else {
            continue;
        };
        let Some(build_indirect_parameters_uniform_index) =
            view_build_indirect_parameters_uniform_indices.get(phase_type_id)
        else {
            continue;
        };

        // Build indexed indirect parameters.
        if let (
            Some(reset_indexed_indirect_batch_sets_bind_group),
            Some(build_indirect_indexed_params_bind_group),
        ) = (
            &phase_build_indirect_params_bind_groups.reset_indexed_indirect_batch_sets,
            &phase_build_indirect_params_bind_groups.build_indexed_indirect,
        ) {
            compute_pass.set_pipeline(reset_indirect_batch_sets_pipeline);
            compute_pass.set_bind_group(0, reset_indexed_indirect_batch_sets_bind_group, &[]);
            let workgroup_count = phase_indirect_parameters_buffers
                .batch_set_count(true)
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }

            compute_pass.set_pipeline(build_indexed_indirect_params_pipeline);

            for indexed_build_indirect_parameters_metadata in
                &build_indirect_parameters_uniform_index.indexed
            {
                compute_pass.set_bind_group(
                    0,
                    build_indirect_indexed_params_bind_group,
                    &[indexed_build_indirect_parameters_metadata.uniform_offset],
                );
                let workgroup_count = indexed_build_indirect_parameters_metadata
                    .batch_count
                    .div_ceil(WORKGROUP_SIZE as u32);
                if workgroup_count > 0 {
                    compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                }
            }
        }

        // Build non-indexed indirect parameters.
        if let (
            Some(reset_non_indexed_indirect_batch_sets_bind_group),
            Some(build_indirect_non_indexed_params_bind_group),
        ) = (
            &phase_build_indirect_params_bind_groups.reset_non_indexed_indirect_batch_sets,
            &phase_build_indirect_params_bind_groups.build_non_indexed_indirect,
        ) {
            compute_pass.set_pipeline(reset_indirect_batch_sets_pipeline);
            compute_pass.set_bind_group(0, reset_non_indexed_indirect_batch_sets_bind_group, &[]);
            let workgroup_count = phase_indirect_parameters_buffers
                .batch_set_count(false)
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }

            compute_pass.set_pipeline(build_non_indexed_indirect_params_pipeline);

            for non_indexed_build_indirect_parameters_metadata in
                &build_indirect_parameters_uniform_index.non_indexed
            {
                compute_pass.set_bind_group(
                    0,
                    build_indirect_non_indexed_params_bind_group,
                    &[non_indexed_build_indirect_parameters_metadata.uniform_offset],
                );
                let workgroup_count = non_indexed_build_indirect_parameters_metadata
                    .batch_count
                    .div_ceil(WORKGROUP_SIZE as u32);
                if workgroup_count > 0 {
                    compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                }
            }
        }
    }
}

impl PreprocessPipelines {
    /// Returns true if the preprocessing and indirect parameters pipelines have
    /// been loaded or false otherwise.
    pub(crate) fn pipelines_are_loaded(
        &self,
        pipeline_cache: &PipelineCache,
        preprocessing_support: &GpuPreprocessingSupport,
    ) -> bool {
        match preprocessing_support.max_supported_mode {
            GpuPreprocessingMode::None => false,
            GpuPreprocessingMode::PreprocessingOnly => {
                self.direct_preprocess.is_loaded(pipeline_cache)
                    && self
                        .gpu_frustum_culling_preprocess
                        .is_loaded(pipeline_cache)
            }
            GpuPreprocessingMode::Culling => {
                self.direct_preprocess.is_loaded(pipeline_cache)
                    && self
                        .gpu_frustum_culling_preprocess
                        .is_loaded(pipeline_cache)
                    && self
                        .early_gpu_occlusion_culling_preprocess
                        .is_loaded(pipeline_cache)
                    && self
                        .late_gpu_occlusion_culling_preprocess
                        .is_loaded(pipeline_cache)
                    && self
                        .gpu_frustum_culling_build_indexed_indirect_params
                        .is_loaded(pipeline_cache)
                    && self
                        .gpu_frustum_culling_build_non_indexed_indirect_params
                        .is_loaded(pipeline_cache)
                    && self.early_phase.is_loaded(pipeline_cache)
                    && self.late_phase.is_loaded(pipeline_cache)
                    && self.main_phase.is_loaded(pipeline_cache)
                    // Missing scene unpacking would leave stale work items and allocations.
                    && self.scene_unpacking.is_loaded(pipeline_cache)
            }
        }
    }
}

impl SceneUnpackingPipelines {
    /// Returns true if all three uniform allocation pipelines have been
    /// loaded into the pipeline cache or false otherwise.
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        [
            self.local_scan.pipeline_id,
            self.global_scan.pipeline_id_global_scan,
            self.fan.pipeline_id_fan,
        ]
        .into_iter()
        .all(|pipeline_id| {
            pipeline_id.is_some_and(|pipeline_id| {
                pipeline_cache.get_compute_pipeline(pipeline_id).is_some()
            })
        })
    }
}

impl PreprocessPhasePipelines {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.reset_indirect_batch_sets.is_loaded(pipeline_cache)
            && self
                .gpu_occlusion_culling_build_indexed_indirect_params
                .is_loaded(pipeline_cache)
            && self
                .gpu_occlusion_culling_build_non_indexed_indirect_params
                .is_loaded(pipeline_cache)
    }
}

impl PreprocessPipeline {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl ResetIndirectBatchSetsPipeline {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl BuildIndirectParametersPipeline {
    /// Returns true if this pipeline has been loaded into the pipeline cache or
    /// false otherwise.
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl SpecializedComputePipeline for PreprocessPipeline {
    type Key = PreprocessPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec!["WRITE_INDIRECT_PARAMETERS_METADATA".into()];
        if key.contains(PreprocessPipelineKey::FRUSTUM_CULLING) {
            shader_defs.push("INDIRECT".into());
            shader_defs.push("FRUSTUM_CULLING".into());
        }
        if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
            shader_defs.push("OCCLUSION_CULLING".into());
            if key.contains(PreprocessPipelineKey::EARLY_PHASE) {
                shader_defs.push("EARLY_PHASE".into());
            } else {
                shader_defs.push("LATE_PHASE".into());
            }
        }

        ComputePipelineDescriptor {
            label: Some(
                format!(
                    "mesh preprocessing ({})",
                    if key.contains(
                        PreprocessPipelineKey::OCCLUSION_CULLING
                            | PreprocessPipelineKey::EARLY_PHASE
                    ) {
                        "early GPU occlusion culling"
                    } else if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
                        "late GPU occlusion culling"
                    } else if key.contains(PreprocessPipelineKey::FRUSTUM_CULLING) {
                        "GPU frustum culling"
                    } else {
                        "direct"
                    }
                )
                .into(),
            ),
            layout: vec![self.bind_group_layout.clone()],
            immediate_size: if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
                4
            } else {
                0
            },
            shader: self.shader.clone(),
            shader_defs,
            ..default()
        }
    }
}

impl FromWorld for PreprocessPipelines {
    fn from_world(world: &mut World) -> Self {
        // GPU culling bind group parameters are a superset of those in the CPU
        // culling (direct) shader.
        let direct_bind_group_layout_entries = preprocess_direct_bind_group_layout_entries();
        let gpu_frustum_culling_bind_group_layout_entries = gpu_culling_bind_group_layout_entries();
        let gpu_early_occlusion_culling_bind_group_layout_entries =
            gpu_occlusion_culling_bind_group_layout_entries().extend_with_indices((
                (
                    12,
                    storage_buffer::<PreprocessWorkItem>(/*has_dynamic_offset=*/ false),
                ),
                (
                    13,
                    storage_buffer::<LatePreprocessWorkItemIndirectParameters>(
                        /*has_dynamic_offset=*/ false,
                    ),
                ),
            ));
        let gpu_late_occlusion_culling_bind_group_layout_entries =
            gpu_occlusion_culling_bind_group_layout_entries().extend_with_indices(((
                13,
                storage_buffer_read_only::<LatePreprocessWorkItemIndirectParameters>(
                    /*has_dynamic_offset=*/ false,
                ),
            ),));

        let reset_indirect_batch_sets_bind_group_layout_entries =
            DynamicBindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (storage_buffer::<IndirectBatchSet>(false),),
            );

        // Indexed and non-indexed bind group parameters share all the bind
        // group layout entries except the final one.
        let build_indexed_indirect_params_bind_group_layout_entries =
            build_indirect_params_bind_group_layout_entries()
                .extend_sequential((storage_buffer::<IndirectParametersIndexed>(false),));
        let build_non_indexed_indirect_params_bind_group_layout_entries =
            build_indirect_params_bind_group_layout_entries()
                .extend_sequential((storage_buffer::<IndirectParametersNonIndexed>(false),));

        let scene_unpacking_bind_group_layout_entries = scene_unpacking_bind_group_layout_entries();

        // Create the bind group layouts.
        let direct_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build mesh uniforms direct bind group layout",
            &direct_bind_group_layout_entries,
        );
        let gpu_frustum_culling_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build mesh uniforms GPU frustum culling bind group layout",
            &gpu_frustum_culling_bind_group_layout_entries,
        );
        let gpu_early_occlusion_culling_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build mesh uniforms GPU early occlusion culling bind group layout",
            &gpu_early_occlusion_culling_bind_group_layout_entries,
        );
        let gpu_late_occlusion_culling_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build mesh uniforms GPU late occlusion culling bind group layout",
            &gpu_late_occlusion_culling_bind_group_layout_entries,
        );
        let reset_indirect_batch_sets_bind_group_layout = BindGroupLayoutDescriptor::new(
            "reset indirect batch sets bind group layout",
            &reset_indirect_batch_sets_bind_group_layout_entries,
        );
        let build_indexed_indirect_params_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build indexed indirect parameters bind group layout",
            &build_indexed_indirect_params_bind_group_layout_entries,
        );
        let build_non_indexed_indirect_params_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build non-indexed indirect parameters bind group layout",
            &build_non_indexed_indirect_params_bind_group_layout_entries,
        );
        let scene_unpacking_bind_group_layout = BindGroupLayoutDescriptor::new(
            "scene unpacking bind group layout",
            &scene_unpacking_bind_group_layout_entries,
        );

        let preprocess_shader = load_embedded_asset!(world, "mesh_preprocess.wesl");
        let reset_indirect_batch_sets_shader =
            load_embedded_asset!(world, "reset_indirect_batch_sets.wesl");
        let build_indirect_params_shader =
            load_embedded_asset!(world, "build_indirect_params.wesl");
        let scene_unpacking_shader = load_embedded_asset!(world, "unpack_scene.wesl");

        let preprocess_phase_pipelines = PreprocessPhasePipelines {
            reset_indirect_batch_sets: ResetIndirectBatchSetsPipeline {
                bind_group_layout: reset_indirect_batch_sets_bind_group_layout.clone(),
                shader: reset_indirect_batch_sets_shader,
                pipeline_id: None,
            },
            gpu_occlusion_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline {
                bind_group_layout: build_indexed_indirect_params_bind_group_layout.clone(),
                shader: build_indirect_params_shader.clone(),
                pipeline_id: None,
            },
            gpu_occlusion_culling_build_non_indexed_indirect_params:
                BuildIndirectParametersPipeline {
                    bind_group_layout: build_non_indexed_indirect_params_bind_group_layout.clone(),
                    shader: build_indirect_params_shader.clone(),
                    pipeline_id: None,
                },
        };

        PreprocessPipelines {
            direct_preprocess: PreprocessPipeline {
                bind_group_layout: direct_bind_group_layout,
                shader: preprocess_shader.clone(),
                pipeline_id: None,
            },
            gpu_frustum_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_frustum_culling_bind_group_layout,
                shader: preprocess_shader.clone(),
                pipeline_id: None,
            },
            early_gpu_occlusion_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_early_occlusion_culling_bind_group_layout,
                shader: preprocess_shader.clone(),
                pipeline_id: None,
            },
            late_gpu_occlusion_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_late_occlusion_culling_bind_group_layout,
                shader: preprocess_shader,
                pipeline_id: None,
            },
            gpu_frustum_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline {
                bind_group_layout: build_indexed_indirect_params_bind_group_layout.clone(),
                shader: build_indirect_params_shader.clone(),
                pipeline_id: None,
            },
            gpu_frustum_culling_build_non_indexed_indirect_params:
                BuildIndirectParametersPipeline {
                    bind_group_layout: build_non_indexed_indirect_params_bind_group_layout.clone(),
                    shader: build_indirect_params_shader,
                    pipeline_id: None,
                },
            early_phase: preprocess_phase_pipelines.clone(),
            late_phase: preprocess_phase_pipelines.clone(),
            main_phase: preprocess_phase_pipelines.clone(),
            scene_unpacking: SceneUnpackingPipelines {
                local_scan: SceneUnpackingPipeline {
                    bind_group_layout: scene_unpacking_bind_group_layout.clone(),
                    shader: scene_unpacking_shader.clone(),
                    pipeline_id: None,
                },
                global_scan: UniformAllocationGlobalScanPipeline {
                    bind_group_layout: scene_unpacking_bind_group_layout.clone(),
                    shader: scene_unpacking_shader.clone(),
                    pipeline_id_global_scan: None,
                },
                fan: UniformAllocationFanPipeline {
                    bind_group_layout: scene_unpacking_bind_group_layout.clone(),
                    shader: scene_unpacking_shader.clone(),
                    pipeline_id_fan: None,
                },
            },
        }
    }
}

fn preprocess_direct_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::COMPUTE,
        (
            // `view`
            (
                0,
                uniform_buffer::<ViewUniform>(/* has_dynamic_offset= */ true),
            ),
            // `current_input`
            (3, storage_buffer_read_only::<MeshInputUniform>(false)),
            // `previous_input`
            (
                4,
                storage_buffer_read_only::<PreviousMeshInputUniform>(false),
            ),
            // `indices`
            (5, storage_buffer_read_only::<PreprocessWorkItem>(false)),
            // `output`
            (6, storage_buffer::<MeshUniform>(false)),
        ),
    )
}

// Returns the first 5 bind group layout entries shared between all invocations
// of the indirect parameters building shader.
fn build_indirect_params_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::COMPUTE,
        (
            // @group(0) @binding(0) var<storage> current_input:
            // array<MeshInput>;
            (0, storage_buffer_read_only::<MeshInputUniform>(false)),
            // @group(0) @binding(1) var<storage> indirect_parameters_metadata:
            // array<IndirectParametersMetadata>;
            (
                1,
                storage_buffer_read_only::<IndirectParametersMetadata>(false),
            ),
            // @group(0) @binding(3) var<storage, read_write>
            // indirect_batch_sets: array<IndirectBatchSet>;
            (3, storage_buffer::<IndirectBatchSet>(false)),
            // @group(0) @binding(4) var<uniform> indirect_parameters_build_job:
            // IndirectParametersBuildJob;
            (4, uniform_buffer::<IndirectParametersBuildJob>(true)),
        ),
    )
}

/// A system that specializes the `mesh_preprocess.wesl` and
/// `build_indirect_params.wesl` pipelines if necessary.
fn gpu_culling_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    // GPU culling bind group parameters are a superset of those in the CPU
    // culling (direct) shader.
    preprocess_direct_bind_group_layout_entries().extend_with_indices((
        // @group(0) @binding(7) var<storage> indirect_parameters_metadata:
        // array<IndirectParametersMetadata>;
        (
            7,
            storage_buffer::<IndirectParametersMetadata>(/* has_dynamic_offset= */ false),
        ),
        // `mesh_culling_data`
        (
            9,
            storage_buffer_read_only::<MeshCullingData>(/* has_dynamic_offset= */ false),
        ),
        // `visibility_ranges`
        (
            10,
            storage_buffer_read_only::<Vec4>(/* has_dynamic_offset= */ false),
        ),
    ))
}

fn gpu_occlusion_culling_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    gpu_culling_bind_group_layout_entries().extend_with_indices((
        (
            2,
            uniform_buffer::<PreviousViewData>(/*has_dynamic_offset=*/ false),
        ),
        (
            11,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
    ))
}

/// Creates bind group layout entries for GPU scene unpacking
/// shader (`unpack_scene`).
fn scene_unpacking_bind_group_layout_entries() -> BindGroupLayoutEntries<7> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            // @group(0) @binding(0) var<uniform> metadata:
            // SceneUnpackingMetadata;
            uniform_buffer::<GpuSceneUnpackingMetadata>(true),
            // @group(0) @binding(1) var<storage> bin_metadata: array<BinMetadata>;
            storage_buffer_read_only::<GpuBinMetadata>(false),
            // @group(0) @binding(2) var<storage, read_write>
            // indirect_parameters_metadata: array<IndirectParametersMetadata>;
            storage_buffer::<IndirectParametersMetadata>(false),
            // @group(0) @binding(3) var<storage, read_write> fan_buffer:
            // array<u32>;
            storage_buffer::<u32>(false),
            storage_buffer_read_only::<GpuRenderBinnedMeshInstance>(false),
            storage_buffer::<PreprocessWorkItem>(false),
            storage_buffer_read_only::<u32>(false),
        ),
    )
}

/// A system that specializes the pipelines relating to mesh preprocessing if
/// necessary.
///
/// These pipelines include those corresponding to the mesh preprocessing shader
/// itself, in addition to those corresponding to the indirect batch set
/// resetting shader, the indirect parameters building shader, and the bin
/// unpacking shader.
pub fn prepare_preprocess_pipelines(
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut specialized_preprocess_pipelines: ResMut<SpecializedComputePipelines<PreprocessPipeline>>,
    mut specialized_reset_indirect_batch_sets_pipelines: ResMut<
        SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>,
    >,
    mut specialized_build_indirect_parameters_pipelines: ResMut<
        SpecializedComputePipelines<BuildIndirectParametersPipeline>,
    >,
    mut specialized_scene_unpacking_pipelines: ResMut<
        SpecializedComputePipelines<SceneUnpackingPipeline>,
    >,
    mut specialized_uniform_allocation_global_scan_pipelines: ResMut<
        SpecializedComputePipelines<UniformAllocationGlobalScanPipeline>,
    >,
    mut specialized_uniform_allocation_fan_pipelines: ResMut<
        SpecializedComputePipelines<UniformAllocationFanPipeline>,
    >,
    preprocess_pipelines: ResMut<PreprocessPipelines>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    let preprocess_pipelines = preprocess_pipelines.into_inner();

    preprocess_pipelines.direct_preprocess.prepare(
        &pipeline_cache,
        &mut specialized_preprocess_pipelines,
        PreprocessPipelineKey::empty(),
    );
    preprocess_pipelines.gpu_frustum_culling_preprocess.prepare(
        &pipeline_cache,
        &mut specialized_preprocess_pipelines,
        PreprocessPipelineKey::FRUSTUM_CULLING,
    );

    if gpu_preprocessing_support.is_culling_supported() {
        preprocess_pipelines
            .early_gpu_occlusion_culling_preprocess
            .prepare(
                &pipeline_cache,
                &mut specialized_preprocess_pipelines,
                PreprocessPipelineKey::FRUSTUM_CULLING
                    | PreprocessPipelineKey::OCCLUSION_CULLING
                    | PreprocessPipelineKey::EARLY_PHASE,
            );
        preprocess_pipelines
            .late_gpu_occlusion_culling_preprocess
            .prepare(
                &pipeline_cache,
                &mut specialized_preprocess_pipelines,
                PreprocessPipelineKey::FRUSTUM_CULLING | PreprocessPipelineKey::OCCLUSION_CULLING,
            );
    }

    let mut build_indirect_parameters_pipeline_key = BuildIndirectParametersPipelineKey::empty();

    // If the GPU and driver support `multi_draw_indirect_count`, tell the
    // shader that.
    if render_device
        .wgpu_device()
        .features()
        .contains(WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT)
    {
        build_indirect_parameters_pipeline_key
            .insert(BuildIndirectParametersPipelineKey::MULTI_DRAW_INDIRECT_COUNT_SUPPORTED);
    }

    preprocess_pipelines
        .gpu_frustum_culling_build_indexed_indirect_params
        .prepare(
            &pipeline_cache,
            &mut specialized_build_indirect_parameters_pipelines,
            build_indirect_parameters_pipeline_key | BuildIndirectParametersPipelineKey::INDEXED,
        );
    preprocess_pipelines
        .gpu_frustum_culling_build_non_indexed_indirect_params
        .prepare(
            &pipeline_cache,
            &mut specialized_build_indirect_parameters_pipelines,
            build_indirect_parameters_pipeline_key,
        );

    if !gpu_preprocessing_support.is_culling_supported() {
        return;
    }

    for (preprocess_phase_pipelines, build_indirect_parameters_phase_pipeline_key) in [
        (
            &mut preprocess_pipelines.early_phase,
            BuildIndirectParametersPipelineKey::EARLY_PHASE,
        ),
        (
            &mut preprocess_pipelines.late_phase,
            BuildIndirectParametersPipelineKey::LATE_PHASE,
        ),
        (
            &mut preprocess_pipelines.main_phase,
            BuildIndirectParametersPipelineKey::MAIN_PHASE,
        ),
    ] {
        preprocess_phase_pipelines
            .reset_indirect_batch_sets
            .prepare(
                &pipeline_cache,
                &mut specialized_reset_indirect_batch_sets_pipelines,
            );
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_indexed_indirect_params
            .prepare(
                &pipeline_cache,
                &mut specialized_build_indirect_parameters_pipelines,
                build_indirect_parameters_pipeline_key
                    | build_indirect_parameters_phase_pipeline_key
                    | BuildIndirectParametersPipelineKey::INDEXED
                    | BuildIndirectParametersPipelineKey::OCCLUSION_CULLING,
            );
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_non_indexed_indirect_params
            .prepare(
                &pipeline_cache,
                &mut specialized_build_indirect_parameters_pipelines,
                build_indirect_parameters_pipeline_key
                    | build_indirect_parameters_phase_pipeline_key
                    | BuildIndirectParametersPipelineKey::OCCLUSION_CULLING,
            );
    }

    // Prepare the uniform allocation compute pipeline.
    preprocess_pipelines.scene_unpacking.prepare(
        &pipeline_cache,
        &mut specialized_scene_unpacking_pipelines,
        &mut specialized_uniform_allocation_global_scan_pipelines,
        &mut specialized_uniform_allocation_fan_pipelines,
    );
}

impl PreprocessPipeline {
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<PreprocessPipeline>,
        key: PreprocessPipelineKey,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let preprocess_pipeline_id = pipelines.specialize(pipeline_cache, self, key);
        self.pipeline_id = Some(preprocess_pipeline_id);
    }
}

impl SpecializedComputePipeline for ResetIndirectBatchSetsPipeline {
    type Key = ();

    fn specialize(&self, _: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("reset indirect batch sets".into()),
            layout: vec![self.bind_group_layout.clone()],
            shader: self.shader.clone(),
            ..default()
        }
    }
}

impl SpecializedComputePipeline for BuildIndirectParametersPipeline {
    type Key = BuildIndirectParametersPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![];
        if key.contains(BuildIndirectParametersPipelineKey::INDEXED) {
            shader_defs.push("INDEXED".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::MULTI_DRAW_INDIRECT_COUNT_SUPPORTED) {
            shader_defs.push("MULTI_DRAW_INDIRECT_COUNT_SUPPORTED".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::OCCLUSION_CULLING) {
            shader_defs.push("OCCLUSION_CULLING".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::EARLY_PHASE) {
            shader_defs.push("EARLY_PHASE".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::LATE_PHASE) {
            shader_defs.push("LATE_PHASE".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::MAIN_PHASE) {
            shader_defs.push("MAIN_PHASE".into());
        }

        let label = format!(
            "{} build {}indexed indirect parameters",
            if !key.contains(BuildIndirectParametersPipelineKey::OCCLUSION_CULLING) {
                "frustum culling"
            } else if key.contains(BuildIndirectParametersPipelineKey::EARLY_PHASE) {
                "early occlusion culling"
            } else if key.contains(BuildIndirectParametersPipelineKey::LATE_PHASE) {
                "late occlusion culling"
            } else {
                "main occlusion culling"
            },
            if key.contains(BuildIndirectParametersPipelineKey::INDEXED) {
                ""
            } else {
                "non-"
            }
        );

        ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![self.bind_group_layout.clone()],
            shader: self.shader.clone(),
            shader_defs,
            ..default()
        }
    }
}

impl SpecializedComputePipeline for SceneUnpackingPipeline {
    type Key = ();

    fn specialize(&self, _: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("scene unpacking".into()),
            layout: vec![self.bind_group_layout.clone()],
            shader: self.shader.clone(),
            shader_defs: vec![],
            entry_point: Some("unpack_scene".into()),
            ..Default::default()
        }
    }
}

impl SpecializedComputePipeline for UniformAllocationGlobalScanPipeline {
    type Key = ();

    fn specialize(&self, _: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("uniform allocation, global scan".into()),
            layout: vec![self.bind_group_layout.clone()],
            shader: self.shader.clone(),
            shader_defs: vec![],
            entry_point: Some("allocate_global_scan".into()),
            ..Default::default()
        }
    }
}

impl SpecializedComputePipeline for UniformAllocationFanPipeline {
    type Key = ();

    fn specialize(&self, _: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("uniform allocation, fan".into()),
            layout: vec![self.bind_group_layout.clone()],
            shader: self.shader.clone(),
            shader_defs: vec![],
            entry_point: Some("allocate_fan".into()),
            ..Default::default()
        }
    }
}

impl ResetIndirectBatchSetsPipeline {
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let reset_indirect_batch_sets_pipeline_id = pipelines.specialize(pipeline_cache, self, ());
        self.pipeline_id = Some(reset_indirect_batch_sets_pipeline_id);
    }
}

impl BuildIndirectParametersPipeline {
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<BuildIndirectParametersPipeline>,
        key: BuildIndirectParametersPipelineKey,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let build_indirect_parameters_pipeline_id = pipelines.specialize(pipeline_cache, self, key);
        self.pipeline_id = Some(build_indirect_parameters_pipeline_id);
    }
}

impl SceneUnpackingPipelines {
    /// Specializes all three pipelines that use the uniform allocation shader.
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        scene_unpacking_pipelines: &mut SpecializedComputePipelines<SceneUnpackingPipeline>,
        uniform_allocation_global_scan_pipelines: &mut SpecializedComputePipelines<
            UniformAllocationGlobalScanPipeline,
        >,
        uniform_allocation_fan_pipelines: &mut SpecializedComputePipelines<
            UniformAllocationFanPipeline,
        >,
    ) {
        if self.local_scan.pipeline_id.is_none() {
            self.local_scan.pipeline_id =
                Some(scene_unpacking_pipelines.specialize(pipeline_cache, &self.local_scan, ()));
        }

        if self.global_scan.pipeline_id_global_scan.is_none() {
            self.global_scan.pipeline_id_global_scan =
                Some(uniform_allocation_global_scan_pipelines.specialize(
                    pipeline_cache,
                    &self.global_scan,
                    (),
                ));
        }

        if self.fan.pipeline_id_fan.is_none() {
            self.fan.pipeline_id_fan =
                Some(uniform_allocation_fan_pipelines.specialize(pipeline_cache, &self.fan, ()));
        }
    }
}

/// A system that attaches buffers to bind groups for the variants of the
/// compute shaders relating to mesh preprocessing.
#[expect(
    clippy::too_many_arguments,
    reason = "it's a system that needs a lot of arguments"
)]
pub fn prepare_preprocess_bind_groups(
    mut commands: Commands,
    views: Query<(Entity, &ExtractedView)>,
    view_depth_pyramids: Query<(&ViewDepthPyramid, &PreviousViewUniformOffset)>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    indirect_parameters_buffers: Res<IndirectParametersBuffers>,
    indirect_parameters_build_jobs: Res<IndirectParametersBuildJobs>,
    scene_unpacking_buffers: Res<SceneUnpackingBuffers>,
    mesh_culling_data_buffer: Res<MeshCullingDataBuffer>,
    visibility_ranges: Res<RenderVisibilityRanges>,
    view_uniforms: Res<ViewUniforms>,
    previous_view_uniforms: Res<PreviousViewUniforms>,
    pipelines: Res<PreprocessPipelines>,
    mut scene_unpacking_bind_groups: ResMut<SceneUnpackingBindGroups>,
) {
    scene_unpacking_bind_groups.clear();

    // Grab the `BatchedInstanceBuffers`.
    let BatchedInstanceBuffers {
        current_input_buffer: current_input_buffer_vec,
        previous_input_buffer: previous_input_buffer_vec,
        phase_instance_buffers,
    } = batched_instance_buffers.into_inner();

    let (Some(current_input_buffer), Some(previous_input_buffer)) = (
        current_input_buffer_vec.buffer().buffer(),
        previous_input_buffer_vec.buffer(),
    ) else {
        return;
    };

    // Record whether we have any meshes that are to be drawn indirectly. If we
    // don't, then we can skip building indirect parameters.
    let mut any_indirect = false;

    // Loop over each view.
    for (view_entity, view) in &views {
        let mut bind_groups = TypeIdHashMap::default();

        // Loop over each phase.
        for (phase_type_id, phase_instance_buffers) in phase_instance_buffers {
            let UntypedPhaseBatchedInstanceBuffers {
                data_buffer: ref data_buffer_vec,
                ref work_item_buffers,
                ref late_indexed_indirect_parameters_buffer,
                ref late_non_indexed_indirect_parameters_buffer,
            } = *phase_instance_buffers;

            let Some(data_buffer) = data_buffer_vec.buffer() else {
                continue;
            };

            // Grab the indirect parameters buffers for this phase.
            let Some(phase_indirect_parameters_buffers) =
                indirect_parameters_buffers.get(phase_type_id)
            else {
                continue;
            };

            let Some(work_item_buffers) = work_item_buffers.get(&view.retained_view_entity) else {
                continue;
            };

            // Create the `PreprocessBindGroupBuilder`.
            let preprocess_bind_group_builder = PreprocessBindGroupBuilder {
                view: view_entity,
                late_indexed_indirect_parameters_buffer,
                late_non_indexed_indirect_parameters_buffer,
                render_device: &render_device,
                pipeline_cache: &pipeline_cache,
                phase_indirect_parameters_buffers,
                mesh_culling_data_buffer: &mesh_culling_data_buffer,
                visibility_range_data_buffer: visibility_ranges.buffer(),
                view_uniforms: &view_uniforms,
                previous_view_uniforms: &previous_view_uniforms,
                pipelines: &pipelines,
                current_input_buffer,
                previous_input_buffer,
                data_buffer,
            };

            // Depending on the type of work items we have, construct the
            // appropriate bind groups.
            let (was_indirect, bind_group) = match *work_item_buffers {
                PreprocessWorkItemBuffers::Direct(ref work_item_buffer) => (
                    false,
                    preprocess_bind_group_builder
                        .create_direct_preprocess_bind_groups(work_item_buffer),
                ),

                PreprocessWorkItemBuffers::Indirect {
                    indexed: ref indexed_work_item_buffer,
                    non_indexed: ref non_indexed_work_item_buffer,
                    gpu_occlusion_culling: Some(ref gpu_occlusion_culling_work_item_buffers),
                } => (
                    true,
                    preprocess_bind_group_builder
                        .create_indirect_occlusion_culling_preprocess_bind_groups(
                            &view_depth_pyramids,
                            indexed_work_item_buffer,
                            non_indexed_work_item_buffer,
                            gpu_occlusion_culling_work_item_buffers,
                        ),
                ),

                PreprocessWorkItemBuffers::Indirect {
                    indexed: ref indexed_work_item_buffer,
                    non_indexed: ref non_indexed_work_item_buffer,
                    gpu_occlusion_culling: None,
                } => (
                    true,
                    preprocess_bind_group_builder
                        .create_indirect_frustum_culling_preprocess_bind_groups(
                            indexed_work_item_buffer,
                            non_indexed_work_item_buffer,
                        ),
                ),
            };

            // Write that bind group in.
            if let Some(bind_group) = bind_group {
                any_indirect = any_indirect || was_indirect;
                bind_groups.insert(*phase_type_id, bind_group);
            }
        }

        // Save the bind groups.
        commands
            .entity(view_entity)
            .insert(PreprocessBindGroups(bind_groups));
    }

    // Now, if there were any indirect draw commands, create the bind groups for
    // the indirect parameters building shader.
    if any_indirect {
        create_build_indirect_parameters_bind_groups(
            &mut commands,
            &render_device,
            &pipeline_cache,
            &pipelines,
            current_input_buffer,
            &indirect_parameters_buffers,
            &indirect_parameters_build_jobs,
        );
    }

    // Create the bind groups we'll need for each dispatch of the bin unpacking
    // and uniform allocation shader.
    for (_, view) in &views {
        create_scene_unpacking_bind_groups(
            &mut scene_unpacking_bind_groups,
            &render_device,
            &pipeline_cache,
            &pipelines,
            &indirect_parameters_buffers,
            phase_instance_buffers,
            &scene_unpacking_buffers,
            &view.retained_view_entity,
        );
    }
}

/// A temporary structure that stores all the information needed to construct
/// bind groups for the mesh preprocessing shader.
struct PreprocessBindGroupBuilder<'a> {
    /// The render-world entity corresponding to the current view.
    view: Entity,
    /// The indirect compute dispatch parameters buffer for indexed meshes in
    /// the late prepass.
    late_indexed_indirect_parameters_buffer:
        &'a RawBufferVec<LatePreprocessWorkItemIndirectParameters>,
    /// The indirect compute dispatch parameters buffer for non-indexed meshes
    /// in the late prepass.
    late_non_indexed_indirect_parameters_buffer:
        &'a RawBufferVec<LatePreprocessWorkItemIndirectParameters>,
    /// The device.
    render_device: &'a RenderDevice,
    /// The pipeline cache
    pipeline_cache: &'a PipelineCache,
    /// The buffers that store indirect draw parameters.
    phase_indirect_parameters_buffers: &'a UntypedPhaseIndirectParametersBuffers,
    /// The GPU buffer that stores the information needed to cull each mesh.
    mesh_culling_data_buffer: &'a MeshCullingDataBuffer,
    /// The device buffer that stores the information needed to process
    /// visibility ranges on the GPU.
    visibility_range_data_buffer: &'a BufferVec<Vec4>,
    /// The GPU buffer that stores information about the view.
    view_uniforms: &'a ViewUniforms,
    /// The GPU buffer that stores information about the view from last frame.
    previous_view_uniforms: &'a PreviousViewUniforms,
    /// The pipelines for the mesh preprocessing shader.
    pipelines: &'a PreprocessPipelines,
    /// The GPU buffer containing the list of [`MeshInputUniform`]s for the
    /// current frame.
    current_input_buffer: &'a Buffer,
    /// The GPU buffer containing the list of [`MeshInputUniform`]s for the
    /// previous frame.
    previous_input_buffer: &'a Buffer,
    /// The GPU buffer containing the list of [`MeshUniform`]s for the current
    /// frame.
    ///
    /// This is the buffer containing the mesh's final transforms that the
    /// shaders will write to.
    data_buffer: &'a Buffer,
}

impl<'a> PreprocessBindGroupBuilder<'a> {
    /// Creates the bind groups for mesh preprocessing when GPU frustum culling
    /// and GPU occlusion culling are both disabled.
    fn create_direct_preprocess_bind_groups(
        &self,
        work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
    ) -> Option<PhasePreprocessBindGroups> {
        // Don't use `as_entire_binding()` here; the shader reads the array
        // length and the underlying buffer may be longer than the actual size
        // of the vector.
        let work_item_buffer_size = NonZero::<u64>::try_from(
            work_item_buffer.len() as u64 * u64::from(PreprocessWorkItem::min_size()),
        )
        .ok();

        Some(PhasePreprocessBindGroups::Direct(
            self.render_device.create_bind_group(
                "preprocess_direct_bind_group",
                &self
                    .pipeline_cache
                    .get_bind_group_layout(&self.pipelines.direct_preprocess.bind_group_layout),
                &BindGroupEntries::with_indices((
                    (0, self.view_uniforms.uniforms.binding()?),
                    (3, self.current_input_buffer.as_entire_binding()),
                    (4, self.previous_input_buffer.as_entire_binding()),
                    (
                        5,
                        BindingResource::Buffer(BufferBinding {
                            buffer: work_item_buffer.buffer()?,
                            offset: 0,
                            size: work_item_buffer_size,
                        }),
                    ),
                    (6, self.data_buffer.as_entire_binding()),
                )),
            ),
        ))
    }

    /// Creates the bind groups for mesh preprocessing when GPU occlusion
    /// culling is enabled.
    fn create_indirect_occlusion_culling_preprocess_bind_groups(
        &self,
        view_depth_pyramids: &Query<(&ViewDepthPyramid, &PreviousViewUniformOffset)>,
        indexed_work_item_buffer: &PartialBufferVec<PreprocessWorkItem>,
        non_indexed_work_item_buffer: &PartialBufferVec<PreprocessWorkItem>,
        gpu_occlusion_culling_work_item_buffers: &GpuOcclusionCullingWorkItemBuffers,
    ) -> Option<PhasePreprocessBindGroups> {
        let GpuOcclusionCullingWorkItemBuffers {
            late_indexed: ref late_indexed_work_item_buffer,
            late_non_indexed: ref late_non_indexed_work_item_buffer,
            ..
        } = *gpu_occlusion_culling_work_item_buffers;

        let (view_depth_pyramid, previous_view_uniform_offset) =
            view_depth_pyramids.get(self.view).ok()?;

        Some(PhasePreprocessBindGroups::IndirectOcclusionCulling {
            early_indexed: self.create_indirect_occlusion_culling_early_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                indexed_work_item_buffer,
                late_indexed_work_item_buffer,
            ),

            early_non_indexed: self.create_indirect_occlusion_culling_early_non_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                non_indexed_work_item_buffer,
                late_non_indexed_work_item_buffer,
            ),

            late_indexed: self.create_indirect_occlusion_culling_late_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                late_indexed_work_item_buffer,
            ),

            late_non_indexed: self.create_indirect_occlusion_culling_late_non_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                late_non_indexed_work_item_buffer,
            ),
        })
    }

    /// Creates the bind group for the first phase of mesh preprocessing of
    /// indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_early_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        indexed_work_item_buffer: &PartialBufferVec<PreprocessWorkItem>,
        late_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let visibility_range_binding = self.visibility_range_data_buffer.binding()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;

        match (
            self.phase_indirect_parameters_buffers
                .indexed
                .metadata_buffer(),
            indexed_work_item_buffer.buffer(),
            late_indexed_work_item_buffer.buffer(),
            self.late_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(indexed_metadata_buffer),
                Some(indexed_work_item_gpu_buffer),
                Some(late_indexed_work_item_gpu_buffer),
                Some(late_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_early_indexed_gpu_occlusion_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .early_gpu_occlusion_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            // @group(0) @binding(3) var<storage> current_input:
                            // array<MeshInput>;
                            (3, self.current_input_buffer.as_entire_binding()),
                            // @group(0) @binding(4) var<storage>
                            // previous_input: array<MeshInput>;
                            (4, self.previous_input_buffer.as_entire_binding()),
                            // @group(0) @binding(5) var<storage> work_items:
                            // array<PreprocessWorkItem>;
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: indexed_work_item_buffer_size,
                                }),
                            ),
                            // @group(0) @binding(6) var<storage, read_write>
                            // output: array<Mesh>;
                            (6, self.data_buffer.as_entire_binding()),
                            // @group(0) @binding(7) var<storage>
                            // indirect_parameters_metadata:
                            // array<IndirectParametersMetadata>;
                            (7, indexed_metadata_buffer.as_entire_binding()),
                            // @group(0) @binding(9) var<storage>
                            // mesh_culling_data: array<MeshCullingData>;
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            // @group(0) @binding(10) var<storage>
                            // visibility_ranges: array<vec4<f32>>;
                            (10, visibility_range_binding.clone()),
                            // @group(0) @binding(0) var<uniform> view: View;
                            (0, view_uniforms_binding.clone()),
                            // @group(0) @binding(11) var depth_pyramid:
                            // texture_2d<f32>;
                            (11, &view_depth_pyramid.all_mips),
                            // @group(0) @binding(2) var<uniform>
                            // previous_view_uniforms: PreviousViewUniforms;
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            // @group(0) @binding(12) var<storage, read_write>
                            // late_preprocess_work_items:
                            // array<PreprocessWorkItem>;
                            (
                                12,
                                BufferBinding {
                                    buffer: late_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: indexed_work_item_buffer_size,
                                },
                            ),
                            // @group(0) @binding(13) var<storage, read_write>
                            // late_preprocess_work_item_indirect_parameters:
                            // array<LatePreprocessWorkItemIndirectParameters>;
                            (
                                13,
                                BufferBinding {
                                    buffer: late_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for the first phase of mesh preprocessing of
    /// non-indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_early_non_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        non_indexed_work_item_buffer: &PartialBufferVec<PreprocessWorkItem>,
        late_non_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let visibility_range_binding = self.visibility_range_data_buffer.binding()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;

        match (
            self.phase_indirect_parameters_buffers
                .non_indexed
                .metadata_buffer(),
            non_indexed_work_item_buffer.buffer(),
            late_non_indexed_work_item_buffer.buffer(),
            self.late_non_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(non_indexed_metadata_buffer),
                Some(non_indexed_work_item_gpu_buffer),
                Some(late_non_indexed_work_item_buffer),
                Some(late_non_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    non_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_early_non_indexed_gpu_occlusion_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .early_gpu_occlusion_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            // @group(0) @binding(3) var<storage> current_input:
                            // array<MeshInput>;
                            (3, self.current_input_buffer.as_entire_binding()),
                            // @group(0) @binding(4) var<storage>
                            // previous_input: array<MeshInput>;
                            (4, self.previous_input_buffer.as_entire_binding()),
                            // @group(0) @binding(5) var<storage> work_items:
                            // array<PreprocessWorkItem>;
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: non_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            // @group(0) @binding(7) var<storage>
                            // indirect_parameters_metadata:
                            // array<IndirectParametersMetadata>;
                            (7, non_indexed_metadata_buffer.as_entire_binding()),
                            // @group(0) @binding(9) var<storage>
                            // mesh_culling_data: array<MeshCullingData>;
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            // @group(0) @binding(10) var<storage>
                            // visibility_ranges: array<vec4<f32>>;
                            (10, visibility_range_binding.clone()),
                            // @group(0) @binding(0) var<uniform> view: View;
                            (0, view_uniforms_binding.clone()),
                            // @group(0) @binding(11) var depth_pyramid:
                            // texture_2d<f32>;
                            (11, &view_depth_pyramid.all_mips),
                            // @group(0) @binding(2) var<uniform>
                            // previous_view_uniforms: PreviousViewUniforms;
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            // @group(0) @binding(12) var<storage, read_write>
                            // late_preprocess_work_items:
                            // array<PreprocessWorkItem>;
                            (
                                12,
                                BufferBinding {
                                    buffer: late_non_indexed_work_item_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                },
                            ),
                            // @group(0) @binding(13) var<storage, read_write>
                            // late_preprocess_work_item_indirect_parameters:
                            // array<LatePreprocessWorkItemIndirectParameters>;
                            (
                                13,
                                BufferBinding {
                                    buffer: late_non_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_non_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for the second phase of mesh preprocessing of
    /// indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_late_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        late_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let visibility_range_binding = self.visibility_range_data_buffer.binding()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;

        match (
            self.phase_indirect_parameters_buffers
                .indexed
                .metadata_buffer(),
            late_indexed_work_item_buffer.buffer(),
            self.late_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(indexed_metadata_buffer),
                Some(late_indexed_work_item_gpu_buffer),
                Some(late_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let late_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    late_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_late_indexed_gpu_occlusion_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .late_gpu_occlusion_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            // @group(0) @binding(3) var<storage> current_input:
                            // array<MeshInput>;
                            (3, self.current_input_buffer.as_entire_binding()),
                            // @group(0) @binding(4) var<storage>
                            // previous_input: array<MeshInput>;
                            (4, self.previous_input_buffer.as_entire_binding()),
                            // @group(0) @binding(5) var<storage> work_items:
                            // array<PreprocessWorkItem>;
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: late_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: late_indexed_work_item_buffer_size,
                                }),
                            ),
                            // @group(0) @binding(6) var<storage, read_write>
                            // output: array<Mesh>;
                            (6, self.data_buffer.as_entire_binding()),
                            // @group(0) @binding(7) var<storage>
                            // indirect_parameters_metadata:
                            // array<IndirectParametersMetadata>;
                            (7, indexed_metadata_buffer.as_entire_binding()),
                            // @group(0) @binding(9) var<storage>
                            // mesh_culling_data: array<MeshCullingData>;
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            // @group(0) @binding(10) var<storage>
                            // visibility_ranges: array<vec4<f32>>;
                            (10, visibility_range_binding.clone()),
                            // @group(0) @binding(0) var<uniform> view: View;
                            (0, view_uniforms_binding.clone()),
                            // @group(0) @binding(11) var depth_pyramid:
                            // texture_2d<f32>;
                            (11, &view_depth_pyramid.all_mips),
                            // @group(0) @binding(2) var<uniform>
                            // previous_view_uniforms: PreviousViewUniforms;
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            // @group(0) @binding(13) var<storage, read_write>
                            // late_preprocess_work_item_indirect_parameters:
                            // array<LatePreprocessWorkItemIndirectParameters>;
                            (
                                13,
                                BufferBinding {
                                    buffer: late_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for the second phase of mesh preprocessing of
    /// non-indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_late_non_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        late_non_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let visibility_range_binding = self.visibility_range_data_buffer.binding()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;

        match (
            self.phase_indirect_parameters_buffers
                .non_indexed
                .metadata_buffer(),
            late_non_indexed_work_item_buffer.buffer(),
            self.late_non_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(non_indexed_metadata_buffer),
                Some(non_indexed_work_item_gpu_buffer),
                Some(late_non_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    late_non_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_late_non_indexed_gpu_occlusion_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .late_gpu_occlusion_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            // @group(0) @binding(3) var<storage> current_input:
                            // array<MeshInput>;
                            (3, self.current_input_buffer.as_entire_binding()),
                            // @group(0) @binding(4) var<storage>
                            // previous_input: array<MeshInput>;
                            (4, self.previous_input_buffer.as_entire_binding()),
                            // @group(0) @binding(5) var<storage> work_items:
                            // array<PreprocessWorkItem>;
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: non_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                }),
                            ),
                            // @group(0) @binding(6) var<storage, read_write>
                            // output: array<Mesh>;
                            (6, self.data_buffer.as_entire_binding()),
                            // @group(0) @binding(7) var<storage>
                            // indirect_parameters_metadata:
                            // array<IndirectParametersMetadata>;
                            (7, non_indexed_metadata_buffer.as_entire_binding()),
                            // @group(0) @binding(9) var<storage>
                            // mesh_culling_data: array<MeshCullingData>;
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            // @group(0) @binding(10) var<storage>
                            // visibility_ranges: array<vec4<f32>>;
                            (10, visibility_range_binding.clone()),
                            // @group(0) @binding(0) var<uniform> view: View;
                            (0, view_uniforms_binding.clone()),
                            // @group(0) @binding(11) var depth_pyramid:
                            // texture_2d<f32>;
                            (11, &view_depth_pyramid.all_mips),
                            // @group(0) @binding(2) var<uniform>
                            // previous_view_uniforms: PreviousViewUniforms;
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            // @group(0) @binding(13) var<storage, read>
                            // late_preprocess_work_item_indirect_parameters:
                            // array<LatePreprocessWorkItemIndirectParameters>;
                            (
                                13,
                                BufferBinding {
                                    buffer: late_non_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_non_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind groups for mesh preprocessing when GPU frustum culling
    /// is enabled, but GPU occlusion culling is disabled.
    fn create_indirect_frustum_culling_preprocess_bind_groups(
        &self,
        indexed_work_item_buffer: &PartialBufferVec<PreprocessWorkItem>,
        non_indexed_work_item_buffer: &PartialBufferVec<PreprocessWorkItem>,
    ) -> Option<PhasePreprocessBindGroups> {
        Some(PhasePreprocessBindGroups::IndirectFrustumCulling {
            indexed: self
                .create_indirect_frustum_culling_indexed_bind_group(indexed_work_item_buffer),
            non_indexed: self.create_indirect_frustum_culling_non_indexed_bind_group(
                non_indexed_work_item_buffer,
            ),
        })
    }

    /// Creates the bind group for mesh preprocessing of indexed meshes when GPU
    /// frustum culling is enabled, but GPU occlusion culling is disabled.
    fn create_indirect_frustum_culling_indexed_bind_group(
        &self,
        indexed_work_item_buffer: &PartialBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let visibility_range_binding = self.visibility_range_data_buffer.binding()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;

        match (
            self.phase_indirect_parameters_buffers
                .indexed
                .metadata_buffer(),
            indexed_work_item_buffer.buffer(),
        ) {
            (Some(indexed_metadata_buffer), Some(indexed_work_item_gpu_buffer)) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_gpu_indexed_frustum_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .gpu_frustum_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, indexed_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (10, visibility_range_binding.clone()),
                            (0, view_uniforms_binding.clone()),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for mesh preprocessing of non-indexed meshes when
    /// GPU frustum culling is enabled, but GPU occlusion culling is disabled.
    fn create_indirect_frustum_culling_non_indexed_bind_group(
        &self,
        non_indexed_work_item_buffer: &PartialBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let visibility_range_binding = self.visibility_range_data_buffer.binding()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;

        match (
            self.phase_indirect_parameters_buffers
                .non_indexed
                .metadata_buffer(),
            non_indexed_work_item_buffer.buffer(),
        ) {
            (Some(non_indexed_metadata_buffer), Some(non_indexed_work_item_gpu_buffer)) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    non_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_gpu_non_indexed_frustum_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .gpu_frustum_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            // @group(0) @binding(3) var<storage> current_input:
                            // array<MeshInput>;
                            (3, self.current_input_buffer.as_entire_binding()),
                            // @group(0) @binding(4) var<storage>
                            // previous_input: array<MeshInput>;
                            (4, self.previous_input_buffer.as_entire_binding()),
                            // @group(0) @binding(5) var<storage> work_items:
                            // array<PreprocessWorkItem>;
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: non_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                }),
                            ),
                            // @group(0) @binding(6) var<storage, read_write>
                            // output: array<Mesh>;
                            (6, self.data_buffer.as_entire_binding()),
                            // @group(0) @binding(7) var<storage>
                            // indirect_parameters_metadata:
                            // array<IndirectParametersMetadata>;
                            (7, non_indexed_metadata_buffer.as_entire_binding()),
                            // @group(0) @binding(9) var<storage>
                            // mesh_culling_data: array<MeshCullingData>;
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            // @group(0) @binding(10) var<storage>
                            // visibility_ranges: array<vec4<f32>>;
                            (10, visibility_range_binding.clone()),
                            // @group(0) @binding(0) var<uniform> view: View;
                            (0, view_uniforms_binding.clone()),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }
}

/// A system that creates bind groups from the indirect parameters metadata and
/// data buffers for the indirect batch set reset shader and the indirect
/// parameter building shader.
fn create_build_indirect_parameters_bind_groups(
    commands: &mut Commands,
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    pipelines: &PreprocessPipelines,
    current_input_buffer: &Buffer,
    indirect_parameters_buffers: &IndirectParametersBuffers,
    indirect_parameters_build_jobs: &IndirectParametersBuildJobs,
) {
    let mut build_indirect_parameters_bind_groups = BuildIndirectParametersBindGroups::new();

    for (phase_type_id, phase_indirect_parameters_buffer) in indirect_parameters_buffers.iter() {
        build_indirect_parameters_bind_groups.insert(
            *phase_type_id,
            PhaseBuildIndirectParametersBindGroups {
                reset_indexed_indirect_batch_sets: phase_indirect_parameters_buffer
                    .indexed
                    .batch_sets_buffer()
                    .map(|indexed_batch_sets_buffer| {
                        render_device.create_bind_group(
                            "reset_indexed_indirect_batch_sets_bind_group",
                            // The early bind group is good for the main phase and late
                            // phase too. They bind the same buffers.
                            &pipeline_cache.get_bind_group_layout(
                                &pipelines
                                    .early_phase
                                    .reset_indirect_batch_sets
                                    .bind_group_layout,
                            ),
                            &BindGroupEntries::sequential((
                                indexed_batch_sets_buffer.as_entire_binding(),
                            )),
                        )
                    }),

                reset_non_indexed_indirect_batch_sets: phase_indirect_parameters_buffer
                    .non_indexed
                    .batch_sets_buffer()
                    .map(|non_indexed_batch_sets_buffer| {
                        render_device.create_bind_group(
                            "reset_non_indexed_indirect_batch_sets_bind_group",
                            // The early bind group is good for the main phase and late
                            // phase too. They bind the same buffers.
                            &pipeline_cache.get_bind_group_layout(
                                &pipelines
                                    .early_phase
                                    .reset_indirect_batch_sets
                                    .bind_group_layout,
                            ),
                            &BindGroupEntries::sequential((
                                non_indexed_batch_sets_buffer.as_entire_binding(),
                            )),
                        )
                    }),

                build_indexed_indirect: match (
                    phase_indirect_parameters_buffer.indexed.metadata_buffer(),
                    phase_indirect_parameters_buffer.indexed.data_buffer(),
                    phase_indirect_parameters_buffer.indexed.batch_sets_buffer(),
                    indirect_parameters_build_jobs.buffer(),
                ) {
                    (
                        Some(indexed_indirect_parameters_metadata_buffer),
                        Some(indexed_indirect_parameters_data_buffer),
                        Some(indexed_batch_sets_buffer),
                        Some(indirect_parameters_build_job_buffer),
                    ) => Some(
                        render_device.create_bind_group(
                            "build_indexed_indirect_parameters_bind_group",
                            // The frustum culling bind group is good for occlusion culling
                            // too. They bind the same buffers.
                            &pipeline_cache.get_bind_group_layout(
                                &pipelines
                                    .gpu_frustum_culling_build_indexed_indirect_params
                                    .bind_group_layout,
                            ),
                            &BindGroupEntries::with_indices((
                                // @group(0) @binding(0) var<storage>
                                // current_input: array<MeshInput>;
                                (0, current_input_buffer.as_entire_binding()),
                                // @group(0) @binding(1) var<storage>
                                // indirect_parameters_metadata:
                                // array<IndirectParametersMetadata>;
                                (
                                    1,
                                    indexed_indirect_parameters_metadata_buffer.as_entire_binding(),
                                ),
                                // @group(0) @binding(3) var<storage,
                                // read_write> indirect_batch_sets:
                                // array<IndirectBatchSet>;
                                (3, indexed_batch_sets_buffer.as_entire_binding()),
                                // @group(0) @binding(4) var<uniform>
                                // indirect_parameters_build_job:
                                // IndirectParametersBuildJob;
                                (
                                    4,
                                    BindingResource::Buffer(BufferBinding {
                                        buffer: indirect_parameters_build_job_buffer,
                                        offset: 0,
                                        size: NonZeroU64::new(
                                            size_of::<IndirectParametersBuildJob>() as u64,
                                        ),
                                    }),
                                ),
                                // @group(0) @binding(5) var<storage,
                                // read_write> indirect_parameters:
                                // array<IndirectParametersIndexed>;
                                (
                                    5,
                                    indexed_indirect_parameters_data_buffer.as_entire_binding(),
                                ),
                            )),
                        ),
                    ),
                    _ => None,
                },

                build_non_indexed_indirect: match (
                    phase_indirect_parameters_buffer
                        .non_indexed
                        .metadata_buffer(),
                    phase_indirect_parameters_buffer.non_indexed.data_buffer(),
                    phase_indirect_parameters_buffer
                        .non_indexed
                        .batch_sets_buffer(),
                    indirect_parameters_build_jobs.buffer(),
                ) {
                    (
                        Some(non_indexed_indirect_parameters_metadata_buffer),
                        Some(non_indexed_indirect_parameters_data_buffer),
                        Some(non_indexed_batch_sets_buffer),
                        Some(indirect_parameters_build_job_buffer),
                    ) => Some(
                        render_device.create_bind_group(
                            "build_non_indexed_indirect_parameters_bind_group",
                            // The frustum culling bind group is good for occlusion culling
                            // too. They bind the same buffers.
                            &pipeline_cache.get_bind_group_layout(
                                &pipelines
                                    .gpu_frustum_culling_build_non_indexed_indirect_params
                                    .bind_group_layout,
                            ),
                            &BindGroupEntries::with_indices((
                                // @group(0) @binding(0) var<storage>
                                // current_input: array<MeshInput>;
                                (0, current_input_buffer.as_entire_binding()),
                                // @group(0) @binding(1) var<storage>
                                // indirect_parameters_metadata:
                                // array<IndirectParametersMetadata>;
                                //
                                // Don't use `as_entire_binding` here; the shader reads
                                // the length and `RawBufferVec` overallocates.
                                (
                                    1,
                                    BufferBinding {
                                        buffer: non_indexed_indirect_parameters_metadata_buffer,
                                        offset: 0,
                                        size: NonZeroU64::new(
                                            phase_indirect_parameters_buffer
                                                .non_indexed
                                                .batch_count()
                                                as u64
                                                * size_of::<IndirectParametersMetadata>() as u64,
                                        ),
                                    },
                                ),
                                // @group(0) @binding(3) var<storage,
                                // read_write> indirect_batch_sets:
                                // array<IndirectBatchSet>;
                                (3, non_indexed_batch_sets_buffer.as_entire_binding()),
                                // @group(0) @binding(4) var<uniform>
                                // indirect_parameters_build_job:
                                // IndirectParametersBuildJob;
                                (
                                    4,
                                    BindingResource::Buffer(BufferBinding {
                                        buffer: indirect_parameters_build_job_buffer,
                                        offset: 0,
                                        size: NonZeroU64::new(
                                            size_of::<IndirectParametersBuildJob>() as u64,
                                        ),
                                    }),
                                ),
                                // @group(0) @binding(5) var<storage,
                                // read_write> indirect_parameters:
                                // array<IndirectParametersNonIndexed>;
                                (
                                    5,
                                    non_indexed_indirect_parameters_data_buffer.as_entire_binding(),
                                ),
                            )),
                        ),
                    ),
                    _ => None,
                },
            },
        );
    }

    commands.insert_resource(build_indirect_parameters_bind_groups);
}

/// Creates scene-unpacking bindings for every prepared phase of one view.
fn create_scene_unpacking_bind_groups(
    groups: &mut SceneUnpackingBindGroups,
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    pipelines: &PreprocessPipelines,
    indirect_parameters_buffers: &IndirectParametersBuffers,
    phase_instance_buffers: &TypeIdHashMap<UntypedPhaseBatchedInstanceBuffers<MeshUniform>>,
    scene_unpacking_buffers: &SceneUnpackingBuffers,
    view_entity: &RetainedViewEntity,
) {
    let Some(metadata_buffer) = scene_unpacking_buffers.metadata.buffer() else {
        return;
    };

    for (phase, indirect_buffers) in indirect_parameters_buffers.iter() {
        let key = SceneUnpackingBuffersKey {
            phase: *phase,
            view: *view_entity,
        };
        let Some(jobs) = scene_unpacking_buffers.view_phase_buffers.get(&key) else {
            continue;
        };
        let Some(work_item_buffers) = phase_instance_buffers
            .get(phase)
            .and_then(|buffers| buffers.work_item_buffers.get(view_entity))
        else {
            continue;
        };
        let (indexed_work_items, non_indexed_work_items) = match work_item_buffers {
            PreprocessWorkItemBuffers::Direct(buffer) => (buffer.buffer(), buffer.buffer()),
            PreprocessWorkItemBuffers::Indirect {
                indexed,
                non_indexed,
                ..
            } => (indexed.buffer(), non_indexed.buffer()),
        };
        let mut bindings = ViewPhaseSceneUnpackingBindGroups {
            indexed: Vec::with_capacity(jobs.indexed_unpacking_jobs.len()),
            non_indexed: Vec::with_capacity(jobs.non_indexed_unpacking_jobs.len()),
        };
        for (jobs, indirect_metadata, work_items, output) in [
            (
                &jobs.indexed_unpacking_jobs,
                indirect_buffers.indexed.metadata_buffer(),
                indexed_work_items,
                &mut bindings.indexed,
            ),
            (
                &jobs.non_indexed_unpacking_jobs,
                indirect_buffers.non_indexed.metadata_buffer(),
                non_indexed_work_items,
                &mut bindings.non_indexed,
            ),
        ] {
            let (Some(indirect_metadata), Some(work_items)) = (indirect_metadata, work_items)
            else {
                continue;
            };
            for job in jobs {
                let bind_group = render_device.create_bind_group(
                    "scene unpacking bind group",
                    &pipeline_cache.get_bind_group_layout(
                        &pipelines.scene_unpacking.local_scan.bind_group_layout,
                    ),
                    &BindGroupEntries::sequential((
                        BindingResource::Buffer(BufferBinding {
                            buffer: metadata_buffer,
                            offset: 0,
                            size: NonZeroU64::new(size_of::<GpuSceneUnpackingMetadata>() as u64),
                        }),
                        job.bin_metadata_buffer.as_entire_binding(),
                        indirect_metadata.as_entire_binding(),
                        job.fan_buffer.as_entire_binding(),
                        job.render_binned_mesh_instance_buffer.as_entire_binding(),
                        work_items.as_entire_binding(),
                        job.bin_index_to_bin_metadata_index_buffer
                            .as_entire_binding(),
                    )),
                );
                output.push(ViewPhaseSceneUnpackingBindGroup {
                    metadata_index: job.metadata_index,
                    bind_group,
                    bin_count: job.bin_count,
                    mesh_instance_count: job.mesh_instance_count,
                });
            }
        }
        groups.insert(key, bindings);
    }
}

/// Writes the information needed to do GPU mesh culling to the GPU.
pub fn write_mesh_culling_data_buffer(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut mesh_culling_data_buffer: ResMut<MeshCullingDataBuffer>,
    pipeline_cache: Res<PipelineCache>,
    mut sparse_buffer_update_jobs: ResMut<SparseBufferUpdateJobs>,
    mut sparse_buffer_update_bind_groups: ResMut<SparseBufferUpdateBindGroups>,
    sparse_buffer_update_pipelines: Res<SparseBufferUpdatePipelines>,
) {
    mesh_culling_data_buffer.write_buffers(&render_device, &render_queue);
    mesh_culling_data_buffer.prepare_to_populate_buffers(
        &render_device,
        &pipeline_cache,
        &mut sparse_buffer_update_jobs,
        &mut sparse_buffer_update_bind_groups,
        &sparse_buffer_update_pipelines,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::{AssetApp, AssetPlugin, AssetServer, Assets};
    use bevy_platform::future::block_on;
    use bevy_render::{
        occlusion_culling::OcclusionCullingPlugin,
        render_resource::{
            BufferDescriptor, BufferInitDescriptor, BufferUsages, CommandEncoderDescriptor,
            MapMode, PollType,
        },
        renderer::RenderInstance,
    };
    use bevy_shader::ShaderLoader;
    use bevy_tasks::{AsyncComputeTaskPool, ComputeTaskPool, IoTaskPool, TaskPool};
    use bytemuck::{bytes_of, cast_slice};
    use nonmax::NonMaxU32;
    use std::{
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };
    use wgpu_types::DeviceDescriptor;

    struct UnpackingGpu {
        device: RenderDevice,
        queue: RenderQueue,
        cache: PipelineCache,
        layout: BindGroupLayoutDescriptor,
        pipelines: [CachedComputePipelineId; 3],
    }

    impl UnpackingGpu {
        fn new() -> Self {
            let instance = RenderInstance::new(default());
            let adapter = block_on(instance.request_adapter(&default()))
                .expect("scene unpacking regression requires a graphics adapter");
            let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
                required_limits: adapter.limits(),
                ..default()
            }))
            .expect("scene unpacking device");
            let device = RenderDevice::new(device);
            IoTaskPool::get_or_init(TaskPool::new);
            AsyncComputeTaskPool::get_or_init(TaskPool::new);
            ComputeTaskPool::get_or_init(TaskPool::new);
            let mut app = App::new();
            app.add_plugins(AssetPlugin::default())
                .init_asset::<Shader>()
                .init_asset_loader::<ShaderLoader>()
                .add_plugins(OcclusionCullingPlugin);
            GpuMeshPreprocessPlugin {
                use_gpu_instance_buffer_builder: true,
            }
            .build(&mut app);
            app.init_resource::<PreprocessPipelines>();
            let root = app
                .world()
                .resource::<PreprocessPipelines>()
                .scene_unpacking
                .local_scan
                .shader
                .clone();
            let library: Handle<Shader> = app
                .world()
                .resource::<AssetServer>()
                .load("embedded://bevy_render/occlusion_culling/mesh_preprocess_types.wesl");
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                app.update();
                let shaders = app.world().resource::<Assets<Shader>>();
                if shaders.contains(root.id()) && shaders.contains(library.id()) {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "embedded scene-unpacking shaders did not load"
                );
                thread::yield_now();
            }
            let mut cache = PipelineCache::new(device.clone(), true);
            for id in [root.id(), library.id()] {
                let shader = app
                    .world_mut()
                    .resource_mut::<Assets<Shader>>()
                    .remove(id)
                    .expect("loaded embedded shader");
                cache.set_shader(id, shader);
            }
            let pipelines = app.world().resource::<PreprocessPipelines>();
            let layout = pipelines
                .scene_unpacking
                .local_scan
                .bind_group_layout
                .clone();
            let ids = [
                cache.queue_compute_pipeline(pipelines.scene_unpacking.local_scan.specialize(())),
                cache.queue_compute_pipeline(pipelines.scene_unpacking.global_scan.specialize(())),
                cache.queue_compute_pipeline(pipelines.scene_unpacking.fan.specialize(())),
            ];
            cache.process_queue();
            for id in ids {
                assert!(
                    cache.get_compute_pipeline(id).is_some(),
                    "scene-unpacking pipeline: {:?}",
                    cache.get_compute_pipeline_state(id)
                );
            }
            Self {
                device,
                queue: RenderQueue::new(queue),
                cache,
                layout,
                pipelines: ids,
            }
        }

        fn buffer(&self, data: &[u32], usage: BufferUsages) -> Buffer {
            self.device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("scene-unpacking regression"),
                contents: cast_slice(data),
                usage,
            })
        }

        fn run(
            &self,
            counts: &[u32],
            instances: &[[u32; 2]],
            expected_allocations: &[(usize, u32)],
            expected_work_items: &[(usize, [u32; 2])],
        ) {
            let bin_count = counts.len() as u32;
            let bins: Vec<_> = counts
                .iter()
                .enumerate()
                .flat_map(|(index, count)| [2 * index as u32, 2 * index as u32 + 1, *count])
                .collect();
            let bin_metadata = self.buffer(&bins, BufferUsages::STORAGE);
            let mut mapping = vec![0xdead_beef; counts.len() * 2];
            for index in 0..counts.len() {
                mapping[index * 2 + 1] = index as u32;
            }
            let mapping = self.buffer(&mapping, BufferUsages::STORAGE);
            let instances_buffer = self.buffer(cast_slice(instances), BufferUsages::STORAGE);
            let fan = self.buffer(&vec![0; counts.len().div_ceil(256)], BufferUsages::STORAGE);
            let indirect_initial = vec![0xdead_beef; (counts.len() * 2 + 4) * 5];
            let work_initial = vec![0xdead_beef; (instances.len() + 6) * 2];
            let output_usage =
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
            let indirect = self.buffer(&indirect_initial, output_usage);
            let work = self.buffer(&work_initial, output_usage);
            let metadata = self.buffer(&[0; 128], BufferUsages::UNIFORM | BufferUsages::COPY_DST);
            let binding = self.device.create_bind_group(
                "scene-unpacking regression",
                &self.cache.get_bind_group_layout(&self.layout),
                &BindGroupEntries::sequential((
                    BindingResource::Buffer(BufferBinding {
                        buffer: &metadata,
                        offset: 0,
                        size: NonZeroU64::new(256),
                    }),
                    bin_metadata.as_entire_binding(),
                    indirect.as_entire_binding(),
                    fan.as_entire_binding(),
                    instances_buffer.as_entire_binding(),
                    work.as_entire_binding(),
                    mapping.as_entire_binding(),
                )),
            );
            let bindings = ViewPhaseSceneUnpackingBindGroup {
                metadata_index: SceneUnpackingMetadataIndex(
                    NonMaxU32::new(1).expect("metadata index"),
                ),
                bind_group: binding,
                bin_count,
                mesh_instance_count: instances.len() as u32,
            };
            let indirect_size = (indirect_initial.len() * 4) as u64;
            let work_size = (work_initial.len() * 4) as u64;
            let readback = self.device.create_buffer(&BufferDescriptor {
                label: Some("scene-unpacking readback"),
                size: indirect_size + work_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            for (base, batch) in [(100, 11), (1000, 22)] {
                self.queue.write_buffer(
                    &metadata,
                    256,
                    bytes_of(&GpuSceneUnpackingMetadata {
                        batch_set_index: batch,
                        bin_count,
                        first_indirect_parameters_index: 2,
                        first_output_mesh_uniform_index: base,
                        base_output_work_item_index: 3,
                        base_indirect_parameters_index: 2,
                        binned_mesh_instance_count: instances.len() as u32,
                        pad: [0; 57],
                    }),
                );
                self.queue
                    .write_buffer(&indirect, 0, cast_slice(&indirect_initial));
                self.queue.write_buffer(&work, 0, cast_slice(&work_initial));
                let mut encoder = self
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    dispatch_scene_unpacking(
                        &mut pass,
                        self.cache
                            .get_compute_pipeline(self.pipelines[0])
                            .expect("local scan"),
                        self.cache
                            .get_compute_pipeline(self.pipelines[1])
                            .expect("global scan"),
                        self.cache
                            .get_compute_pipeline(self.pipelines[2])
                            .expect("fan"),
                        &bindings,
                    );
                }
                encoder.copy_buffer_to_buffer(&indirect, 0, &readback, 0, indirect_size);
                encoder.copy_buffer_to_buffer(&work, 0, &readback, indirect_size, work_size);
                self.queue.submit([encoder.finish()]);
                let slice = readback.slice(..);
                let (sender, receiver) = mpsc::channel();
                slice.map_async(MapMode::Read, move |result| {
                    sender.send(result).expect("readback receiver");
                });
                self.device
                    .poll(PollType::wait_indefinitely())
                    .expect("GPU completion");
                receiver
                    .recv_timeout(Duration::from_secs(10))
                    .expect("readback callback")
                    .expect("mapped output");
                let mapped = slice
                    .get_mapped_range()
                    .expect("mapped scene-unpacking output");
                let words: &[u32] = cast_slice(&mapped);
                for &(bin, offset) in expected_allocations {
                    let start = (2 + bin * 2) * 5;
                    assert_eq!(
                        &words[start..start + 5],
                        &[base + offset, batch, 0, 0, 0],
                        "bin {bin}, first uniform {base}"
                    );
                }
                for bin in 0..counts.len() {
                    let hole = (3 + bin * 2) * 5;
                    assert_eq!(
                        &words[hole..hole + 5],
                        &[0xdead_beef; 5],
                        "unused indirect slot"
                    );
                }
                let work_words = &words[indirect_initial.len()..];
                assert_eq!(&work_words[..6], &[0xdead_beef; 6], "work-item prefix");
                assert_eq!(
                    &work_words[work_words.len() - 6..],
                    &[0xdead_beef; 6],
                    "work-item suffix"
                );
                for &(index, expected) in expected_work_items {
                    let start = (3 + index) * 2;
                    assert_eq!(
                        &work_words[start..start + 2],
                        &expected,
                        "work item {index}"
                    );
                }
                drop(mapped);
                readback.unmap();
            }
        }
    }

    #[test]
    #[ignore = "requires a graphics adapter"]
    fn scene_unpacking_allocates_and_unpacks_independent_ranges() {
        let gpu = UnpackingGpu::new();
        let sparse = [
            [101, 5],
            [202, 1],
            [303, 3],
            [404, 5],
            [505, 1],
            [606, 5],
            [707, 5],
            [808, 3],
            [909, 1],
            [1001, 5],
            [1002, 1],
            [1003, 5],
            [1004, 1],
            [1005, 5],
        ];
        gpu.run(
            &[5, 2, 7],
            &sparse,
            &[(0, 0), (1, 5), (2, 7)],
            &[(0, [101, 6]), (1, [202, 2]), (2, [303, 4]), (13, [1005, 6])],
        );

        let single: Vec<_> = (0..4097).map(|index| [60_000 + index, 1]).collect();
        gpu.run(
            &[4097],
            &single,
            &[(0, 0)],
            &[(0, [60_000, 2]), (256, [60_256, 2]), (4096, [64_096, 2])],
        );

        for (count, allocations, work_items) in [
            (
                257,
                vec![(0, 0), (255, 255), (256, 256)],
                vec![(0, [70_000, 2]), (255, [70_255, 512]), (256, [70_256, 514])],
            ),
            (
                65_537,
                vec![
                    (0, 0),
                    (255, 255),
                    (256, 256),
                    (65_535, 65_535),
                    (65_536, 65_536),
                ],
                vec![
                    (0, [70_000, 2]),
                    (65_535, [135_535, 131_072]),
                    (65_536, [135_536, 131_074]),
                ],
            ),
        ] {
            let instances: Vec<_> = (0..count)
                .map(|index| [70_000 + index, index * 2 + 1])
                .collect();
            gpu.run(
                &vec![1; count as usize],
                &instances,
                &allocations,
                &work_items,
            );
        }
    }
}
