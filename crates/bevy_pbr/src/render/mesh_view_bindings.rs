use crate::{
    AreaLightLuts, DfgLut, ScreenSpaceTransmission, ViewFogUniformOffset,
    ViewLightProbesUniformOffset, ViewLightsUniformOffset, ViewScreenSpaceReflectionsUniformOffset,
};
use arrayvec::ArrayVec;
use bevy_core_pipeline::{
    oit::{
        OitBuffers, OrderIndependentTransparencySettings,
        OrderIndependentTransparencySettingsOffset,
    },
    prepass::ViewPrepassTextures,
    tonemapping::{
        get_lut_bind_group_layout_entries, get_lut_bindings, Tonemapping, TonemappingLuts,
    },
};
use bevy_ecs::{
    component::Component,
    entity::{Entity, EntityHashMap},
    query::Has,
    resource::Resource,
    system::{Commands, Local, Query, Res},
    world::{FromWorld, World},
};
use bevy_light::{EnvironmentMapLight, IrradianceVolume};
use bevy_math::Vec4;
use bevy_platform::sync::Arc;
use bevy_render::{
    camera::ExtractedCamera,
    globals::{GlobalsBuffer, GlobalsUniform},
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderAdapter, RenderDevice},
    texture::{FallbackImage, FallbackImageZero, GpuImage},
    view::{
        Msaa, RenderVisibilityRanges, ViewUniform, ViewUniformOffset, ViewUniforms,
        VISIBILITY_RANGES_STORAGE_BUFFER_COUNT, VISIBILITY_RANGE_UNIFORM_BUFFER_SIZE,
    },
};
use core::num::NonZero;
use core::{array::from_fn, fmt::Write};
use smallvec::SmallVec;

use crate::{
    contact_shadows::{
        ContactShadowsBuffer, ContactShadowsUniform, ViewContactShadowsUniformOffset,
    },
    decal::{
        self,
        clustered::{
            DecalsBuffer, RenderClusteredDecals, RenderViewClusteredDecalBindGroupEntries,
        },
    },
    environment_map::{self, RenderViewEnvironmentMapBindGroupEntries},
    irradiance_volume::{self, RenderViewIrradianceVolumeBindGroupEntries},
    prepass,
    resources::{AtmosphereBuffer, AtmosphereTextures, GpuAtmosphere},
    Bluenoise, ExtractedAtmosphere, FogMeta, GlobalClusterableObjectMeta, GpuClusteredLights,
    GpuFog, GpuLights, LightMeta, LightProbesBuffer, LightProbesUniform, MeshPipeline,
    MeshPipelineKey, RenderViewLightProbes, ScreenSpaceAmbientOcclusionResources,
    ScreenSpaceReflectionsBuffer, ScreenSpaceReflectionsUniform, ShadowSamplers,
    ViewClusterBindings, ViewShadowBindings, ViewTransmissionTexture,
    CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT,
};

#[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
use bevy_render::render_resource::binding_types::texture_cube;

#[cfg(debug_assertions)]
use {crate::MESH_PIPELINE_VIEW_LAYOUT_SAFE_MAX_TEXTURES, bevy_utils::once, tracing::warn};

pub const VISIBILITY_RANGES_BINDING_INDEX: u32 = 11;
pub const TONEMAPPING_LUT_TEXTURE_BINDING_INDEX: u32 = 15;
pub const TONEMAPPING_LUT_SAMPLER_BINDING_INDEX: u32 = 16;
pub const LINEAR_SAMPLER_BINDING_INDEX: u32 = 22;

/// A `Linear/ClampToEdge` sampler shared across multiple view bindings
#[derive(Resource)]
pub struct LinearSampler(pub Sampler);

impl FromWorld for LinearSampler {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        Self(render_device.create_sampler(&SamplerDescriptor {
            label: Some("linear_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        }))
    }
}

#[derive(Clone)]
pub struct MeshPipelineViewLayout {
    pub main_layout: BindGroupLayoutDescriptor,
    pub binding_array_layout: BindGroupLayoutDescriptor,
    pub empty_layout: BindGroupLayoutDescriptor,
}

bitflags::bitflags! {
    /// A key that uniquely identifies a [`MeshPipelineViewLayout`].
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct MeshPipelineViewLayoutKey: u32 {
        const MULTISAMPLED                     = 1 << 0;
        const DEPTH_PREPASS                    = 1 << 1;
        const NORMAL_PREPASS                   = 1 << 2;
        const MOTION_VECTOR_PREPASS            = 1 << 3;
        const DEFERRED_PREPASS                 = 1 << 4;
        const OIT_ENABLED                      = 1 << 5;
        const ATMOSPHERE                       = 1 << 6;
        const STBN                             = 1 << 7;
        const TONEMAP_IN_SHADER                = 1 << 8;
        const ENVIRONMENT_MAP                  = 1 << 9;
        const SCREEN_SPACE_AMBIENT_OCCLUSION   = 1 << 10;
        const IRRADIANCE_VOLUME                = 1 << 11;
        const SCREEN_SPACE_REFLECTIONS         = 1 << 12;
        const CONTACT_SHADOWS                  = 1 << 13;
        const DISTANCE_FOG                     = 1 << 14;
        const AREA_LIGHT_LUTS                  = 1 << 15;
        const VIEW_TRANSMISSION_TEXTURE        = 1 << 16;
    }
}

impl MeshPipelineViewLayoutKey {
    /// Builds a unique label for each layout based on the flags
    pub fn label(&self) -> String {
        let iter = self
            .iter_names()
            .filter(|(_, key)| self.contains(*key))
            .map(|(name, _)| name.to_lowercase());
        let (lower, _) = iter.size_hint();
        let sep = ",";
        let mut result = String::with_capacity(sep.len() * lower);

        for (i, name) in iter.enumerate() {
            if i != 0 {
                result.push_str(sep);
            }
            write!(&mut result, "{}", name).unwrap();
        }

        format!("mesh_view_layout:{}", result)
    }
}

impl From<MeshPipelineKey> for MeshPipelineViewLayoutKey {
    fn from(value: MeshPipelineKey) -> Self {
        let mut result = MeshPipelineViewLayoutKey::empty();

        if value.msaa_samples() > 1 {
            result |= MeshPipelineViewLayoutKey::MULTISAMPLED;
        }
        if value.contains(MeshPipelineKey::DEPTH_PREPASS) {
            result |= MeshPipelineViewLayoutKey::DEPTH_PREPASS;
        }
        if value.contains(MeshPipelineKey::NORMAL_PREPASS) {
            result |= MeshPipelineViewLayoutKey::NORMAL_PREPASS;
        }
        if value.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
            result |= MeshPipelineViewLayoutKey::MOTION_VECTOR_PREPASS;
        }
        if value.contains(MeshPipelineKey::DEFERRED_PREPASS) {
            result |= MeshPipelineViewLayoutKey::DEFERRED_PREPASS;
        }
        if value.contains(MeshPipelineKey::OIT_ENABLED) {
            result |= MeshPipelineViewLayoutKey::OIT_ENABLED;
        }
        if value.contains(MeshPipelineKey::ATMOSPHERE) {
            result |= MeshPipelineViewLayoutKey::ATMOSPHERE;
        }

        if cfg!(feature = "bluenoise_texture") {
            result |= MeshPipelineViewLayoutKey::STBN;
        }

        if cfg!(feature = "area_light_luts") {
            result |= MeshPipelineViewLayoutKey::AREA_LIGHT_LUTS;
        }

        if value.contains(MeshPipelineKey::TONEMAP_IN_SHADER) {
            result |= MeshPipelineViewLayoutKey::TONEMAP_IN_SHADER;
        }
        if value.contains(MeshPipelineKey::ENVIRONMENT_MAP) {
            result |= MeshPipelineViewLayoutKey::ENVIRONMENT_MAP;
        }
        if value.contains(MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION) {
            result |= MeshPipelineViewLayoutKey::SCREEN_SPACE_AMBIENT_OCCLUSION;
        }
        if value.contains(MeshPipelineKey::IRRADIANCE_VOLUME) {
            result |= MeshPipelineViewLayoutKey::IRRADIANCE_VOLUME;
        }
        if value.contains(MeshPipelineKey::SCREEN_SPACE_REFLECTIONS) {
            result |= MeshPipelineViewLayoutKey::SCREEN_SPACE_REFLECTIONS;
        }
        if value.contains(MeshPipelineKey::CONTACT_SHADOWS) {
            result |= MeshPipelineViewLayoutKey::CONTACT_SHADOWS;
        }
        if value.contains(MeshPipelineKey::DISTANCE_FOG) {
            result |= MeshPipelineViewLayoutKey::DISTANCE_FOG;
        }
        if value.contains(MeshPipelineKey::VIEW_TRANSMISSION_TEXTURE) {
            result |= MeshPipelineViewLayoutKey::VIEW_TRANSMISSION_TEXTURE;
        }

        result
    }
}

impl From<Msaa> for MeshPipelineViewLayoutKey {
    fn from(value: Msaa) -> Self {
        let mut result = MeshPipelineViewLayoutKey::empty();

        if value.samples() > 1 {
            result |= MeshPipelineViewLayoutKey::MULTISAMPLED;
        }

        result
    }
}

impl From<Option<&ViewPrepassTextures>> for MeshPipelineViewLayoutKey {
    fn from(value: Option<&ViewPrepassTextures>) -> Self {
        let mut result = MeshPipelineViewLayoutKey::empty();

        if let Some(prepass_textures) = value {
            if prepass_textures.depth.is_some() {
                result |= MeshPipelineViewLayoutKey::DEPTH_PREPASS;
            }
            if prepass_textures.normal.is_some() {
                result |= MeshPipelineViewLayoutKey::NORMAL_PREPASS;
            }
            if prepass_textures.motion_vectors.is_some() {
                result |= MeshPipelineViewLayoutKey::MOTION_VECTOR_PREPASS;
            }
            if prepass_textures.deferred.is_some() {
                result |= MeshPipelineViewLayoutKey::DEFERRED_PREPASS;
            }
        }

        result
    }
}

pub(crate) fn buffer_layout(
    buffer_binding_type: BufferBindingType,
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZero<u64>>,
) -> BindGroupLayoutEntryBuilder {
    match buffer_binding_type {
        BufferBindingType::Uniform => uniform_buffer_sized(has_dynamic_offset, min_binding_size),
        BufferBindingType::Storage { read_only } => {
            if read_only {
                storage_buffer_read_only_sized(has_dynamic_offset, min_binding_size)
            } else {
                storage_buffer_sized(has_dynamic_offset, min_binding_size)
            }
        }
    }
}

/// The minimum size of the visibility ranges binding: one element of the
/// runtime-sized storage array, or the whole fixed-size uniform array.
pub(crate) fn visibility_ranges_min_binding_size(
    buffer_binding_type: BufferBindingType,
) -> NonZero<u64> {
    match buffer_binding_type {
        BufferBindingType::Uniform => <[Vec4; VISIBILITY_RANGE_UNIFORM_BUFFER_SIZE]>::min_size(),
        BufferBindingType::Storage { .. } => Vec4::min_size(),
    }
}

/// Returns the appropriate bind group layout vec based on the parameters
fn layout_entries(
    layout_key: MeshPipelineViewLayoutKey,
    &MeshPipelineViewLayoutParams {
        clustered_forward_buffer_binding_type,
        visibility_ranges_buffer_binding_type,
        environment_map_entries,
        irradiance_volume_entries,
        clustered_decal_entries,
        is_oit_supported,
    }: &MeshPipelineViewLayoutParams,
) -> [Vec<BindGroupLayoutEntry>; 2] {
    let mut entries = DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::FRAGMENT,
        (
            // View
            (
                0,
                uniform_buffer::<ViewUniform>(true).visibility(ShaderStages::VERTEX_FRAGMENT),
            ),
            // Lights
            (1, uniform_buffer::<GpuLights>(true)),
            // Point Shadow Texture Cube Array
            (
                2,
                #[cfg(all(
                    not(target_abi = "sim"),
                    any(
                        not(feature = "webgl"),
                        not(target_arch = "wasm32"),
                        feature = "webgpu"
                    )
                ))]
                texture_cube_array(TextureSampleType::Depth),
                #[cfg(any(
                    target_abi = "sim",
                    all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu"))
                ))]
                texture_cube(TextureSampleType::Depth),
            ),
            // Point Shadow Texture Array Comparison Sampler
            (3, sampler(SamplerBindingType::Comparison)),
            // Directional Shadow Texture Array
            (
                4,
                #[cfg(any(
                    not(feature = "webgl"),
                    not(target_arch = "wasm32"),
                    feature = "webgpu"
                ))]
                texture_2d_array(TextureSampleType::Depth),
                #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
                texture_2d(TextureSampleType::Depth),
            ),
            // PointLights
            (
                5,
                buffer_layout(
                    clustered_forward_buffer_binding_type,
                    false,
                    Some(GpuClusteredLights::min_size(
                        clustered_forward_buffer_binding_type,
                    )),
                ),
            ),
            // ClusteredLightIndexLists
            (
                6,
                buffer_layout(
                    clustered_forward_buffer_binding_type,
                    false,
                    Some(
                        ViewClusterBindings::min_size_clusterable_object_index_lists(
                            clustered_forward_buffer_binding_type,
                        ),
                    ),
                ),
            ),
            // ClusterOffsetsAndCounts
            (
                7,
                buffer_layout(
                    clustered_forward_buffer_binding_type,
                    false,
                    Some(ViewClusterBindings::min_size_cluster_offsets_and_counts(
                        clustered_forward_buffer_binding_type,
                    )),
                ),
            ),
            // Globals
            (
                8,
                uniform_buffer::<GlobalsUniform>(false).visibility(ShaderStages::VERTEX_FRAGMENT),
            ),
            // Light probes
            (9, uniform_buffer::<LightProbesUniform>(true)),
            // Visibility ranges
            (
                VISIBILITY_RANGES_BINDING_INDEX,
                buffer_layout(
                    visibility_ranges_buffer_binding_type,
                    false,
                    Some(visibility_ranges_min_binding_size(
                        visibility_ranges_buffer_binding_type,
                    )),
                )
                .visibility(ShaderStages::VERTEX),
            ),
        ),
    );

    if layout_key.contains(MeshPipelineViewLayoutKey::DISTANCE_FOG) {
        entries = entries.extend_with_indices((
            // Fog
            (10, uniform_buffer::<GpuFog>(true)),
        ));
    }
    if layout_key.contains(MeshPipelineViewLayoutKey::SCREEN_SPACE_REFLECTIONS) {
        entries = entries.extend_with_indices((
            // Screen space reflection settings
            (12, uniform_buffer::<ScreenSpaceReflectionsUniform>(true)),
        ));
    }
    if layout_key.contains(MeshPipelineViewLayoutKey::CONTACT_SHADOWS) {
        entries = entries.extend_with_indices((
            // Contact shadows settings
            (13, uniform_buffer::<ContactShadowsUniform>(true)),
        ));
    }
    if layout_key.contains(MeshPipelineViewLayoutKey::SCREEN_SPACE_AMBIENT_OCCLUSION) {
        entries = entries.extend_with_indices((
            // Screen space ambient occlusion texture
            (
                14,
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ));
    }

    if layout_key.contains(MeshPipelineViewLayoutKey::TONEMAP_IN_SHADER) {
        // Tonemapping
        let tonemapping_lut_entries = get_lut_bind_group_layout_entries();
        entries = entries.extend_with_indices((
            (
                TONEMAPPING_LUT_TEXTURE_BINDING_INDEX,
                tonemapping_lut_entries[0],
            ),
            (
                TONEMAPPING_LUT_SAMPLER_BINDING_INDEX,
                tonemapping_lut_entries[1],
            ),
        ));
    }

    // Prepass
    if cfg!(any(feature = "webgpu", not(target_arch = "wasm32")))
        || !layout_key.contains(MeshPipelineViewLayoutKey::MULTISAMPLED)
    {
        for (entry, binding) in prepass::get_bind_group_layout_entries(layout_key)
            .iter()
            .zip([17, 18, 19, 20])
        {
            if let Some(entry) = entry {
                entries = entries.extend_with_indices(((binding as u32, *entry),));
            }
        }
    }

    // View Transmission Texture
    if layout_key.contains(MeshPipelineViewLayoutKey::VIEW_TRANSMISSION_TEXTURE) {
        entries = entries.extend_with_indices(((
            21,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),));
    }

    // Common linear sampler (shared by transmission, atmosphere, area-light LUTs, DFG LUT, and PCSS shadows)
    entries = entries.extend_with_indices(((
        LINEAR_SAMPLER_BINDING_INDEX,
        sampler(SamplerBindingType::Filtering),
    ),));

    // OIT
    if layout_key.contains(MeshPipelineViewLayoutKey::OIT_ENABLED) {
        // Check if we can use OIT. This is a hack to avoid errors on webgl --
        // the OIT plugin will warn the user that OIT is not supported on their
        // platform, so we don't need to do it here.
        if is_oit_supported {
            entries = entries.extend_with_indices((
                (
                    23,
                    uniform_buffer::<OrderIndependentTransparencySettings>(true),
                ),
                // oit_nodes_capacity
                (24, uniform_buffer::<u32>(false)),
                // oit_nodes
                (25, storage_buffer_sized(false, None)),
                // oit_heads,
                (26, storage_buffer_sized(false, None)),
                // oit_atomic_counter
                (
                    27,
                    storage_buffer_sized(false, NonZero::<u64>::new(size_of::<u32>() as u64)),
                ),
            ));
        }
    }

    // Atmosphere
    if layout_key.contains(MeshPipelineViewLayoutKey::ATMOSPHERE) {
        entries = entries.extend_with_indices((
            // transmittance LUT
            (
                28,
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
            // atmosphere data buffer
            (29, storage_buffer_read_only::<GpuAtmosphere>(false)),
        ));
    }

    // Blue noise
    if layout_key.contains(MeshPipelineViewLayoutKey::STBN) {
        entries = entries.extend_with_indices(((
            30,
            texture_2d_array(TextureSampleType::Float { filterable: false }),
        ),));
    }
    // LTC LUTs for area lights
    if cfg!(feature = "area_light_luts") {
        entries = entries.extend_with_indices(((
            31,
            texture_2d_array(TextureSampleType::Float { filterable: true }),
        ),));
    }
    // DFG LUT
    if cfg!(feature = "dfg_lut") {
        entries = entries.extend_with_indices(((
            32,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),));
    }

    let mut binding_array_entries = DynamicBindGroupLayoutEntries::new(ShaderStages::FRAGMENT);
    if layout_key.contains(MeshPipelineViewLayoutKey::ENVIRONMENT_MAP) {
        binding_array_entries = binding_array_entries.extend_with_indices((
            (0, environment_map_entries[0]),
            (1, environment_map_entries[1]),
            (2, environment_map_entries[2]),
        ));
    }

    if layout_key.contains(MeshPipelineViewLayoutKey::IRRADIANCE_VOLUME) {
        // Irradiance volumes
        binding_array_entries = binding_array_entries.extend_with_indices((
            (3, irradiance_volume_entries[0]),
            (4, irradiance_volume_entries[1]),
        ));
    }

    // Clustered decals
    if let Some(clustered_decal_entries) = clustered_decal_entries {
        binding_array_entries = binding_array_entries.extend_with_indices((
            (5, clustered_decal_entries[0]),
            (6, clustered_decal_entries[1]),
            (7, clustered_decal_entries[2]),
        ));
    }

    [entries.to_vec(), binding_array_entries.to_vec()]
}

/// Parameters needed by [`layout_entries`].
#[derive(Clone, Copy)]
struct MeshPipelineViewLayoutParams {
    clustered_forward_buffer_binding_type: BufferBindingType,
    visibility_ranges_buffer_binding_type: BufferBindingType,
    environment_map_entries: [BindGroupLayoutEntryBuilder; 3],
    irradiance_volume_entries: [BindGroupLayoutEntryBuilder; 2],
    clustered_decal_entries: Option<[BindGroupLayoutEntryBuilder; 3]>,
    is_oit_supported: bool,
}

/// Stores the view layouts entries for creating bind group layouts of pipeline keys.
#[derive(Resource, Clone)]
pub struct MeshPipelineViewLayouts {
    params: Arc<MeshPipelineViewLayoutParams>,
}

pub fn init_mesh_pipeline_view_layouts(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_adapter: Res<RenderAdapter>,
) {
    let clustered_forward_buffer_binding_type =
        render_device.get_supported_read_only_binding_type(CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT);
    let visibility_ranges_buffer_binding_type =
        render_device.get_supported_read_only_binding_type(VISIBILITY_RANGES_STORAGE_BUFFER_COUNT);

    let res = MeshPipelineViewLayouts {
        params: Arc::new(MeshPipelineViewLayoutParams {
            clustered_forward_buffer_binding_type,
            visibility_ranges_buffer_binding_type,
            environment_map_entries: environment_map::get_bind_group_layout_entries(
                &render_device,
                &render_adapter,
            ),
            irradiance_volume_entries: irradiance_volume::get_bind_group_layout_entries(
                &render_device,
                &render_adapter,
            ),
            clustered_decal_entries: decal::clustered::get_bind_group_layout_entries(
                &render_device,
                &render_adapter,
            ),
            is_oit_supported: bevy_core_pipeline::oit::resolve::is_oit_supported(
                &render_adapter,
                &render_device,
                false,
            ),
        }),
    };

    commands.insert_resource(res);
}

impl MeshPipelineViewLayouts {
    /// Get view bind group layout for the given key.
    pub fn get_view_layout(&self, layout_key: MeshPipelineViewLayoutKey) -> MeshPipelineViewLayout {
        let mut entries = layout_entries(layout_key, &self.params);

        #[cfg(debug_assertions)]
        let texture_count: usize = entries
            .iter()
            .flat_map(|e| {
                e.iter()
                    .filter(|entry| matches!(entry.ty, BindingType::Texture { .. }))
            })
            .count();

        #[cfg(debug_assertions)]
        if texture_count > MESH_PIPELINE_VIEW_LAYOUT_SAFE_MAX_TEXTURES {
            // Issue our own warning here because Naga's error message is a bit cryptic in this situation
            once!(warn!(
                "Too many textures in mesh pipeline view layout, this might cause us \
            to hit `wgpu::Limits::max_sampled_textures_per_shader_stage` in some environments."
            ));
        }

        MeshPipelineViewLayout {
            main_layout: BindGroupLayoutDescriptor {
                label: layout_key.label().into(),
                entries: core::mem::take(&mut entries[0]),
            },
            binding_array_layout: BindGroupLayoutDescriptor {
                label: layout_key
                    .label()
                    .replace("mesh_view_layout:", "mesh_view_layout_binding_array:")
                    .into(),
                entries: core::mem::take(&mut entries[1]),
            },
            empty_layout: BindGroupLayoutDescriptor {
                label: "mesh_view_layout_empty".into(),
                entries: vec![],
            },
        }
    }
}

#[derive(Component)]
pub struct MeshViewBindGroup {
    pub main: BindGroup,
    pub main_offsets: ArrayVec<u32, 7>,
    pub binding_array: BindGroup,
    pub empty: BindGroup,
}

/// Retains view bindings across frames; dynamic uniform offsets are not cached.
#[derive(Default)]
pub struct MeshViewBindGroupCache {
    views: EntityHashMap<CachedMeshViewBindGroups>,
}

#[derive(Default)]
struct CachedMeshViewBindGroups {
    seen: bool,
    groups: [SmallVec<[CachedMeshViewBindGroup; 2]>; 3],
}

struct CachedMeshViewBindGroup {
    layout: BindGroupLayoutId,
    entries: Vec<(u32, MeshViewBinding)>,
    bind_group: BindGroup,
    age: u8,
}

enum MeshViewBinding {
    Buffer(Buffer, u64, Option<BufferSize>),
    Sampler(Sampler),
    TextureView(TextureView),
    TextureViewArray(Vec<TextureView>),
}

impl MeshViewBinding {
    fn new(resource: &BindingResource<'_>) -> Self {
        match resource {
            BindingResource::Buffer(binding) => {
                Self::Buffer(binding.buffer.clone().into(), binding.offset, binding.size)
            }
            BindingResource::Sampler(sampler) => Self::Sampler((*sampler).clone().into()),
            BindingResource::TextureView(view) => Self::TextureView((*view).clone().into()),
            BindingResource::TextureViewArray(views) => {
                Self::TextureViewArray(views.iter().map(|view| (*view).clone().into()).collect())
            }
            _ => unreachable!("mesh views bind buffers, samplers, and texture views"),
        }
    }

    fn matches(&self, resource: &BindingResource<'_>) -> bool {
        match (self, resource) {
            (Self::Buffer(buffer, offset, size), BindingResource::Buffer(binding)) => {
                **buffer == *binding.buffer && *offset == binding.offset && *size == binding.size
            }
            (Self::Sampler(cached), BindingResource::Sampler(current)) => **cached == **current,
            (Self::TextureView(cached), BindingResource::TextureView(current)) => {
                **cached == **current
            }
            (Self::TextureViewArray(cached), BindingResource::TextureViewArray(current)) => {
                cached.len() == current.len()
                    && cached
                        .iter()
                        .zip(*current)
                        .all(|(cached, current)| **cached == **current)
            }
            _ => false,
        }
    }
}

impl MeshViewBindGroupCache {
    fn begin_frame(&mut self) {
        for cached in self.views.values_mut() {
            cached.seen = false;
            for group in cached.groups.iter_mut().flatten() {
                group.age = group.age.saturating_add(1);
            }
        }
    }

    fn finish_frame(&mut self, commands: &mut Commands) {
        self.views.retain(|&entity, cached| {
            if !cached.seen {
                commands.entity(entity).try_remove::<MeshViewBindGroup>();
            }
            cached.seen
        });
    }
}

impl CachedMeshViewBindGroups {
    fn prepare(
        &mut self,
        render_device: &RenderDevice,
        layouts: [&BindGroupLayout; 3],
        entries: [&[BindGroupEntry<'_>]; 3],
        offsets: ArrayVec<u32, 7>,
    ) -> MeshViewBindGroup {
        self.seen = true;
        let [main, binding_array, empty] =
            from_fn(|index| {
                let groups = &mut self.groups[index];
                let layout = layouts[index];
                let entries = entries[index];
                let bind_group = if let Some(cached) =
                    groups.iter_mut().find(|cached| {
                        cached.layout == layout.id()
                            && cached.entries.len() == entries.len()
                            && cached.entries.iter().zip(entries).all(
                                |((binding, resource), entry)| {
                                    *binding == entry.binding && resource.matches(&entry.resource)
                                },
                            )
                    }) {
                    cached.age = 0;
                    cached.bind_group.clone()
                } else {
                    let bind_group = render_device.create_bind_group(
                        [
                            "mesh_view_bind_group",
                            "mesh_view_bind_group_binding_array",
                            "mesh_view_bind_group_empty",
                        ][index],
                        layout,
                        entries,
                    );
                    groups.push(CachedMeshViewBindGroup {
                        layout: layout.id(),
                        entries: entries
                            .iter()
                            .map(|entry| (entry.binding, MeshViewBinding::new(&entry.resource)))
                            .collect(),
                        bind_group: bind_group.clone(),
                        age: 0,
                    });
                    bind_group
                };
                // Keep the preceding frame's backing when phase buffers alternate.
                groups.retain(|cached| cached.age < 2);
                bind_group
            });
        MeshViewBindGroup {
            main,
            main_offsets: offsets,
            binding_array,
            empty,
        }
    }
}

pub fn prepare_mesh_view_bind_groups(
    mut commands: Commands,
    (render_device, pipeline_cache, render_adapter): (
        Res<RenderDevice>,
        Res<PipelineCache>,
        Res<RenderAdapter>,
    ),
    mesh_pipeline: Res<MeshPipeline>,
    shadow_samplers: Res<ShadowSamplers>,
    (light_meta, global_clusterable_object_meta, fog_meta, view_uniforms): (
        Res<LightMeta>,
        Res<GlobalClusterableObjectMeta>,
        Res<FogMeta>,
        Res<ViewUniforms>,
    ),
    views: Query<(
        Entity,
        Option<&ExtractedCamera>,
        &ViewShadowBindings,
        &ViewClusterBindings,
        &Msaa,
        Option<&ScreenSpaceAmbientOcclusionResources>,
        Option<&ViewPrepassTextures>,
        Option<&ViewTransmissionTexture>,
        Option<&AtmosphereTextures>,
        Option<&AtmosphereBuffer>,
        &Tonemapping,
        (
            Option<&RenderViewLightProbes<EnvironmentMapLight>>,
            Option<&RenderViewLightProbes<IrradianceVolume>>,
        ),
        Has<ExtractedAtmosphere>,
        (
            &ViewUniformOffset,
            &ViewLightsUniformOffset,
            &ViewLightProbesUniformOffset,
            Option<&ViewFogUniformOffset>,
            Option<&ViewScreenSpaceReflectionsUniformOffset>,
            Option<&ViewContactShadowsUniformOffset>,
            Option<&OrderIndependentTransparencySettingsOffset>,
            Has<ScreenSpaceTransmission>,
        ),
    )>,
    (images, fallback_image, fallback_image_zero): (
        Res<RenderAssets<GpuImage>>,
        Res<FallbackImage>,
        Res<FallbackImageZero>,
    ),
    globals_buffer: Res<GlobalsBuffer>,
    tonemapping_luts: Res<TonemappingLuts>,
    light_probes_buffer: Res<LightProbesBuffer>,
    visibility_ranges: Res<RenderVisibilityRanges>,
    (ssr_buffer, contact_shadows_buffer, oit_buffers): (
        Res<ScreenSpaceReflectionsBuffer>,
        Res<ContactShadowsBuffer>,
        Res<OitBuffers>,
    ),
    (decals_buffer, render_decals, blue_noise, area_light_luts, dfg_lut): (
        Res<DecalsBuffer>,
        Res<RenderClusteredDecals>,
        Res<Bluenoise>,
        Res<AreaLightLuts>,
        Res<DfgLut>,
    ),
    (linear_sampler, mut bind_group_cache): (Res<LinearSampler>, Local<MeshViewBindGroupCache>),
    // TODO: Figure out how to reuse the memory. `BindGroupEntry` is non-send on wasm with atomics.
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))] mut entries_cache: Local<
        Vec<BindGroupEntry>,
    >,
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
    mut entries_binding_array_cache: Local<Vec<BindGroupEntry>>,
) {
    bind_group_cache.begin_frame();
    if let (
        Some(view_binding),
        Some(light_binding),
        Some(clusterable_objects_binding),
        Some(globals),
        Some(light_probes_binding),
        Some(visibility_ranges_buffer),
    ) = (
        view_uniforms.uniforms.binding(),
        light_meta.view_gpu_lights.binding(),
        global_clusterable_object_meta
            .gpu_clustered_lights
            .binding(),
        globals_buffer.buffer.binding(),
        light_probes_buffer.binding(),
        visibility_ranges.buffer().buffer(),
    ) {
        for (
            entity,
            camera,
            shadow_bindings,
            cluster_bindings,
            msaa,
            ssao_resources,
            prepass_textures,
            transmission_texture,
            atmosphere_textures,
            atmosphere_buffer,
            tonemapping,
            (render_view_environment_maps, render_view_irradiance_volumes),
            has_atmosphere,
            (
                view_uniform_offset,
                view_lights_offset,
                view_light_probes_offset,
                view_fog_offset,
                view_ssr_offset,
                view_contact_shadows_offset,
                view_oit_settings_offset,
                has_transmission,
            ),
        ) in &views
        {
            let mut entries = DynamicBindGroupEntries::new();
            let mut entries_binding_array = DynamicBindGroupEntries::new();
            #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
            {
                // Take cache that has static lifetime for `DynamicBindGroupEntries`.
                // See <https://users.rust-lang.org/t/how-to-cache-a-vectors-capacity/94478/10>.
                entries.entries = core::mem::take(&mut *entries_cache)
                    .into_iter()
                    .map(|_| -> BindGroupEntry { unreachable!() })
                    .collect();
                entries_binding_array.entries = core::mem::take(&mut *entries_binding_array_cache)
                    .into_iter()
                    .map(|_| -> BindGroupEntry { unreachable!() })
                    .collect();
            }

            let tonemap_in_shader = camera.is_some_and(|camera| camera.tonemap_in_shader);
            let mut layout_key = MeshPipelineViewLayoutKey::from(*msaa)
                | MeshPipelineViewLayoutKey::from(prepass_textures);
            let mut offsets = ArrayVec::from_iter([
                view_uniform_offset.offset,
                view_lights_offset.offset,
                **view_light_probes_offset,
            ]);

            entries = entries.extend_with_indices((
                (0, view_binding.clone()),
                (1, light_binding.clone()),
                (2, &shadow_bindings.point_light_depth_texture_view),
                (3, &shadow_samplers.shadow_comparison_sampler),
                (4, &shadow_bindings.directional_light_depth_texture_view),
                (5, clusterable_objects_binding.clone()),
                (
                    6,
                    cluster_bindings
                        .clusterable_object_index_lists_binding()
                        .unwrap(),
                ),
                (7, cluster_bindings.offsets_and_counts_binding().unwrap()),
                (8, globals.clone()),
                (9, light_probes_binding.clone()),
                (
                    VISIBILITY_RANGES_BINDING_INDEX,
                    visibility_ranges_buffer.as_entire_binding(),
                ),
                (LINEAR_SAMPLER_BINDING_INDEX, &linear_sampler.0),
            ));

            if let Some(view_fog_offset) = view_fog_offset {
                layout_key |= MeshPipelineViewLayoutKey::DISTANCE_FOG;
                offsets.push(view_fog_offset.offset);
                entries =
                    entries.extend_with_indices(((10, fog_meta.gpu_fogs.binding().unwrap()),));
            }

            if let Some(view_ssr_offset) = view_ssr_offset {
                layout_key |= MeshPipelineViewLayoutKey::SCREEN_SPACE_REFLECTIONS;
                offsets.push(**view_ssr_offset);
                entries = entries.extend_with_indices(((12, ssr_buffer.binding().unwrap()),));
            }

            if let Some(view_contact_shadows_offset) = view_contact_shadows_offset {
                layout_key |= MeshPipelineViewLayoutKey::CONTACT_SHADOWS;
                offsets.push(**view_contact_shadows_offset);
                entries = entries
                    .extend_with_indices(((13, contact_shadows_buffer.0.binding().unwrap()),));
            }

            if let Some(view_oit_settings_offset) = view_oit_settings_offset {
                layout_key |= MeshPipelineViewLayoutKey::OIT_ENABLED;
                offsets.push(view_oit_settings_offset.offset);
                entries = entries.extend_with_indices((
                    (23, oit_buffers.settings.binding().unwrap()),
                    (24, oit_buffers.nodes_capacity.binding().unwrap()),
                    (25, oit_buffers.nodes.binding().unwrap()),
                    (26, oit_buffers.heads.binding().unwrap()),
                    (27, oit_buffers.atomic_counter.binding().unwrap()),
                ));
            }

            if has_atmosphere
                && let Some(atmosphere_textures) = atmosphere_textures
                && let Some(atmosphere_buffer) = atmosphere_buffer
                && let Some(atmosphere_buffer_binding) = atmosphere_buffer.buffer.binding()
            {
                layout_key |= MeshPipelineViewLayoutKey::ATMOSPHERE;
                entries = entries.extend_with_indices((
                    (28, &atmosphere_textures.transmittance_lut.default_view),
                    (29, atmosphere_buffer_binding),
                ));
            }

            if cfg!(feature = "bluenoise_texture") {
                layout_key |= MeshPipelineViewLayoutKey::STBN;
                let stbn_view = &images
                    .get(&blue_noise.texture)
                    .expect("STBN texture is added unconditionally with at least a placeholder")
                    .texture_view;
                entries = entries.extend_with_indices(((30, stbn_view),));
            }

            if tonemap_in_shader {
                layout_key |= MeshPipelineViewLayoutKey::TONEMAP_IN_SHADER;
                let lut_bindings =
                    get_lut_bindings(&images, &tonemapping_luts, tonemapping, &fallback_image);
                entries = entries.extend_with_indices((
                    (TONEMAPPING_LUT_TEXTURE_BINDING_INDEX, lut_bindings.0),
                    (TONEMAPPING_LUT_SAMPLER_BINDING_INDEX, lut_bindings.1),
                ));
            }

            if let Some(ssao_resources) = ssao_resources {
                layout_key |= MeshPipelineViewLayoutKey::SCREEN_SPACE_AMBIENT_OCCLUSION;
                let ssao_view = &ssao_resources
                    .screen_space_ambient_occlusion_texture
                    .default_view;
                entries = entries.extend_with_indices(((14, ssao_view),));
            }

            if has_transmission {
                layout_key |= MeshPipelineViewLayoutKey::VIEW_TRANSMISSION_TEXTURE;
                let transmission_view = transmission_texture
                    .map(|transmission| &transmission.view)
                    .unwrap_or(&fallback_image_zero.texture_view);
                entries = entries.extend_with_indices(((21, transmission_view),));
            }

            // When using WebGL, we can't have a multisampled texture with `TEXTURE_BINDING`
            // See https://github.com/gfx-rs/wgpu/issues/5263
            let prepass_bindings;
            if cfg!(any(feature = "webgpu", not(target_arch = "wasm32"))) || msaa.samples() == 1 {
                prepass_bindings = prepass::get_bindings(prepass_textures);
                for (binding, index) in prepass_bindings
                    .iter()
                    .map(Option::as_ref)
                    .zip([17, 18, 19, 20])
                    .flat_map(|(b, i)| b.map(|b| (b, i)))
                {
                    entries = entries.extend_with_indices(((index, binding),));
                }
            };

            // LTC LUTs for area lights
            if cfg!(feature = "area_light_luts") {
                let ltc_view = images
                    .get(&area_light_luts.image)
                    .map(|img| &img.texture_view)
                    .unwrap_or(&fallback_image.d2_array.texture_view);
                entries = entries.extend_with_indices(((31, ltc_view),));
            }

            // DFG LUT
            if cfg!(feature = "dfg_lut") {
                let dfg_view = images
                    .get(&dfg_lut.texture)
                    .map(|img| &img.texture_view)
                    .unwrap_or(&fallback_image.d2.texture_view);
                entries = entries.extend_with_indices(((32, dfg_view),));
            }

            let environment_map_bind_group_entries =
                render_view_environment_maps.map(|render_view_environment_maps| {
                    layout_key |= MeshPipelineViewLayoutKey::ENVIRONMENT_MAP;

                    RenderViewEnvironmentMapBindGroupEntries::get(
                        Some(render_view_environment_maps),
                        &images,
                        &fallback_image,
                        &render_device,
                        &render_adapter,
                    )
                });
            match environment_map_bind_group_entries {
                Some(RenderViewEnvironmentMapBindGroupEntries::Single {
                    diffuse_texture_view,
                    specular_texture_view,
                    sampler,
                }) => {
                    entries_binding_array = entries_binding_array.extend_with_indices((
                        (0, diffuse_texture_view),
                        (1, specular_texture_view),
                        (2, sampler),
                    ));
                }
                Some(RenderViewEnvironmentMapBindGroupEntries::Multiple {
                    ref diffuse_texture_views,
                    ref specular_texture_views,
                    sampler,
                }) => {
                    entries_binding_array = entries_binding_array.extend_with_indices((
                        (0, diffuse_texture_views.as_slice()),
                        (1, specular_texture_views.as_slice()),
                        (2, sampler),
                    ));
                }
                None => {}
            }

            let irradiance_volume_bind_group_entries = if render_view_irradiance_volumes.is_some() {
                layout_key |= MeshPipelineViewLayoutKey::IRRADIANCE_VOLUME;

                Some(RenderViewIrradianceVolumeBindGroupEntries::get(
                    render_view_irradiance_volumes,
                    &images,
                    &fallback_image,
                    &render_device,
                    &render_adapter,
                ))
            } else {
                None
            };

            match irradiance_volume_bind_group_entries {
                Some(RenderViewIrradianceVolumeBindGroupEntries::Single {
                    texture_view,
                    sampler,
                }) => {
                    entries_binding_array = entries_binding_array
                        .extend_with_indices(((3, texture_view), (4, sampler)));
                }
                Some(RenderViewIrradianceVolumeBindGroupEntries::Multiple {
                    ref texture_views,
                    sampler,
                }) => {
                    entries_binding_array = entries_binding_array
                        .extend_with_indices(((3, texture_views.as_slice()), (4, sampler)));
                }
                None => {}
            }

            let decal_bind_group_entries = RenderViewClusteredDecalBindGroupEntries::get(
                &render_decals,
                &decals_buffer,
                &images,
                &fallback_image,
                &render_device,
                &render_adapter,
            );

            // Add the decal bind group entries.
            if let Some(ref render_view_decal_bind_group_entries) = decal_bind_group_entries {
                entries_binding_array = entries_binding_array.extend_with_indices((
                    // `clustered_decals`
                    (
                        5,
                        render_view_decal_bind_group_entries
                            .decals
                            .as_entire_binding(),
                    ),
                    // `clustered_decal_textures`
                    (
                        6,
                        render_view_decal_bind_group_entries
                            .texture_views
                            .as_slice(),
                    ),
                    // `clustered_decal_sampler`
                    (7, render_view_decal_bind_group_entries.sampler),
                ));
            }

            let layout = mesh_pipeline.get_view_layout(layout_key);
            let main_layout = pipeline_cache.get_bind_group_layout(&layout.main_layout);
            let binding_array_layout =
                pipeline_cache.get_bind_group_layout(&layout.binding_array_layout);
            let empty_layout = pipeline_cache.get_bind_group_layout(&layout.empty_layout);
            let groups = bind_group_cache.views.entry(entity).or_default().prepare(
                &render_device,
                [&main_layout, &binding_array_layout, &empty_layout],
                [&entries, &entries_binding_array, &[]],
                offsets,
            );
            commands.entity(entity).insert(groups);

            #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
            {
                entries.entries.clear();
                entries_binding_array.entries.clear();
                *entries_cache = entries
                    .entries
                    .into_iter()
                    .map(|_| -> BindGroupEntry<'static> { unreachable!() })
                    .collect();
                *entries_binding_array_cache = entries_binding_array
                    .entries
                    .into_iter()
                    .map(|_| -> BindGroupEntry<'static> { unreachable!() })
                    .collect();
            }
        }
    }
    bind_group_cache.finish_frame(&mut commands);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_platform::future::block_on;
    use bevy_render::renderer::RenderInstance;
    use wgpu_types::DeviceDescriptor;

    #[test]
    #[ignore = "requires a graphics adapter with texture binding arrays"]
    fn view_bindings_reuse_alternating_resources_and_retire_absent_views() {
        let instance = RenderInstance::new(Default::default());
        let adapter = block_on(instance.request_adapter(&Default::default()))
            .expect("a graphics adapter is available");
        let (device, _queue) = block_on(adapter.request_device(&DeviceDescriptor {
            required_features: WgpuFeatures::TEXTURE_BINDING_ARRAY,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("the adapter supports texture binding arrays");
        let device = RenderDevice::new(device);
        let main_layout = device.create_bind_group_layout(
            "cached view test",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    storage_buffer_read_only::<u32>(true),
                    sampler(SamplerBindingType::NonFiltering),
                ),
            ),
        );
        let array_layout = device.create_bind_group_layout(
            "cached view array test",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: NonZero::new(2),
            }],
        );
        let empty_layout = device.create_bind_group_layout("cached empty view test", &[]);
        let buffer = || {
            device.create_buffer(&BufferDescriptor {
                label: None,
                size: 1024,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let texture = || {
            device
                .create_texture(&TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };
        let a = buffer();
        let b = buffer();
        let textures = [texture(), texture()];
        let view_sampler = device.create_sampler(&SamplerDescriptor::default());
        let mut world = World::new();
        let view = world.spawn_empty().id();
        let mut cache = MeshViewBindGroupCache::default();
        let mut prepare = |cache: &mut MeshViewBindGroupCache,
                           buffer: &Buffer,
                           offset,
                           size,
                           textures: [&TextureView; 2],
                           sampler: &Sampler,
                           layout: &BindGroupLayout,
                           dynamic_offset| {
            cache.begin_frame();
            let textures = textures.map(|view| &**view);
            let main = BindGroupEntries::sequential((
                BufferBinding {
                    buffer,
                    offset,
                    size,
                },
                sampler,
            ));
            let array = BindGroupEntries::single(textures.as_slice());
            let group = cache.views.entry(view).or_default().prepare(
                &device,
                [layout, &array_layout, &empty_layout],
                [&main, &array, &[]],
                ArrayVec::from_iter([dynamic_offset]),
            );
            let result = (
                group.main.id(),
                group.binding_array.id(),
                group.empty.id(),
                group.main_offsets[0],
            );
            let mut commands = world.commands();
            commands.entity(view).insert(group);
            cache.finish_frame(&mut commands);
            world.flush();
            result
        };
        let views = [&textures[0], &textures[1]];
        let first = prepare(
            &mut cache,
            &a,
            0,
            NonZero::new(4),
            views,
            &view_sampler,
            &main_layout,
            0,
        );
        let second = prepare(
            &mut cache,
            &b,
            0,
            NonZero::new(4),
            views,
            &view_sampler,
            &main_layout,
            256,
        );
        assert_ne!(first.0, second.0, "a new backing needs new main bindings");
        assert_eq!((first.1, first.2), (second.1, second.2));
        let third = prepare(
            &mut cache,
            &a,
            0,
            NonZero::new(4),
            views,
            &view_sampler,
            &main_layout,
            512,
        );
        assert_eq!(
            (third.0, third.1, third.2),
            (first.0, first.1, first.2),
            "A/B/A reuses both backings"
        );
        assert_eq!(
            third.3, 512,
            "a cache hit refreshes the current dynamic offset"
        );

        let mut previous = third.0;
        for (offset, size) in [(256, NonZero::new(4)), (256, NonZero::new(8)), (256, None)] {
            let current = prepare(
                &mut cache,
                &a,
                offset,
                size,
                views,
                &view_sampler,
                &main_layout,
                0,
            );
            assert_ne!(
                current.0, previous,
                "buffer binding base and size belong to the identity"
            );
            previous = current.0;
            assert!(cache.views[&view]
                .groups
                .iter()
                .all(|groups| groups.len() <= 2));
        }
        let reordered = prepare(
            &mut cache,
            &a,
            256,
            None,
            [views[1], views[0]],
            &view_sampler,
            &main_layout,
            0,
        );
        assert_eq!(reordered.0, previous);
        assert_ne!(
            reordered.1, first.1,
            "texture array order belongs to the identity"
        );
        assert_eq!(reordered.2, first.2);

        let replacement_sampler = device.create_sampler(&SamplerDescriptor::default());
        let changed_sampler = prepare(
            &mut cache,
            &a,
            256,
            None,
            views,
            &replacement_sampler,
            &main_layout,
            0,
        );
        assert_ne!(changed_sampler.0, reordered.0);
        let replacement_layout = device.create_bind_group_layout(
            "replacement view layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    storage_buffer_read_only::<u32>(true),
                    sampler(SamplerBindingType::NonFiltering),
                ),
            ),
        );
        let changed_layout = prepare(
            &mut cache,
            &a,
            256,
            None,
            views,
            &replacement_sampler,
            &replacement_layout,
            0,
        );
        assert_ne!(changed_layout.0, changed_sampler.0);
        drop(prepare);
        cache.begin_frame();
        cache.finish_frame(&mut world.commands());
        world.flush();
        assert!(cache.views.is_empty());
        assert!(
            world.get::<MeshViewBindGroup>(view).is_none(),
            "an unavailable view cannot retain drawable bindings"
        );
    }
}
