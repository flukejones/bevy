use core::{any::TypeId, num::NonZeroU64};

use bevy_ecs::entity::Entity;
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutId, BindingResource, Buffer,
        TextureView,
    },
    renderer::RenderDevice,
};
use smallvec::SmallVec;

/// Retained mesh-preprocessing bindings, partitioned by view and render phase.
///
/// Entries must be visited each preparation frame. Unvisited phases and binding
/// roles retire; mesh output bindings retain at most two alternating buffers.
#[derive(Default)]
pub struct PreprocessBindGroupCache {
    phases: HashMap<(Option<Entity>, TypeId), CachedPhaseBindGroups>,
}

#[derive(Default)]
pub(super) struct CachedPhaseBindGroups {
    groups: HashMap<&'static str, CachedBindGroupSlot>,
    alternating_binding: Option<u32>,
}

#[derive(Default)]
struct CachedBindGroupSlot {
    variants: SmallVec<[CachedBindGroup; 2]>,
    used: bool,
}

struct CachedBindGroup {
    layout: BindGroupLayoutId,
    entries: SmallVec<[CachedBinding; 12]>,
    group: BindGroup,
}

enum CachedBinding {
    Buffer {
        binding: u32,
        buffer: Buffer,
        offset: u64,
        size: Option<NonZeroU64>,
    },
    TextureView {
        binding: u32,
        view: TextureView,
    },
}

impl PreprocessBindGroupCache {
    pub(super) fn begin_frame(&mut self) {
        for phase in self.phases.values_mut() {
            for group in phase.groups.values_mut() {
                group.used = false;
            }
        }
    }

    pub(super) fn phase(
        &mut self,
        view: Option<Entity>,
        phase: TypeId,
    ) -> &mut CachedPhaseBindGroups {
        self.phases
            .entry((view, phase))
            .or_insert_with(|| CachedPhaseBindGroups {
                // Phase collection alternates the mesh uniform output buffers.
                alternating_binding: view.map(|_| 6),
                ..Default::default()
            })
    }

    pub(super) fn end_frame(&mut self) {
        self.phases.retain(|_, phase| {
            phase.groups.retain(|_, group| group.used);
            !phase.groups.is_empty()
        });
    }

    pub(super) fn clear(&mut self) {
        self.phases.clear();
    }
}

impl CachedPhaseBindGroups {
    pub(super) fn prepare(
        &mut self,
        render_device: &RenderDevice,
        label: &'static str,
        layout: &BindGroupLayout,
        entries: &[BindGroupEntry<'_>],
    ) -> BindGroup {
        let slot = self.groups.entry(label).or_default();
        slot.used = true;
        slot.variants.retain(|cached| {
            cached.layout == layout.id()
                && cached.entries.len() == entries.len()
                && cached
                    .entries
                    .iter()
                    .zip(entries)
                    .all(|(cached, entry)| cached.matches(entry, self.alternating_binding))
        });
        if let Some(index) = slot.variants.iter().position(|cached| {
            cached
                .entries
                .iter()
                .zip(entries)
                .all(|(cached, entry)| cached.matches(entry, None))
        }) {
            slot.variants.swap(0, index);
            return slot.variants[0].group.clone();
        }

        let group = render_device.create_bind_group(label, layout, entries);
        slot.variants.truncate(1);
        slot.variants.insert(
            0,
            CachedBindGroup {
                layout: layout.id(),
                entries: entries.iter().map(CachedBinding::new).collect(),
                group: group.clone(),
            },
        );
        group
    }
}

impl CachedBinding {
    fn new(entry: &BindGroupEntry<'_>) -> Self {
        match &entry.resource {
            BindingResource::Buffer(value) => Self::Buffer {
                binding: entry.binding,
                buffer: Buffer::from(value.buffer.clone()),
                offset: value.offset,
                size: value.size,
            },
            BindingResource::TextureView(view) => Self::TextureView {
                binding: entry.binding,
                view: TextureView::from((*view).clone()),
            },
            _ => unreachable!("mesh preprocessing binds only buffers and depth texture views"),
        }
    }

    fn matches(&self, entry: &BindGroupEntry<'_>, alternating_binding: Option<u32>) -> bool {
        match (self, &entry.resource) {
            (
                Self::Buffer {
                    binding,
                    buffer,
                    offset,
                    size,
                },
                BindingResource::Buffer(value),
            ) => {
                *binding == entry.binding
                    && *offset == value.offset
                    && *size == value.size
                    && (&**buffer == value.buffer
                        || (alternating_binding == Some(*binding)
                            && buffer.size() == value.buffer.size()))
            }
            (Self::TextureView { binding, view }, BindingResource::TextureView(value)) => {
                *binding == entry.binding && &**view == *value
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy_core_pipeline::core_3d::Opaque3d;
    use bevy_ecs::world::World;
    use bevy_platform::future::block_on;
    use bevy_render::{
        render_resource::{
            binding_types::{storage_buffer, storage_buffer_read_only, texture_2d},
            BindGroupEntries, BindGroupLayoutEntries, Buffer, BufferBinding, BufferDescriptor,
            BufferUsages, Extent3d, ShaderStages, TextureDescriptor, TextureDimension,
            TextureFormat, TextureSampleType, TextureUsages, TextureView,
        },
        renderer::RenderInstance,
    };
    use wgpu_types::DeviceDescriptor;

    use super::*;

    #[test]
    #[ignore = "requires a graphics adapter"]
    fn preprocessing_bindings_reuse_and_retire_gpu_resources() {
        let instance = RenderInstance::new(Default::default());
        let adapter = block_on(instance.request_adapter(&Default::default()))
            .expect("preprocessing cache regression requires a graphics adapter");
        let (device, _) = block_on(adapter.request_device(&DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("preprocessing cache device");
        let device = RenderDevice::new(device);
        let alignment = u64::from(device.limits().min_storage_buffer_offset_alignment);
        let buffer = || {
            device.create_buffer(&BufferDescriptor {
                label: Some("preprocessing cache regression"),
                size: alignment * 2,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let input = buffer();
        let replacement_input = buffer();
        let output_a = buffer();
        let output_b = buffer();
        let output_c = buffer();
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("preprocessing cache regression"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth = texture.create_view(&Default::default());
        let replacement_depth = texture.create_view(&Default::default());
        let layout_entries = BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, storage_buffer_read_only::<u32>(false)),
                (6, storage_buffer::<u32>(false)),
                (
                    11,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
            ),
        );
        let layout = device.create_bind_group_layout("preprocessing cache", &layout_entries);
        let replacement_layout =
            device.create_bind_group_layout("preprocessing cache", &layout_entries);
        let view = World::new().spawn_empty().id();
        let phase = TypeId::of::<Opaque3d>();
        let mut cache = PreprocessBindGroupCache::default();
        let prepare = |cache: &mut PreprocessBindGroupCache,
                       input: BufferBinding<'_>,
                       output: &Buffer,
                       depth: &TextureView,
                       layout: &BindGroupLayout| {
            cache.begin_frame();
            let group = cache.phase(Some(view), phase).prepare(
                &device,
                "preprocessing cache",
                layout,
                &BindGroupEntries::with_indices((
                    (0, input),
                    (6, output.as_entire_binding()),
                    (11, depth),
                )),
            );
            cache.end_frame();
            group.id()
        };
        let mut binding = BufferBinding {
            buffer: &input,
            offset: 0,
            size: NonZeroU64::new(4),
        };
        let first = prepare(&mut cache, binding.clone(), &output_a, &depth, &layout);
        assert_eq!(
            prepare(&mut cache, binding.clone(), &output_a, &depth, &layout),
            first,
            "unchanged resources reuse their native bind group"
        );
        let second = prepare(&mut cache, binding.clone(), &output_b, &depth, &layout);
        assert_ne!(second, first);
        assert_eq!(
            prepare(&mut cache, binding.clone(), &output_a, &depth, &layout),
            first,
            "phase buffer alternation must reuse A after B"
        );

        binding.buffer = &replacement_input;
        let replaced = prepare(&mut cache, binding.clone(), &output_a, &depth, &layout);
        assert_ne!(replaced, first);
        binding.buffer = &input;
        let restored = prepare(&mut cache, binding.clone(), &output_b, &depth, &layout);
        assert_ne!(restored, second, "input replacement retires both variants");
        binding.offset = alignment;
        let offset = prepare(&mut cache, binding.clone(), &output_b, &depth, &layout);
        assert_ne!(offset, restored, "binding offsets are part of validity");
        binding.size = NonZeroU64::new(8);
        let resized = prepare(&mut cache, binding.clone(), &output_b, &depth, &layout);
        assert_ne!(resized, offset, "shader-visible array lengths must refresh");
        let retextured = prepare(
            &mut cache,
            binding.clone(),
            &output_b,
            &replacement_depth,
            &layout,
        );
        assert_ne!(retextured, resized, "depth texture-view identity matters");
        let relaid = prepare(
            &mut cache,
            binding.clone(),
            &output_b,
            &replacement_depth,
            &replacement_layout,
        );
        assert_ne!(relaid, retextured, "replacement layouts retire old groups");
        let third = prepare(
            &mut cache,
            binding.clone(),
            &output_c,
            &replacement_depth,
            &replacement_layout,
        );
        let fourth = prepare(
            &mut cache,
            binding.clone(),
            &output_a,
            &replacement_depth,
            &replacement_layout,
        );
        assert_ne!(third, fourth);
        assert_ne!(
            prepare(
                &mut cache,
                binding.clone(),
                &output_b,
                &replacement_depth,
                &replacement_layout,
            ),
            relaid,
            "a third output evicts the least recently used variant"
        );

        cache.begin_frame();
        cache.end_frame();
        assert_ne!(
            prepare(
                &mut cache,
                binding,
                &output_a,
                &replacement_depth,
                &replacement_layout,
            ),
            fourth,
            "an unused view/phase retires its cached resources"
        );
    }
}
