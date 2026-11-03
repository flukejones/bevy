---
title: GPU scene-preparation source storage and dispatches
pull_requests: []
---

GPU scene preparation uses shared source storage and grouped dispatches to reduce per-batch resource and dispatch overhead. Custom renderers using the preprocessing internals must update these APIs:

- `RenderMultidrawableBatchSetGpuBuffers` has no replacement per-batch GPU owner. Queue instances through `BinnedRenderPhase`; `SceneUnpackingBuffers` owns the shared storage and current dispatch jobs.
- Initialize `SceneUnpackingBuffers` through `World::init_resource` or `FromWorld`, with `RenderDevice` available. It no longer implements `Default`.
- Register `write_binned_instance_buffers::<BPI>` without the former `GFBD` type parameter.
- `SceneUnpackingJob::source_buffer` contains packed bin metadata, instances and the sparse bin map. `GpuSceneUnpackingMetadata` supplies their word offsets and the shared fan-scratch offset. Custom shader bindings must use the shared-source layout.
- A `SceneUnpackingJob` describes a dispatch, which can contain several batch sets or part of a larger batch set. Dispatch its current `workgroup_count`; `bin_workgroup_count > 1` selects the large local/global/fan scan. The dynamic metadata binding spans `SCENE_UNPACKING_WORKGROUPS_PER_DISPATCH` records. Retain every current job, even when several jobs share the same GPU resources.
