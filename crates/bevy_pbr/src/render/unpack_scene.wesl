//! Allocates contiguous mesh-uniform ranges and unpacks cached batch-set instances.
//!
//! Local scan and unpacking are independent, so one dispatch covers both their
//! bin and instance ranges. Batch sets with more than 256 bins finish allocation
//! with a global prefix scan and fan pass before mesh preprocessing.

import bevy_render::occlusion_culling::mesh_preprocess_types::{BinMetadata, IndirectParametersMetadata, PreprocessWorkItem};

/// Current-frame ranges for one cached batch set.
struct SceneUnpackingMetadata {
    /// The index of this batch set in the `IndirectBatchSet` array.
    ///
    /// We write this into the `indirect_parameters_metadata`.
    batch_set_index: u32,

    /// The number of bins (a.k.a. draws, a.k.a. batches) in this batch set.
    bin_count: u32,

    /// The index of the first set of indirect parameters for this batch set.
    ///
    /// This is also the index of the first `IndirectParametersMetadata`, as
    /// that's a parallel array with the indirect parameters.
    first_indirect_parameters_index: u32,

    /// The index of the first `MeshUniform` slot for this batch set.
    first_output_mesh_uniform_index: u32,

    /// First output work item for this batch set.
    base_output_work_item_index: u32,
    /// First indirect parameter record used by its work items.
    base_indirect_parameters_index: u32,
    /// Number of cached mesh instances, independently of the bin count.
    binned_mesh_instance_count: u32,
    pad_a: u32,
    pad: array<vec4<u32>, 14u>,
};

/// The number of threads in a workgroup.
const WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> metadata: SceneUnpackingMetadata;

/// Information for each bin, including the indirect parameters offset and the
/// instance count.
@group(0) @binding(1) var<storage> bin_metadata: array<BinMetadata>;

/// The array of indirect parameters metadata that we fill out, one for each
/// batch.
@group(0) @binding(2) var<storage, read_write> indirect_parameters_metadata:
    array<IndirectParametersMetadata>;

/// A temporary buffer that stores the mesh uniform index of the last instance
/// plus one for each workgroup (i.e. for each 256-bin chunk).
///
/// This is accumulated in the second stage and written out in the third.
@group(0) @binding(3) var<storage, read_write> fan_buffer: array<u32>;

struct BinnedMeshInstance {
    input_uniform_index: u32,
    bin_index: u32,
};

@group(0) @binding(4) var<storage> binned_mesh_instances: array<BinnedMeshInstance>;
@group(0) @binding(5) var<storage, read_write> preprocess_work_items: array<PreprocessWorkItem>;
@group(0) @binding(6) var<storage> bin_index_to_bin_metadata_index: array<u32>;

/// Scratch memory that stores the prefix sum for every element in our chunk.
var<workgroup> output_offsets: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn unpack_scene(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (global_id.x < metadata.binned_mesh_instance_count) {
        let instance = binned_mesh_instances[global_id.x];
        let bin_metadata_index = bin_index_to_bin_metadata_index[instance.bin_index];
        let output_index = metadata.base_output_work_item_index + global_id.x;
        preprocess_work_items[output_index].input_index = instance.input_uniform_index;
        preprocess_work_items[output_index].output_or_indirect_parameters_index =
            metadata.base_indirect_parameters_index +
            bin_metadata[bin_metadata_index].indirect_parameters_offset;
    }

    // Every invocation in a workgroup takes the same allocation path. Extra
    // workgroups can still unpack instances after the last bin.
    let block_start = group_id.x * WORKGROUP_SIZE;
    if (block_start >= metadata.bin_count) {
        return;
    }
    if (metadata.bin_count == 1u) {
        if (local_id.x == 0u) {
            write_indirect_metadata(0u, metadata.first_output_mesh_uniform_index);
        }
        return;
    }

    let block_size = min(WORKGROUP_SIZE, metadata.bin_count - block_start);
    if (local_id.x == 0u) {
        output_offsets[0] = select(0u, metadata.first_output_mesh_uniform_index, group_id.x == 0u);
    } else if (local_id.x < block_size) {
        output_offsets[local_id.x] = bin_metadata[global_id.x - 1u].instance_count;
    } else {
        output_offsets[local_id.x] = 0u;
    }
    workgroupBarrier();

    hillis_steele_scan(local_id.x, block_size);
    if (local_id.x < block_size) {
        write_indirect_metadata(global_id.x, output_offsets[local_id.x]);
    }
    if (local_id.x == block_size - 1u) {
        fan_buffer[group_id.x] = output_offsets[local_id.x] + bin_metadata[global_id.x].instance_count;
    }
}

fn write_indirect_metadata(bin_index: u32, base_output_index: u32) {
    let index = metadata.first_indirect_parameters_index +
        bin_metadata[bin_index].indirect_parameters_offset;
    indirect_parameters_metadata[index].base_output_index = base_output_index;
    indirect_parameters_metadata[index].batch_set_index = metadata.batch_set_index;
    indirect_parameters_metadata[index].mesh_index = 0u;
    indirect_parameters_metadata[index].early_instance_count = 0u;
    indirect_parameters_metadata[index].late_instance_count = 0u;
}

/// The second step of the prefix sum.
///
/// This step takes the intermediate fan values computed in the previous step
/// (i.e. the sum going out of each chunk) and performs one or more Hillis-Steele
/// scans in order to compute the fan value going into each chunk.
///
/// This step is omitted if there are 256 or fewer total draws.
@compute @workgroup_size(256, 1, 1)
fn allocate_global_scan(@builtin(local_invocation_id) local_id: vec3<u32>) {
    var sum = 0u;
    let chunk_count = div_ceil(metadata.bin_count, WORKGROUP_SIZE);

    // Do a sequential loop over each block of 256 chunks. Because each
    // iteration of this loop covers 64K meshes, the fact that it's sequential
    // isn't going to be a problem in practice.
    for (var block_start = 0u; block_start < chunk_count; block_start += WORKGROUP_SIZE) {
        // Set up the Hillis-Steele scan.
        let block_end = min(block_start + WORKGROUP_SIZE, chunk_count);
        let global_id = block_start + local_id.x;
        if (global_id < block_end) {
            output_offsets[local_id.x] = fan_buffer[global_id];
        }
        workgroupBarrier();

        // Perform the scan.
        hillis_steele_scan(local_id.x, WORKGROUP_SIZE);

        // Write the value back.
        // Note that we don't need a workgroup barrier here because
        // `hillis_steele_scan` already did one.
        if (global_id < block_end) {
            fan_buffer[global_id] = sum + output_offsets[local_id.x];
        }

        // Save the sum coming out of this block for the next one.
        sum += output_offsets[WORKGROUP_SIZE - 1u];
        workgroupBarrier();
    }
}

/// The third step of the prefix sum.
///
/// We take the summed fan value computed in the previous step and add it in to
/// each value of each chunk beyond the first. We dispatch one fewer workgroup
/// here than in step (1), because there's nothing to do for the first chunk.
///
/// This step is omitted if there are 256 or fewer total draws.
@compute @workgroup_size(256, 1, 1)
fn allocate_fan(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let id = global_id.x + WORKGROUP_SIZE;
    let bin_count = metadata.bin_count;
    if (id >= bin_count) {
        return;
    }

    let fan_value = fan_buffer[group_id.x];
    let indirect_parameters_offset =
        metadata.first_indirect_parameters_index +
        bin_metadata[id].indirect_parameters_offset;
    indirect_parameters_metadata[indirect_parameters_offset].base_output_index += fan_value;
}

/// Inclusive scan of shifted instance counts.
/// https://en.wikipedia.org/wiki/Prefix_sum#Algorithm_1:_Shorter_span,_more_parallel
fn hillis_steele_scan(local_id: u32, count: u32) {
    for (var offset = 1u; offset < count; offset *= 2u) {
        var term = 0u;
        if (local_id >= offset) {
            term = output_offsets[local_id - offset];
        }
        workgroupBarrier();
        output_offsets[local_id] += term;
        workgroupBarrier();
    }
}

/// Divides unsigned integer a by b, rounding up.
fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}
