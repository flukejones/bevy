use core::{iter, marker::PhantomData, ops::Range, slice};

use crate::{
    render_resource::{AtomicPod, Buffer},
    renderer::{RenderDevice, RenderQueue},
};
use bytemuck::{must_cast_slice, NoUninit};
use encase::{
    internal::{WriteInto, Writer},
    ShaderType,
};
use thiserror::Error;
use wgpu::{BindingResource, BufferAddress, BufferUsages};

use super::GpuArrayBufferable;

/// A structure for storing raw bytes that have already been properly formatted
/// for use by the GPU.
///
/// "Properly formatted" means that item data already meets the alignment and padding
/// requirements for how it will be used on the GPU. The item type must implement [`NoUninit`]
/// for its data representation to be directly copyable.
///
/// Index, vertex, and instance-rate vertex buffers have no alignment nor padding requirements and
/// so this helper type is a good choice for them.
///
/// The contained data is stored in system RAM. Calling [`reserve`](RawBufferVec::reserve)
/// allocates VRAM from the [`RenderDevice`].
/// [`write_buffer`](RawBufferVec::write_buffer) queues copying of the data
/// from system RAM to VRAM.
///
/// Other options for storing GPU-accessible data are:
/// * [`BufferVec`]
/// * [`DynamicStorageBuffer`](crate::render_resource::DynamicStorageBuffer)
/// * [`DynamicUniformBuffer`](crate::render_resource::DynamicUniformBuffer)
/// * [`GpuArrayBuffer`](crate::render_resource::GpuArrayBuffer)
/// * [`StorageBuffer`](crate::render_resource::StorageBuffer)
/// * [`Texture`](crate::render_resource::Texture)
/// * [`UniformBuffer`](crate::render_resource::UniformBuffer)
pub struct RawBufferVec<T: NoUninit> {
    values: Vec<T>,
    buffer: Option<Buffer>,
    capacity: usize,
    item_size: usize,
    buffer_usage: BufferUsages,
    label: Option<String>,
    changed: bool,
    values_changed: bool,
}

impl<T: NoUninit> RawBufferVec<T> {
    /// Creates a new [`RawBufferVec`] with the given [`BufferUsages`].
    pub const fn new(buffer_usage: BufferUsages) -> Self {
        Self {
            values: Vec::new(),
            buffer: None,
            capacity: 0,
            item_size: size_of::<T>(),
            buffer_usage,
            label: None,
            changed: false,
            values_changed: false,
        }
    }

    /// Returns a handle to the buffer, if the data has been uploaded.
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Returns the binding for the buffer if the data has been uploaded.
    #[inline]
    pub fn binding(&self) -> Option<BindingResource<'_>> {
        Some(BindingResource::Buffer(
            self.buffer()?.as_entire_buffer_binding(),
        ))
    }

    /// Returns the amount of space that the GPU will use before reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of items that have been pushed to this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Adds a new value and returns its index.
    pub fn push(&mut self, value: T) -> usize {
        let index = self.values.len();
        self.values.push(value);
        self.values_changed = true;
        index
    }

    pub fn append(&mut self, other: &mut RawBufferVec<T>) {
        self.values_changed |= !other.is_empty();
        other.values_changed |= !other.is_empty();
        self.values.append(&mut other.values);
    }

    /// Returns the value at the given index.
    pub fn get(&self, index: u32) -> Option<&T> {
        self.values.get(index as usize)
    }

    /// Sets the value at the given index.
    ///
    /// The index must be less than [`RawBufferVec::len`].
    pub fn set(&mut self, index: u32, value: T) {
        self.values[index as usize] = value;
        self.values_changed = true;
    }

    /// Preallocates space for `count` elements in the internal CPU-side buffer.
    ///
    /// Unlike [`RawBufferVec::reserve`], this doesn't have any effect on the GPU buffer.
    pub fn reserve_internal(&mut self, count: usize) {
        self.values.reserve(count);
    }

    /// Changes the debugging label of the buffer.
    ///
    /// The next time the buffer is updated (via [`reserve`](Self::reserve)), Bevy will inform
    /// the driver of the new label.
    pub fn set_label(&mut self, label: Option<&str>) {
        let label = label.map(str::to_string);

        if label != self.label {
            self.changed = true;
        }

        self.label = label;
    }

    /// Returns the label
    pub fn get_label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Creates a [`Buffer`] on the [`RenderDevice`] with size
    /// at least `size_of::<T>() * capacity`, unless a such a buffer already exists.
    ///
    /// If a [`Buffer`] exists, but is too small, references to it will be discarded,
    /// and a new [`Buffer`] will be created. Any previously created [`Buffer`]s
    /// that are no longer referenced will be deleted by the [`RenderDevice`]
    /// once it is done using them (typically 1-2 frames).
    ///
    /// In addition to any [`BufferUsages`] provided when
    /// the `RawBufferVec` was created, the buffer on the [`RenderDevice`]
    /// is marked as [`BufferUsages::COPY_DST`](BufferUsages).
    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) {
        let size = self.item_size * capacity;
        if capacity > self.capacity || (self.changed && size > 0) {
            self.capacity = capacity;
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: make_buffer_label::<Self>(&self.label),
                size: size as BufferAddress,
                usage: BufferUsages::COPY_DST | self.buffer_usage,
                mapped_at_creation: false,
            }));
            self.changed = false;
            self.values_changed = true;
        }
    }

    /// Queues writing of data from system RAM to VRAM using the [`RenderDevice`]
    /// and the provided [`RenderQueue`].
    ///
    /// Before queuing the write, a [`reserve`](RawBufferVec::reserve) operation
    /// is executed.
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.values.is_empty() {
            return;
        }
        self.reserve(self.values.len(), device);
        if let Some(buffer) = &self.buffer {
            let range = 0..self.item_size * self.values.len();
            let bytes: &[u8] = must_cast_slice(&self.values);
            queue.write_buffer(buffer, 0, &bytes[range]);
            self.values_changed = false;
        }
    }

    /// Uploads CPU-authored data only after mutation or backing-buffer replacement.
    /// The callback must encode the complete upload before the buffer's consumers.
    /// GPU-written contents are not tracked; use [`Self::write_buffer`] to reset them.
    pub(crate) fn write_buffer_if_changed(
        &mut self,
        device: &RenderDevice,
        write: impl FnOnce(&Buffer, &[u8]),
    ) {
        if self.values.is_empty() {
            return;
        }
        self.reserve(self.values.len(), device);
        if self.values_changed
            && let Some(buffer) = &self.buffer
        {
            write(buffer, must_cast_slice(&self.values));
            self.values_changed = false;
        }
    }

    /// Queues writing of data from system RAM to VRAM using the [`RenderDevice`]
    /// and the provided [`RenderQueue`].
    ///
    /// If the buffer is not initialized on the GPU or the range is bigger than the capacity it will
    /// return an error. You'll need to either reserve a new buffer which will lose data on the GPU
    /// or create a new buffer and copy the old data to it.
    ///
    /// This will only write the data contained in the given range. It is useful if you only want
    /// to update a part of the buffer.
    pub fn write_buffer_range(
        &mut self,
        render_queue: &RenderQueue,
        range: Range<usize>,
    ) -> Result<(), WriteBufferRangeError> {
        if self.values.is_empty() {
            return Err(WriteBufferRangeError::NoValuesToUpload);
        }
        if range.end > self.item_size * self.capacity {
            return Err(WriteBufferRangeError::RangeBiggerThanBuffer);
        }
        if let Some(buffer) = &self.buffer {
            // Cast only the bytes we need to write
            let bytes: &[u8] = must_cast_slice(&self.values[range.start..range.end]);
            render_queue.write_buffer(buffer, (range.start * self.item_size) as u64, bytes);
            Ok(())
        } else {
            Err(WriteBufferRangeError::BufferNotInitialized)
        }
    }

    /// Reduces the length of the buffer.
    pub fn truncate(&mut self, len: usize) {
        self.values_changed |= len < self.values.len();
        self.values.truncate(len);
    }

    /// Removes all elements from the buffer.
    pub fn clear(&mut self) {
        self.values_changed |= !self.values.is_empty();
        self.values.clear();
    }

    /// Removes and returns the last element in the buffer.
    pub fn pop(&mut self) -> Option<T> {
        let value = self.values.pop();
        self.values_changed |= value.is_some();
        value
    }

    pub fn swap_remove(&mut self, index: usize) -> T {
        self.values_changed = true;
        self.values.swap_remove(index)
    }

    pub fn values(&self) -> &Vec<T> {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut Vec<T> {
        self.values_changed = true;
        &mut self.values
    }
}

impl<T> RawBufferVec<T>
where
    T: NoUninit + Default,
{
    pub fn grow_set(&mut self, index: u32, value: T) {
        self.values.reserve(index as usize + 1);
        self.values_changed = true;
        while index as usize + 1 > self.len() {
            self.values.push(T::default());
        }
        self.values[index as usize] = value;
    }
}

impl<T: NoUninit> Extend<T> for RawBufferVec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.values_changed = true;
        self.values.extend(iter);
    }
}

/// A [`RawBufferVec`] that holds data that implements [`AtomicPod`].
///
/// This allows multiple threads to update the buffer to be sent to the GPU
/// simultaneously. Note that they may only update *existing* data; pushing
/// *new* data still requires exclusive access.
pub struct AtomicRawBufferVec<T>
where
    T: AtomicPod,
{
    /// The underlying values.
    ///
    /// These are stored as their blob representation to allow for thread-safe
    /// update.
    values: Vec<T::Blob>,
    /// The GPU buffer, if allocated.
    buffer: Option<Buffer>,
    /// The capacity of the GPU buffer.
    capacity: usize,
    /// The allowed `wgpu` buffer usages for the GPU buffer.
    buffer_usage: BufferUsages,
    /// An optional debug label to identify this buffer.
    label: Option<String>,
    /// Whether the buffer has been mutated on the CPU since the last time it
    /// was uploaded to the GPU.
    changed: bool,
    phantom: PhantomData<T>,
}

impl<T> AtomicRawBufferVec<T>
where
    T: AtomicPod,
{
    /// Creates a new [`AtomicRawBufferVec`].
    ///
    /// The `buffer_usage` parameter tells `wgpu` which usages are allowed for
    /// the backing buffer.
    pub const fn new(buffer_usage: BufferUsages) -> Self {
        Self {
            values: Vec::new(),
            buffer: None,
            capacity: 0,
            buffer_usage,
            label: None,
            changed: false,
            phantom: PhantomData,
        }
    }

    /// Creates a new [`AtomicRawBufferVec`] with a custom label.
    ///
    /// The `buffer_usage` parameter tells `wgpu` which usages are allowed for
    /// the backing buffer.
    pub fn with_label(buffer_usage: BufferUsages, label: &str) -> Self {
        Self {
            values: Vec::new(),
            buffer: None,
            capacity: 0,
            buffer_usage,
            label: Some(label.to_string()),
            changed: false,
            phantom: PhantomData,
        }
    }

    /// Removes all elements from the buffer.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> u32 {
        self.values.len() as u32
    }

    /// Returns true if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds a new value to the buffer, and returns its index.
    ///
    /// Internally, the value is converted to its blob representation.
    pub fn push(&mut self, value: T) -> u32 {
        let index = self.values.len();
        self.values.push(T::Blob::default());
        value.write_to_blob(&self.values[index]);
        index as u32
    }

    /// Copies a value out of the buffer.
    pub fn get(&self, index: u32) -> T {
        T::read_from_blob(&self.values[index as usize])
    }

    /// Sets the value at the given index.
    ///
    /// If the index isn't in range of the buffer, this method panics.
    ///
    /// Internally, the value is converted to its blob representation.
    ///
    /// Note that this method is thread-safe and doesn't require `&mut self`.
    /// It's your responsibility, however, to ensure synchronization; though
    /// this method is memory-safe, it's possible for other threads to observe
    /// partially-overwritten values if [`Self::get`] or similar methods are
    /// called while the write operation is occurring.
    pub fn set(&self, index: u32, value: T) {
        value.write_to_blob(&self.values[index as usize]);
    }

    /// Creates a [`Buffer`] on the [`RenderDevice`] with size
    /// at least `size_of::<T>() * capacity`, unless a such a buffer already exists.
    ///
    /// If a [`Buffer`] exists, but is too small, references to it will be discarded,
    /// and a new [`Buffer`] will be created. Any previously created [`Buffer`]s
    /// that are no longer referenced will be deleted by the [`RenderDevice`]
    /// once it is done using them (typically 1-2 frames).
    ///
    /// In addition to any [`BufferUsages`] provided when
    /// the `AtomicRawBufferVec` was created, the buffer on the [`RenderDevice`]
    /// is marked as [`BufferUsages::COPY_DST`](BufferUsages).
    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) {
        let size = size_of::<T::Blob>() * capacity;
        if capacity > self.capacity || (self.changed && size > 0) {
            self.capacity = capacity;
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: make_buffer_label::<Self>(&self.label),
                size: size as BufferAddress,
                usage: BufferUsages::COPY_DST | self.buffer_usage,
                mapped_at_creation: false,
            }));
            self.changed = false;
        }
    }

    /// Queues writing of data from system RAM to VRAM using the
    /// [`RenderDevice`] and the provided [`RenderQueue`].
    ///
    /// Before queuing the write, a [`reserve`](AtomicRawBufferVec::reserve)
    /// operation is executed.
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.write_buffer_range(0..self.values.len(), device, queue);
    }

    /// Queues writing of data from system RAM to VRAM using the
    /// [`RenderDevice`] and the provided [`RenderQueue`].
    ///
    /// Before queuing the write, a [`reserve`](AtomicRawBufferVec::reserve)
    /// operation is executed.
    pub fn write_buffer_range(
        &mut self,
        range: Range<usize>,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) {
        assert!(
            range.start <= range.end
                && range.start <= self.values.len()
                && range.end <= self.values.len()
        );
        if range.start == range.end {
            return;
        }
        self.reserve(range.end, device);

        let Some(buffer) = &self.buffer else { return };

        // SAFETY: We checked the range above to make sure it's in bounds.
        // We have `&mut self`, so there are no other references to our
        // buffer, and the `Blob` type must implement `AtomicPodBlob`, which
        // guarantees that it be bit-equivalent to an array of `AtomicU32`s
        // (i.e. POD except that they're atomic).
        unsafe {
            let bytes: &[u8] = slice::from_raw_parts(
                self.values.as_ptr().add(range.start).cast::<u8>(),
                (range.end - range.start) * size_of::<T::Blob>(),
            );
            let start_offset = range.start as u64 * size_of::<T::Blob>() as u64;
            queue.write_buffer(buffer, start_offset, bytes);
        }
    }

    /// Returns a handle to the buffer, if the data has been uploaded.
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Grows the buffer by adding default values so that it's at least the
    /// given size.
    ///
    /// If the buffer is already large enough, this method does nothing.
    pub fn grow(&mut self, new_len: u32) {
        if self.len() < new_len {
            self.values.resize_with(new_len as usize, T::Blob::default);
        }
    }

    /// Truncates the buffer to the given length.
    ///
    /// If the buffer is already shorter, this method does nothing.
    pub fn truncate(&mut self, len: u32) {
        self.values.truncate(len as usize);
    }
}

/// Like [`RawBufferVec`], but doesn't require that the data type `T` be
/// [`NoUninit`].
///
/// This is a high-performance data structure that you should use whenever
/// possible if your data is more complex than is suitable for [`RawBufferVec`].
/// The [`ShaderType`] trait from the `encase` library is used to ensure that
/// the data is correctly aligned for use by the GPU.
///
/// For performance reasons, unlike [`RawBufferVec`], this type doesn't allow
/// CPU access to the data after it's been added via [`BufferVec::push`]. If you
/// need CPU access to the data, consider another type, such as
/// [`StorageBuffer`][super::StorageBuffer].
///
/// Other options for storing GPU-accessible data are:
/// * [`DynamicStorageBuffer`](crate::render_resource::DynamicStorageBuffer)
/// * [`DynamicUniformBuffer`](crate::render_resource::DynamicUniformBuffer)
/// * [`GpuArrayBuffer`](crate::render_resource::GpuArrayBuffer)
/// * [`RawBufferVec`]
/// * [`StorageBuffer`](crate::render_resource::StorageBuffer)
/// * [`Texture`](crate::render_resource::Texture)
/// * [`UniformBuffer`](crate::render_resource::UniformBuffer)
pub struct BufferVec<T>
where
    T: ShaderType + WriteInto,
{
    data: Vec<u8>,
    buffer: Option<Buffer>,
    capacity: usize,
    buffer_usage: BufferUsages,
    label: Option<String>,
    label_changed: bool,
    phantom: PhantomData<T>,
}

impl<T> BufferVec<T>
where
    T: ShaderType + WriteInto,
{
    /// Creates a new [`BufferVec`] with the given [`BufferUsages`].
    pub const fn new(buffer_usage: BufferUsages) -> Self {
        Self {
            data: vec![],
            buffer: None,
            capacity: 0,
            buffer_usage,
            label: None,
            label_changed: false,
            phantom: PhantomData,
        }
    }

    /// Returns a handle to the buffer, if the data has been uploaded.
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Returns the binding for the buffer if the data has been uploaded.
    #[inline]
    pub fn binding(&self) -> Option<BindingResource<'_>> {
        Some(BindingResource::Buffer(
            self.buffer()?.as_entire_buffer_binding(),
        ))
    }

    /// Returns the amount of space that the GPU will use before reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of items that have been pushed to this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len() / u64::from(T::min_size()) as usize
    }

    /// Returns true if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Adds a new value and returns its index.
    pub fn push(&mut self, value: T) -> usize {
        let element_size = u64::from(T::min_size()) as usize;
        let offset = self.data.len();

        // We can't optimize and push uninitialized data here (using e.g. spare_capacity_mut())
        // because write_into() does not initialize inner padding bytes in T's expansion
        self.data.extend(iter::repeat_n(0, element_size));

        // Take a slice of the new data for `write_into` to use. This is
        // important: it hoists the bounds check up here so that the compiler
        // can eliminate all the bounds checks that `write_into` will emit.
        let mut dest = &mut self.data[offset..(offset + element_size)];
        value.write_into(&mut Writer::new(&value, &mut dest, 0).unwrap());

        offset / u64::from(T::min_size()) as usize
    }

    /// Changes the debugging label of the buffer.
    ///
    /// The next time the buffer is updated (via [`Self::reserve`]), Bevy will inform
    /// the driver of the new label.
    pub fn set_label(&mut self, label: Option<&str>) {
        let label = label.map(str::to_string);

        if label != self.label {
            self.label_changed = true;
        }

        self.label = label;
    }

    /// Returns the label
    pub fn get_label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Preallocates space for `count` elements in the internal CPU-side buffer.
    ///
    /// Unlike [`Self::reserve`], this doesn't have any effect on the GPU buffer.
    pub fn reserve_internal(&mut self, count: usize) {
        self.data.reserve(count * u64::from(T::min_size()) as usize);
    }

    /// Creates a [`Buffer`] on the [`RenderDevice`] with size
    /// at least `size_of::<T>() * capacity`, unless such a buffer already exists.
    ///
    /// If a [`Buffer`] exists, but is too small, references to it will be discarded,
    /// and a new [`Buffer`] will be created. Any previously created [`Buffer`]s
    /// that are no longer referenced will be deleted by the [`RenderDevice`]
    /// once it is done using them (typically 1-2 frames).
    ///
    /// In addition to any [`BufferUsages`] provided when
    /// the `BufferVec` was created, the buffer on the [`RenderDevice`]
    /// is marked as [`BufferUsages::COPY_DST`](BufferUsages).
    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) {
        if capacity <= self.capacity && !self.label_changed {
            return;
        }

        self.capacity = capacity;
        let size = u64::from(T::min_size()) as usize * capacity;
        self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: make_buffer_label::<Self>(&self.label),
            size: size as BufferAddress,
            usage: BufferUsages::COPY_DST | self.buffer_usage,
            mapped_at_creation: false,
        }));
        self.label_changed = false;
    }

    /// Queues writing of data from system RAM to VRAM using the [`RenderDevice`]
    /// and the provided [`RenderQueue`].
    ///
    /// Before queuing the write, a [`reserve`](BufferVec::reserve) operation is
    /// executed.
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.data.is_empty() {
            return;
        }

        self.reserve(self.data.len() / u64::from(T::min_size()) as usize, device);

        let Some(buffer) = &self.buffer else { return };
        queue.write_buffer(buffer, 0, &self.data);
    }

    /// Queues writing of data from system RAM to VRAM using the [`RenderDevice`]
    /// and the provided [`RenderQueue`].
    ///
    /// If the buffer is not initialized on the GPU or the range is bigger than the capacity it will
    /// return an error. You'll need to either reserve a new buffer which will lose data on the GPU
    /// or create a new buffer and copy the old data to it.
    ///
    /// This will only write the data contained in the given range. It is useful if you only want
    /// to update a part of the buffer.
    pub fn write_buffer_range(
        &mut self,
        render_queue: &RenderQueue,
        range: Range<usize>,
    ) -> Result<(), WriteBufferRangeError> {
        if self.data.is_empty() {
            return Err(WriteBufferRangeError::NoValuesToUpload);
        }
        let item_size = u64::from(T::min_size()) as usize;
        if range.end > item_size * self.capacity {
            return Err(WriteBufferRangeError::RangeBiggerThanBuffer);
        }
        if let Some(buffer) = &self.buffer {
            let bytes = &self.data[range.start..range.end];
            render_queue.write_buffer(buffer, (range.start * item_size) as u64, bytes);
            Ok(())
        } else {
            Err(WriteBufferRangeError::BufferNotInitialized)
        }
    }

    /// Reduces the length of the buffer.
    pub fn truncate(&mut self, len: usize) {
        self.data.truncate(u64::from(T::min_size()) as usize * len);
    }

    /// Removes all elements from the buffer.
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

/// Like a [`BufferVec`], but only reserves space on the GPU for elements
/// instead of initializing them CPU-side.
///
/// This type is useful when you're accumulating "output slots" for a GPU
/// compute shader to write into.
///
/// The type `T` need not be [`NoUninit`], unlike [`RawBufferVec`]; it only has to
/// be [`GpuArrayBufferable`].
pub struct UninitBufferVec<T>
where
    T: GpuArrayBufferable,
{
    buffer: Option<Buffer>,
    len: usize,
    capacity: usize,
    item_size: usize,
    buffer_usage: BufferUsages,
    label: Option<String>,
    label_changed: bool,
    phantom: PhantomData<T>,
}

impl<T> UninitBufferVec<T>
where
    T: GpuArrayBufferable,
{
    /// Creates a new [`UninitBufferVec`] with the given [`BufferUsages`].
    pub const fn new(buffer_usage: BufferUsages) -> Self {
        Self {
            len: 0,
            buffer: None,
            capacity: 0,
            item_size: T::SHADER_SIZE.get() as usize,
            buffer_usage,
            label: None,
            label_changed: false,
            phantom: PhantomData,
        }
    }

    /// Returns the buffer, if allocated.
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Returns the binding for the buffer if the data has been uploaded.
    #[inline]
    pub fn binding(&self) -> Option<BindingResource<'_>> {
        Some(BindingResource::Buffer(
            self.buffer()?.as_entire_buffer_binding(),
        ))
    }

    /// Reserves space for one more element in the buffer and returns its index.
    pub fn add(&mut self) -> usize {
        self.add_multiple(1)
    }

    /// Reserves space for the given number of elements in the buffer and
    /// returns the index of the first one.
    pub fn add_multiple(&mut self, count: usize) -> usize {
        let index = self.len;
        self.len += count;
        index
    }

    /// Returns true if no elements have been added to this [`UninitBufferVec`].
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Removes all elements from the buffer.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Returns the length of the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the amount of space that the GPU will use before reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Changes the debugging label of the buffer.
    ///
    /// The next time the buffer is updated (via [`Self::reserve`]), Bevy will inform
    /// the driver of the new label.
    pub fn set_label(&mut self, label: Option<&str>) {
        let label = label.map(str::to_string);

        if label != self.label {
            self.label_changed = true;
        }

        self.label = label;
    }

    /// Returns the label
    pub fn get_label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Materializes the buffer on the GPU with space for `capacity` elements.
    ///
    /// If the buffer is already big enough, this function doesn't reallocate
    /// the buffer.
    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) {
        if capacity <= self.capacity && !self.label_changed {
            return;
        }

        self.capacity = capacity;
        let size = self.item_size * capacity;
        self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: make_buffer_label::<Self>(&self.label),
            size: size as BufferAddress,
            usage: BufferUsages::COPY_DST | self.buffer_usage,
            mapped_at_creation: false,
        }));

        self.label_changed = false;
    }

    /// Materializes the buffer on the GPU, with an appropriate size for the
    /// elements that have been pushed so far.
    pub fn write_buffer(&mut self, device: &RenderDevice) {
        if !self.is_empty() {
            self.reserve(self.len, device);
        }
    }
}

/// A hybrid of [`RawBufferVec`] and [`UninitBufferVec`] that allows the CPU to
/// push elements and to leave room for uninitialized elements for the GPU to
/// populate at the end of the array.
///
/// All CPU elements mush be pushed *before* any trailing uninitialized elements
/// can be reserved. In debug mode, this data structure enforces these
/// preconditions with assertions.
pub struct PartialBufferVec<T>
where
    T: NoUninit,
{
    /// The CPU-side values.
    values: Vec<T>,
    /// The GPU buffer, if allocated.
    buffer: Option<Buffer>,
    /// The total number of elements that have been allocated in the GPU buffer.
    capacity: usize,
    /// The number of extra uninitialized elements at the end.
    ///
    /// Thus the total needed length of the buffer is `self.values.len() +
    /// self.uninit_element_count`.
    uninit_element_count: usize,
    /// The allowed `wgpu` usages of the buffer.
    buffer_usages: BufferUsages,
    /// An identifying debugging label for this buffer.
    label: String,
}

impl<T> PartialBufferVec<T>
where
    T: NoUninit,
{
    /// Creates a new [`PartialBufferVec`] with the given allowed usages and the
    /// given debugging label.
    ///
    /// `BufferUsages::COPY_DST` is implicitly added to the supplied set of
    /// usages.
    pub fn new(buffer_usages: BufferUsages, label: String) -> PartialBufferVec<T> {
        PartialBufferVec {
            values: vec![],
            buffer: None,
            capacity: 0,
            uninit_element_count: 0,
            buffer_usages: buffer_usages | BufferUsages::COPY_DST,
            label,
        }
    }

    /// Returns the allocated GPU buffer, if one exists.
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Clears out the buffer, setting both the number of CPU-initialized
    /// elements and extra uninitialized trailing elements to 0.
    pub fn clear(&mut self) {
        self.values.clear();
        self.uninit_element_count = 0;
    }

    /// Ensures that the GPU buffer is allocated with the given capacity.
    ///
    /// If the cached GPU buffer is already big enough, this method does
    /// nothing.
    fn reserve(&mut self, capacity: usize, render_device: &RenderDevice) {
        if capacity <= self.capacity {
            return;
        }

        let size = size_of::<T>() * capacity;
        self.capacity = capacity;
        self.buffer = Some(render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: size as u64,
            usage: self.buffer_usages,
            mapped_at_creation: false,
        }));
    }

    /// Writes the buffer to the GPU.
    ///
    /// `Self::reserve` is called automatically to ensure that the buffer has
    /// the correct length (including enough space to hold all the trailing
    /// uninitialized elements).
    pub fn write_buffer(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        if self.is_empty() {
            return;
        }

        self.reserve(self.len(), render_device);

        let Some(ref buffer) = self.buffer else {
            return;
        };
        render_queue.write_buffer(buffer, 0, must_cast_slice(&self.values[..]));
    }

    /// Returns true if this buffer is empty: i.e. if [`Self::len`] would return
    /// 0.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the total number of elements, both initialized and
    /// uninitialized, in this buffer.
    ///
    /// That is, this method returns the sum of the number of initialized values
    /// that the CPU pushed and the number of trailing uninitialized elements
    /// that have been allocated.
    pub fn len(&self) -> usize {
        self.values.len() + self.uninit_element_count
    }

    /// Pushes an element with the given value to the buffer and returns its
    /// index.
    ///
    /// Since this element is initialized by the CPU, and all CPU-initialized
    /// elements must precede any trailing uninitialized elements, this method
    /// panics in debug mode if [`Self::push_multiple_uninit`] has been called
    /// since the last call to [`Self::clear`].
    pub fn push_init(&mut self, value: T) -> usize {
        debug_assert_eq!(self.uninit_element_count, 0);
        let index = self.values.len();
        self.values.push(value);
        index
    }

    /// Pushes the given number of uninitialized elements to the end of the list
    /// and returns the index of the first such element that was pushed.
    ///
    /// After calling this method with a nonzero `count`, it's no longer legal
    /// to call [`Self::push_init`] without calling [`Self::clear`] first.
    pub fn push_multiple_uninit(&mut self, count: usize) -> usize {
        let first_index = self.values.len() + self.uninit_element_count;
        self.uninit_element_count += count;
        first_index
    }

    /// Sets the value at the given index to the given value.
    pub fn set(&mut self, index: usize, value: T) {
        self.values[index] = value;
    }
}

impl<T> PartialBufferVec<T>
where
    T: NoUninit + Default,
{
    /// Pushes `count` copies of `T::default` to the array.
    pub fn push_multiple_init(&mut self, count: usize) -> usize {
        debug_assert_eq!(self.uninit_element_count, 0);
        let index = self.values.len();
        self.values.extend(iter::repeat_n(T::default(), count));
        index
    }
}

/// Error returned when `write_buffer_range` fails
///
/// See [`RawBufferVec::write_buffer_range`] [`BufferVec::write_buffer_range`]
#[derive(Debug, Eq, PartialEq, Copy, Clone, Error)]
pub enum WriteBufferRangeError {
    #[error("the range is bigger than the capacity of the buffer")]
    RangeBiggerThanBuffer,
    #[error("the gpu buffer is not initialized")]
    BufferNotInitialized,
    #[error("there are no values to upload")]
    NoValuesToUpload,
}

#[inline]
#[cfg_attr(
    not(feature = "type_label_buffers"),
    expect(
        clippy::extra_unused_type_parameters,
        reason = "conditional compilation"
    )
)]
pub(crate) fn make_buffer_label<'a, T>(label: &'a Option<String>) -> Option<&'a str> {
    #[cfg(feature = "type_label_buffers")]
    if label.is_none() {
        return Some(core::any::type_name::<T>());
    }
    label.as_deref()
}

#[cfg(test)]
mod tests {
    use super::{RawBufferVec, UninitBufferVec};
    use crate::{
        render_resource::Buffer,
        renderer::{RenderDevice, RenderInstance, RenderQueue},
    };
    use bevy_math::Vec3;
    use bevy_platform::future::block_on;
    use bytemuck::cast_slice;
    use core::iter::once_with;
    use encase::ShaderType;
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::mpsc::channel,
        time::Duration,
    };
    use wgpu::{BufferDescriptor, BufferUsages, DeviceDescriptor, MapMode, PollType};

    #[derive(ShaderType, Clone)]
    struct Padded {
        value: Vec3,
    }

    #[test]
    fn uninit_buffer_vec_uses_shader_size() {
        // A `Vec3` is 12 bytes on the CPU but the struct is 16 bytes in WGSL.
        assert_eq!(size_of::<Padded>(), 12);
        let buffer = UninitBufferVec::<Padded>::new(BufferUsages::STORAGE);
        assert_eq!(buffer.item_size, 16);
    }

    #[test]
    #[ignore = "requires a graphics adapter"]
    fn cached_uploads_track_mutations_and_backing_replacement() {
        let instance = RenderInstance::new(Default::default());
        let adapter = block_on(instance.request_adapter(&Default::default()))
            .expect("graphics adapter for buffer uploads");
        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor::default()))
            .expect("device for buffer uploads");
        let device = RenderDevice::new(device);
        let queue = RenderQueue::new(queue);
        let write = |buffer: &Buffer, bytes: &[u8]| queue.write_buffer(buffer, 0, bytes);
        let mut values = RawBufferVec::new(BufferUsages::COPY_SRC);
        values.extend([11u32, 29, 47]);
        values.write_buffer_if_changed(&device, write);
        let original = values.buffer().expect("initial uploaded buffer").clone();
        assert_eq!(read_words(&device, &queue, &original, 3), [11, 29, 47]);

        queue.write_buffer(&original, 0, cast_slice(&[101u32, 103, 107]));
        values.write_buffer_if_changed(&device, |_, _| panic!("unchanged values were uploaded"));
        assert_eq!(read_words(&device, &queue, &original, 3), [101, 103, 107]);

        values.set(0, 13);
        values.write_buffer_if_changed(&device, write);
        assert_eq!(read_words(&device, &queue, &original, 3), [13, 29, 47]);
        values.values_mut()[1] = 31;
        values.write_buffer_if_changed(&device, write);
        assert_eq!(read_words(&device, &queue, &original, 3), [13, 31, 47]);

        values.reserve(6, &device);
        values.write_buffer_if_changed(&device, write);
        let grown = values.buffer().expect("grown uploaded buffer").clone();
        assert_ne!(grown.id(), original.id());
        assert_eq!(read_words(&device, &queue, &grown, 3), [13, 31, 47]);
        assert_eq!(read_words(&device, &queue, &original, 3), [13, 31, 47]);

        values.set_label(Some("replacement backing"));
        values.write_buffer_if_changed(&device, write);
        let replacement = values.buffer().expect("relabeled uploaded buffer").clone();
        assert_ne!(replacement.id(), grown.id());
        assert_eq!(read_words(&device, &queue, &replacement, 3), [13, 31, 47]);

        values.swap_remove(0);
        values.write_buffer_if_changed(&device, write);
        assert_eq!(read_words(&device, &queue, &replacement, 2), [47, 31]);
        values.pop();
        values.write_buffer_if_changed(&device, write);
        assert_eq!(read_words(&device, &queue, &replacement, 1), [47]);
        let mut appended = RawBufferVec::new(BufferUsages::COPY_SRC);
        appended.extend([53, 59]);
        values.append(&mut appended);
        values.write_buffer_if_changed(&device, write);
        assert_eq!(read_words(&device, &queue, &replacement, 3), [47, 53, 59]);
        values.push(61);
        values.write_buffer_if_changed(&device, write);
        let replacement = values.buffer().expect("appended uploaded buffer").clone();
        assert_eq!(
            read_words(&device, &queue, &replacement, 4),
            [47, 53, 59, 61]
        );
        values.truncate(3);
        values.write_buffer_if_changed(&device, write);
        assert_eq!(read_words(&device, &queue, &replacement, 3), [47, 53, 59]);

        values.clear();
        values.write_buffer_if_changed(&device, |_, _| panic!("empty values were uploaded"));
        values.grow_set(2, 71);
        values.write_buffer_if_changed(&device, write);
        assert_eq!(read_words(&device, &queue, &replacement, 3), [0, 0, 71]);

        queue.write_buffer(&replacement, 0, cast_slice(&[73u32, 79, 83]));
        values.write_buffer(&device, &queue);
        assert_eq!(read_words(&device, &queue, &replacement, 3), [0, 0, 71]);
        queue.write_buffer(&replacement, 0, cast_slice(&[79u32, 83, 89]));
        values.write_buffer_if_changed(&device, |_, _| panic!("complete upload remained dirty"));
        assert_eq!(read_words(&device, &queue, &replacement, 3), [79, 83, 89]);

        catch_unwind(AssertUnwindSafe(|| {
            values.extend(
                [97].into_iter()
                    .chain(once_with(|| panic!("failed source"))),
            );
        }))
        .expect_err("source iterator fails after appending one value");
        values.write_buffer_if_changed(&device, write);
        assert_eq!(read_words(&device, &queue, &replacement, 4), [0, 0, 71, 97]);
    }

    fn read_words(
        device: &RenderDevice,
        queue: &RenderQueue,
        buffer: &Buffer,
        count: u64,
    ) -> Vec<u32> {
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("buffer upload readback"),
            size: count * 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, count * 4);
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = channel();
        slice.map_async(MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("buffer upload readback receiver");
        });
        device
            .poll(PollType::wait_indefinitely())
            .expect("buffer upload GPU completion");
        receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("buffer upload readback callback")
            .expect("mapped buffer upload readback");
        let mapped = slice
            .get_mapped_range()
            .expect("buffer upload readback bytes");
        let words = cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        words
    }
}
