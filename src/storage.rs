//! Backing storage for a GGUF file, and the zero-copy slice path.

use alloc::rc::Rc;
#[cfg(all(feature = "std", feature = "memmap"))]
use burn_tensor::AllocationProperty;
use burn_tensor::Bytes;

#[cfg(feature = "std")]
use crate::base::GgufError;

/// Where a GGUF file's bytes come from.
///
/// GGUF is a single flat file with an offset table, which is exactly the shape that
/// memory-mapping handles well: the header is parsed eagerly and tensor payloads are
/// handed out as subranges of the mapping without copying.
pub(crate) enum Storage {
    /// The whole file is resident and sliceable.
    ///
    /// `Bytes` is backed by a refcounted `bytes::Bytes` when it was built with
    /// [`Bytes::from_shared`], so [`Storage::slice`] is a refcount bump rather than a
    /// memcpy.
    Memory(Rc<Bytes>),
    /// A file opened for buffered reads, used when memory-mapping is unavailable.
    ///
    /// Only constructed when the memmap feature is off, so with the default features
    /// this arm is dead code that still has to compile and stay tested.
    #[cfg(feature = "std")]
    #[cfg_attr(feature = "memmap", allow(dead_code))]
    Buffered {
        file: Rc<std::cell::RefCell<std::fs::File>>,
    },
}

impl core::fmt::Debug for Storage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            // Deliberately not printing the buffer: a mapped model is gigabytes of data.
            Self::Memory(bytes) => f
                .debug_struct("Memory")
                .field("len", &bytes.len())
                .finish_non_exhaustive(),
            #[cfg(feature = "std")]
            Self::Buffered { file } => f
                .debug_struct("Buffered")
                .field("len", &file.borrow().metadata().ok().map(|m| m.len()))
                .finish_non_exhaustive(),
        }
    }
}

impl Storage {
    /// Wrap an already-resident buffer, enabling zero-copy slices.
    pub(crate) fn from_bytes(bytes: Bytes) -> Self {
        Self::Memory(Rc::new(bytes))
    }

    /// Memory-map a file, enabling zero-copy slices.
    ///
    /// The mapping is moved into a `bytes::Bytes` so that it can be shared as `Send +
    /// Sync + 'static` and sliced cheaply, and so that GPU backends can see the
    /// [`AllocationProperty::File`] hint and stage transfers through pinned memory.
    #[cfg(all(feature = "std", feature = "memmap"))]
    pub(crate) fn from_mmap(path: &std::path::Path) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path).map_err(|error| {
            GgufError::Io(format!("failed to open {}: {error}", path.display()))
        })?;
        // A zero-length mapping is rejected by memmap2, and it is not a GGUF file either.
        let len = file
            .metadata()
            .map_err(|error| GgufError::Io(format!("failed to stat {}: {error}", path.display())))?
            .len();
        if len == 0 {
            return Err(GgufError::Io(format!("{} is empty", path.display())));
        }
        if len > usize::MAX as u64 {
            return Err(GgufError::Io(format!(
                "{} is {len} bytes, which does not fit in memory on this host",
                path.display()
            )));
        }

        let mapping = unsafe {
            memmap2::MmapOptions::new().map(&file).map_err(|error| {
                GgufError::Io(format!("failed to map {}: {error}", path.display()))
            })?
        };

        let shared = bytes::Bytes::from_owner(mapping);
        Ok(Self::from_bytes(Bytes::from_shared(
            shared,
            AllocationProperty::File,
        )))
    }

    /// Open a file for buffered, copying reads.
    #[cfg(feature = "std")]
    #[cfg_attr(feature = "memmap", allow(dead_code))]
    pub(crate) fn from_file(path: &std::path::Path) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path).map_err(|error| {
            GgufError::Io(format!("failed to open {}: {error}", path.display()))
        })?;
        Ok(Self::Buffered {
            file: Rc::new(std::cell::RefCell::new(file)),
        })
    }

    /// The length of the file in bytes, when it is known without reading.
    #[cfg(feature = "std")]
    pub(crate) fn len(&self) -> Option<usize> {
        match self {
            Self::Memory(bytes) => Some(bytes.len()),
            #[cfg(feature = "std")]
            Self::Buffered { file } => file
                .borrow()
                .metadata()
                .ok()
                .map(|meta| meta.len() as usize),
        }
    }

    /// Read `len` bytes at `offset` into a new vector.
    ///
    /// This always copies. Use [`Storage::slice`] when the data will end up in a
    /// [`burn_tensor::TensorData`] and no copy is wanted.
    #[cfg(feature = "std")]
    pub(crate) fn read_vec(&self, offset: usize, len: usize) -> Result<Vec<u8>, GgufError> {
        let mut out = vec![0u8; len];
        self.read_into(&mut out, offset)?;
        Ok(out)
    }

    /// Copy `buf.len()` bytes from `offset` into `buf`.
    #[cfg(feature = "std")]
    pub(crate) fn read_into(&self, buf: &mut [u8], offset: usize) -> Result<(), GgufError> {
        match self {
            Self::Memory(bytes) => {
                let data = bytes.as_ref();
                let end = offset.checked_add(buf.len()).ok_or_else(|| {
                    GgufError::Io(format!(
                        "offset {offset} plus {} bytes overflows",
                        buf.len()
                    ))
                })?;
                if end > data.len() {
                    return Err(GgufError::UnexpectedEof {
                        context: "tensor data",
                        offset,
                        needed: buf.len(),
                        available: data.len().saturating_sub(offset),
                    });
                }
                buf.copy_from_slice(&data[offset..end]);
                Ok(())
            }
            Self::Buffered { file } => {
                use std::io::{Read, Seek, SeekFrom};
                let mut file = file.borrow_mut();
                file.seek(SeekFrom::Start(offset as u64)).map_err(|error| {
                    GgufError::Io(format!("failed to seek to {offset}: {error}"))
                })?;
                file.read_exact(buf).map_err(|error| {
                    GgufError::Io(format!(
                        "failed to read {} bytes at {offset}: {error}",
                        buf.len()
                    ))
                })
            }
        }
    }

    /// Borrow `start..end` without copying.
    ///
    /// Returns `None` for a buffered backend, which cannot hand out a borrow, so callers
    /// fall back to [`Storage::read_vec`].
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<Bytes> {
        match self {
            Self::Memory(bytes) => {
                // Check the range up front. `Bytes::split` is happy to hand back an
                // oversized second half on some backends, and a caller that asked for
                // bytes that are not there must get `None`, not a short slice.
                if end < start || end > bytes.len() {
                    return None;
                }
                // `split` twice yields the middle as its own refcounted allocation.
                let (_, right) = Bytes::clone(bytes).split(start).ok()?;
                let (middle, _) = right.split(end - start).ok()?;
                Some(middle)
            }
            #[cfg(feature = "std")]
            Self::Buffered { .. } => None,
        }
    }

    /// Borrow `len` bytes at `offset` directly out of the mapping, without even
    /// splitting a `Bytes` handle.
    ///
    /// Only the metadata pass uses this, where the borrow does not escape the reader.
    pub(crate) fn as_slice(&self) -> Option<&[u8]> {
        match self {
            Self::Memory(bytes) => Some(bytes.as_ref()),
            #[cfg(feature = "std")]
            Self::Buffered { .. } => None,
        }
    }

    /// Read `len` bytes at `offset`, preferring a borrow and falling back to a copy.
    #[cfg(feature = "std")]
    pub(crate) fn read_vec_or_slice(
        &self,
        offset: usize,
        len: usize,
    ) -> Result<Vec<u8>, GgufError> {
        match self.as_slice() {
            Some(data) => {
                let end = offset.checked_add(len).ok_or_else(|| {
                    GgufError::Io(format!("offset {offset} plus {len} bytes overflows"))
                })?;
                if end > data.len() {
                    return Err(GgufError::UnexpectedEof {
                        context: "tensor data",
                        offset,
                        needed: len,
                        available: data.len().saturating_sub(offset),
                    });
                }
                Ok(data[offset..end].to_vec())
            }
            None => self.read_vec(offset, len),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a `Bytes` that supports arbitrary-offset slicing.
    ///
    /// `from_bytes_vec` only splits at its own alignment, whereas the shared controller
    /// used by the memory-mapped path slices anywhere, which is the behaviour worth
    /// testing here.
    fn storage(data: &[u8]) -> Storage {
        let shared = bytes::Bytes::copy_from_slice(data);
        Storage::from_bytes(Bytes::from_shared(
            shared,
            burn_tensor::AllocationProperty::File,
        ))
    }

    #[test]
    fn should_report_length() {
        assert_eq!(storage(&[1, 2, 3, 4]).len(), Some(4));
    }

    #[test]
    fn should_slice_without_copying() {
        let storage = storage(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let slice = storage.slice(2, 5).expect("memory backend can slice");
        assert_eq!(&slice[..], &[2u8, 3, 4]);
    }

    #[test]
    fn should_refuse_to_slice_out_of_bounds_or_backwards() {
        let storage = storage(&[0, 1, 2, 3]);
        assert!(storage.slice(2, 10).is_none());
        assert!(storage.slice(4, 2).is_none());
    }

    #[test]
    fn should_slice_the_whole_range() {
        let storage = storage(&[9, 8, 7]);
        let slice = storage.slice(0, 3).expect("whole slice");
        assert_eq!(&slice[..], &[9u8, 8, 7]);
    }

    #[test]
    fn should_expose_the_mapping_borrow() {
        assert_eq!(storage(&[4, 5, 6]).as_slice(), Some(&[4u8, 5, 6][..]));
    }

    #[cfg(feature = "std")]
    #[test]
    fn should_copy_on_read() {
        let storage = storage(&[1, 2, 3, 4, 5]);
        assert_eq!(storage.read_vec(1, 3).unwrap(), vec![2, 3, 4]);
        assert_eq!(storage.read_vec_or_slice(1, 3).unwrap(), vec![2, 3, 4]);
        assert_eq!(storage.read_vec(0, 5).unwrap(), vec![1, 2, 3, 4, 5]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn should_report_eof_on_short_reads() {
        let storage = storage(&[1, 2, 3]);
        assert!(matches!(
            storage.read_vec(1, 10),
            Err(GgufError::UnexpectedEof { .. })
        ));
        assert!(matches!(
            storage.read_vec_or_slice(1, 10),
            Err(GgufError::UnexpectedEof { .. })
        ));
        // Checked arithmetic must not wrap near the end of the address space.
        assert!(storage.read_vec(usize::MAX, 4).is_err());
        assert!(storage.read_vec_or_slice(usize::MAX, 4).is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn should_read_through_a_buffered_backend() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, [10u8, 20, 30, 40]).unwrap();

        let storage = Storage::from_file(&path).unwrap();
        assert_eq!(storage.len(), Some(4));
        // A buffered backend cannot hand out a borrow or a zero-copy slice.
        assert!(storage.as_slice().is_none());
        assert!(storage.slice(0, 4).is_none());
        assert_eq!(storage.read_vec(1, 2).unwrap(), vec![20, 30]);
        assert!(storage.read_vec(3, 9).is_err());
    }

    #[cfg(all(feature = "std", feature = "memmap"))]
    #[test]
    fn should_map_a_file_and_read_it_without_copying() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mapped.bin");
        std::fs::write(&path, [7u8, 8, 9, 10]).unwrap();

        let storage = Storage::from_mmap(&path).unwrap();
        assert_eq!(storage.len(), Some(4));
        let slice = storage.slice(1, 3).expect("mmap can slice");
        assert_eq!(&slice[..], &[8u8, 9]);
    }

    #[cfg(all(feature = "std", feature = "memmap"))]
    #[test]
    fn should_report_a_clear_error_for_a_missing_file() {
        let error = Storage::from_mmap(std::path::Path::new("definitely/not/here.bin"));
        assert!(matches!(error, Err(GgufError::Io(_))), "{error:?}");
    }

    #[cfg(all(feature = "std", feature = "memmap"))]
    #[test]
    fn should_reject_an_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        std::fs::write(&path, []).unwrap();
        assert!(matches!(Storage::from_mmap(&path), Err(GgufError::Io(_))));
    }
}
