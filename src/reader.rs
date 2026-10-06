//! The GGUF reader: header, metadata and tensor descriptors.

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use burn_tensor::Bytes;

use crate::base::{
    DEFAULT_ALIGNMENT, GgufError, GgufHeader, GgufVersion, MAGIC, MAX_METADATA_KV,
    MAX_TENSOR_COUNT, align_offset, check_limit, read_le_u64, read_u32_le,
};
use crate::kv::{MetadataEntry, MetadataValue, read_entry};
use crate::storage::Storage;
use crate::tensor::{GgmlType, TensorInfo};

/// Metadata key naming the architecture, for example `general.architecture`.
pub const KEY_ARCHITECTURE: &str = "general.architecture";
/// Metadata key carrying the container's global alignment.
pub const KEY_ALIGNMENT: &str = "general.alignment";
/// Metadata key carrying the ggml quantization scheme version.
pub const KEY_QUANTIZATION_VERSION: &str = "general.quantization_version";
/// Metadata key naming the scheme that most tensors use.
pub const KEY_FILE_TYPE: &str = "general.file_type";

/// The `general.file_type` enumeration.
///
/// This is a separate enumeration from [`GgmlType`] and the two numberings do not line
/// up: value 14 is `MOSTLY_Q4_K_S` here but `Q6_K` as a `ggml_type` tag. Reading one as
/// the other silently reports the wrong scheme, so they are kept apart here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufFileType {
    /// Every tensor is `F32`.
    AllF32,
    /// Most tensors are `F16`.
    MostlyF16,
    /// Most tensors are `Q4_0`.
    MostlyQ4_0,
    /// Most tensors are `Q4_1`.
    MostlyQ4_1,
    /// Most tensors are `Q4_1`, with some `F16`.
    MostlyQ4_1SomeF16,
    /// Most tensors are `Q8_0`.
    MostlyQ8_0,
    /// Most tensors are `Q5_0`.
    MostlyQ5_0,
    /// Most tensors are `Q5_1`.
    MostlyQ5_1,
    /// Most tensors are `Q2_K`.
    MostlyQ2K,
    /// Most tensors are `Q3_K`, small variant.
    MostlyQ3KS,
    /// Most tensors are `Q3_K`, medium variant.
    MostlyQ3KM,
    /// Most tensors are `Q3_K`, large variant.
    MostlyQ3KL,
    /// Most tensors are `Q4_K`, small variant.
    MostlyQ4KS,
    /// Most tensors are `Q4_K`, medium variant.
    MostlyQ4KM,
    /// Most tensors are `Q5_K`, small variant.
    MostlyQ5KS,
    /// Most tensors are `Q5_K`, medium variant.
    MostlyQ5KM,
    /// Most tensors are `Q6_K`.
    MostlyQ6K,
}

impl TryFrom<u64> for GgufFileType {
    type Error = ();

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::AllF32,
            1 => Self::MostlyF16,
            2 => Self::MostlyQ4_0,
            3 => Self::MostlyQ4_1,
            4 => Self::MostlyQ4_1SomeF16,
            // 5 and 6 were MOSTLY_Q4_2 and MOSTLY_Q4_3, both withdrawn from the format.
            7 => Self::MostlyQ8_0,
            8 => Self::MostlyQ5_0,
            9 => Self::MostlyQ5_1,
            10 => Self::MostlyQ2K,
            11 => Self::MostlyQ3KS,
            12 => Self::MostlyQ3KM,
            13 => Self::MostlyQ3KL,
            14 => Self::MostlyQ4KS,
            15 => Self::MostlyQ4KM,
            16 => Self::MostlyQ5KS,
            17 => Self::MostlyQ5KM,
            18 => Self::MostlyQ6K,
            _ => return Err(()),
        })
    }
}

impl GgufFileType {
    /// The `ggml` element type this scheme's name refers to.
    ///
    /// The `_S` and `_M` variants of a `K`-quant describe which tensors were left at full
    /// precision rather than a different packing, so they collapse onto the same type.
    pub const fn tensor_type(self) -> GgmlType {
        match self {
            Self::AllF32 => GgmlType::F32,
            Self::MostlyF16 => GgmlType::F16,
            Self::MostlyQ4_0 => GgmlType::Q4_0,
            Self::MostlyQ4_1 | Self::MostlyQ4_1SomeF16 => GgmlType::Q4_1,
            Self::MostlyQ8_0 => GgmlType::Q8_0,
            Self::MostlyQ5_0 => GgmlType::Q5_0,
            Self::MostlyQ5_1 => GgmlType::Q5_1,
            Self::MostlyQ2K => GgmlType::Q2K,
            Self::MostlyQ3KS | Self::MostlyQ3KM | Self::MostlyQ3KL => GgmlType::Q3K,
            Self::MostlyQ4KS | Self::MostlyQ4KM => GgmlType::Q4K,
            Self::MostlyQ5KS | Self::MostlyQ5KM => GgmlType::Q5K,
            Self::MostlyQ6K => GgmlType::Q6K,
        }
    }

    /// The canonical name used in GGUF file names, such as `Q4_K_M`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::AllF32 => "F32",
            Self::MostlyF16 => "F16",
            Self::MostlyQ4_0 => "Q4_0",
            Self::MostlyQ4_1 => "Q4_1",
            Self::MostlyQ4_1SomeF16 => "Q4_1_SOME_F16",
            Self::MostlyQ8_0 => "Q8_0",
            Self::MostlyQ5_0 => "Q5_0",
            Self::MostlyQ5_1 => "Q5_1",
            Self::MostlyQ2K => "Q2_K",
            Self::MostlyQ3KS => "Q3_K_S",
            Self::MostlyQ3KM => "Q3_K_M",
            Self::MostlyQ3KL => "Q3_K_L",
            Self::MostlyQ4KS => "Q4_K_S",
            Self::MostlyQ4KM => "Q4_K_M",
            Self::MostlyQ5KS => "Q5_K_S",
            Self::MostlyQ5KM => "Q5_K_M",
            Self::MostlyQ6K => "Q6_K",
        }
    }
}

/// A parsed GGUF file.
///
/// Parsing is eager for everything structural — the header, every metadata entry and
/// every tensor descriptor are decoded up front — while tensor payloads stay in the
/// backing storage and are only read when a caller asks for them. That split is what
/// makes it affordable to open a 200 GB model: a few thousand descriptors cost
/// microseconds, and no weight is touched.
pub struct GgufReader {
    storage: Storage,
    version: GgufVersion,
    alignment: usize,
    metadata: Vec<MetadataEntry>,
    tensors: Vec<TensorInfo>,
    /// Absolute byte offset of the tensor data section.
    data_offset: usize,
}

impl core::fmt::Debug for GgufReader {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GgufReader")
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("metadata_entries", &self.metadata.len())
            .field("tensors", &self.tensors.len())
            .field("data_offset", &self.data_offset)
            .finish()
    }
}

impl GgufReader {
    /// Parse a GGUF file already held in memory.
    ///
    /// The buffer is copied once into a refcounted, freely sliceable allocation, because
    /// [`Bytes::from_bytes_vec`] only splits at its own alignment and a quantized tensor
    /// may start anywhere. The copy is what makes every later tensor read zero-copy.
    pub fn from_bytes(bytes: Bytes) -> Result<Self, GgufError> {
        let shared = bytes::Bytes::copy_from_slice(bytes.as_ref());
        Self::parse(Storage::from_bytes(Bytes::from_shared(
            shared,
            burn_tensor::AllocationProperty::Native,
        )))
    }

    /// Parse a `&'static [u8]`, for embedded models.
    ///
    /// The payload is copied once, so this is for small embedded models rather than large
    /// ones; use [`GgufReader::open`] for anything sizable.
    pub fn from_static(data: &'static [u8]) -> Result<Self, GgufError> {
        Self::from_bytes(Bytes::from_bytes_vec(data.to_vec()))
    }

    /// Memory-map and parse a GGUF file.
    ///
    /// Tensor payloads are then reachable as zero-copy subranges of the mapping.
    #[cfg(all(feature = "std", feature = "memmap"))]
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, GgufError> {
        Self::parse(Storage::from_mmap(path.as_ref())?)
    }

    /// Parse a GGUF file, memory-mapping it when available.
    #[cfg(feature = "std")]
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, GgufError> {
        #[cfg(feature = "memmap")]
        {
            Self::from_file(path)
        }
        #[cfg(not(feature = "memmap"))]
        {
            Self::parse(Storage::from_file(path.as_ref())?)
        }
    }

    /// Parse a GGUF file held in a `Vec<u8>`.
    #[cfg(feature = "std")]
    pub fn from_vec(data: Vec<u8>) -> Result<Self, GgufError> {
        Self::from_bytes(Bytes::from_bytes_vec(data))
    }

    /// Read the whole structure out of `storage`.
    fn parse(storage: Storage) -> Result<Self, GgufError> {
        // A buffered backend cannot lend out a borrow, so the header pass pulls the file
        // in. With `memmap` (the default) this is a refcount, not a copy.
        #[cfg(feature = "std")]
        let bytes: Vec<u8> = match storage.as_slice() {
            Some(_) => Vec::new(),
            None => {
                let len = storage
                    .len()
                    .ok_or_else(|| GgufError::Io("cannot determine the file length".into()))?;
                storage.read_vec(0, len)?
            }
        };
        #[cfg(feature = "std")]
        let prefix: &[u8] = match storage.as_slice() {
            Some(data) => data,
            None => &bytes,
        };
        #[cfg(not(feature = "std"))]
        let prefix: &[u8] = storage
            .as_slice()
            .ok_or_else(|| GgufError::Io("GGUF reading requires a resident buffer".into()))?;

        let header = read_header(prefix)?;

        let mut offset = GgufHeader::size(header.version);
        let mut metadata = Vec::with_capacity(
            check_limit(
                "metadata_kv_count",
                header.metadata_kv_count,
                MAX_METADATA_KV,
            )?
            .min(1024),
        );
        for _ in 0..header.metadata_kv_count {
            let (entry, next) = read_entry(prefix, offset)?;
            metadata.push(entry);
            offset = next;
        }

        let alignment = read_alignment(&metadata)?;

        let tensor_count = check_limit("tensor_count", header.tensor_count, MAX_TENSOR_COUNT)?;
        let mut tensors = Vec::with_capacity(tensor_count.min(1024));
        for _ in 0..tensor_count {
            let (info, next) = read_tensor_info(prefix, offset)?;
            tensors.push(info);
            offset = next;
        }

        // The tensor data section starts at the next aligned offset after the last
        // descriptor. Tensor offsets in the file are relative to this point.
        let data_offset = align_offset(offset, alignment)?;

        // A resident buffer always knows its own length, so this check is unconditional. A
        // buffered backend can only answer with `std`, and without it the check is skipped
        // rather than skipped *silently* by guessing a size.
        #[cfg(feature = "std")]
        let total = storage.len().unwrap_or(usize::MAX);
        #[cfg(not(feature = "std"))]
        let total = usize::MAX;
        for info in &tensors {
            validate_tensor(info, data_offset, total)?;
        }
        validate_tensor_alignment(&tensors, alignment)?;

        Ok(Self {
            storage,
            version: header.version,
            alignment,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// The container version.
    pub const fn version(&self) -> GgufVersion {
        self.version
    }

    /// The global alignment, from `general.alignment` or the default of 32.
    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    /// Absolute byte offset of the tensor data section.
    pub const fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Every metadata entry, in file order.
    pub fn metadata(&self) -> &[MetadataEntry] {
        &self.metadata
    }

    /// Look up a metadata value by key.
    pub fn metadata_value(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.value)
    }

    /// Look up a metadata value as an integer, accepting any integer width.
    pub fn metadata_usize(&self, key: &str) -> Option<usize> {
        self.metadata_value(key).and_then(MetadataValue::as_usize)
    }

    /// Look up a metadata value as a float, accepting any numeric width.
    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata_value(key).and_then(MetadataValue::as_f32)
    }

    /// The architecture name, for example `llama`.
    ///
    /// This is the key that selects which module implementation maps the tensor names,
    /// which is why the parser itself stays architecture-agnostic.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata_value(KEY_ARCHITECTURE)
            .and_then(MetadataValue::as_str)
    }

    /// The ggml quantization scheme version, required by the specification when any
    /// tensor is quantized.
    pub fn quantization_version(&self) -> Option<u64> {
        self.metadata_value(KEY_QUANTIZATION_VERSION)
            .and_then(MetadataValue::as_u64)
    }

    /// Every tensor descriptor, in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Look up a tensor descriptor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|info| info.name == name)
    }

    /// The dominant element type across all tensors.
    ///
    /// A model is described by the `general.file_type` metadata, but not every writer sets
    /// it, so this falls back to the most common tensor type. Counts are per tensor rather
    /// than per byte, which matches how `file_type` itself is derived.
    pub fn dominant_type(&self) -> Option<GgmlType> {
        if self.tensors.is_empty() {
            return None;
        }
        if let Some(value) = self.metadata_value("general.file_type")
            && let Some(file_type) = GgufFileType::try_from(value.as_u64()?).ok()
        {
            return Some(file_type.tensor_type());
        }

        // Walk `ALL` once to count, then again to find the winner, so the tally is keyed by
        // type rather than by array position.
        let mut best = self.tensors[0].tensor_type;
        let mut best_count = 0usize;
        for candidate in GgmlType::ALL {
            let count = self
                .tensors
                .iter()
                .filter(|info| info.tensor_type == *candidate)
                .count();
            if count > best_count {
                best_count = count;
                best = *candidate;
            }
        }
        Some(best)
    }

    /// The raw bytes of a tensor's data, copying them.
    ///
    /// This is the fallback for backends that cannot be handed a reference.
    #[cfg(feature = "std")]
    pub fn tensor_data_vec(&self, info: &TensorInfo) -> Result<Vec<u8>, GgufError> {
        let nbytes = info.nbytes()?;
        self.read_tensor(info, nbytes)
    }

    /// Copy `nbytes` bytes of a tensor's data, bounds checked.
    #[cfg(feature = "std")]
    pub fn read_tensor(&self, info: &TensorInfo, nbytes: usize) -> Result<Vec<u8>, GgufError> {
        let start = self.tensor_start(info);
        self.storage.read_vec_or_slice(start, nbytes)
    }

    /// A zero-copy view of a tensor's data.
    ///
    /// Returns `None` when the backend is buffered, or when the requested range falls
    /// outside the mapping.
    pub fn tensor_data(&self, info: &TensorInfo) -> Result<Option<Bytes>, GgufError> {
        let nbytes = info.nbytes()?;
        let start = self.tensor_start(info);
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| GgufError::TensorSizeOverflow {
                name: info.name.clone(),
            })?;
        Ok(self.storage.slice(start, end))
    }

    /// The raw bytes of a tensor's data.
    ///
    /// Zero-copy when the file is mapped, otherwise a copy. This works without `std` as
    /// long as the data is resident, which is what makes it usable from the store's
    /// lazy closures regardless of how the crate was built.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<Bytes, GgufError> {
        let nbytes = info.nbytes()?;
        let start = self.tensor_start(info);
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| GgufError::TensorSizeOverflow {
                name: info.name.clone(),
            })?;
        if let Some(slice) = self.storage.slice(start, end) {
            return Ok(slice);
        }
        #[cfg(feature = "std")]
        {
            Ok(Bytes::from_bytes_vec(self.storage.read_vec(start, nbytes)?))
        }
        #[cfg(not(feature = "std"))]
        {
            Err(GgufError::Io(format!(
                "tensor '{}' needs to be read from a buffered file, which requires std",
                info.name
            )))
        }
    }

    /// The raw bytes of a tensor's data.
    ///
    /// Zero-copy when the file is mapped, otherwise a copy.
    #[cfg(feature = "std")]
    pub fn tensor_data_bytes(&self, info: &TensorInfo) -> Result<Bytes, GgufError> {
        self.tensor_bytes(info)
    }

    /// The absolute file offset of a tensor's data.
    pub fn tensor_start(&self, info: &TensorInfo) -> usize {
        self.data_offset.saturating_add(info.offset)
    }

    /// Whether a tensor's data can be handed out without copying.
    pub fn is_zero_copy(&self) -> bool {
        self.storage.as_slice().is_some()
    }
}

/// Read and validate the fixed header.
fn read_header(bytes: &[u8]) -> Result<GgufHeader, GgufError> {
    if bytes.len() < 4 {
        return Err(GgufError::UnexpectedEof {
            context: "GGUF magic",
            offset: 0,
            needed: 4,
            available: bytes.len(),
        });
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != MAGIC {
        return Err(GgufError::BadMagic {
            found: [bytes[0], bytes[1], bytes[2], bytes[3]],
        });
    }

    let version = GgufVersion::try_from(read_u32_le(bytes, 4, "GGUF version")?)?;

    let (tensor_count, metadata_kv_count) = if version.counts_are_u64() {
        (
            read_le_u64(bytes.get(8..16).ok_or_else(|| GgufError::UnexpectedEof {
                context: "GGUF tensor_count",
                offset: 8,
                needed: 8,
                available: bytes.len().saturating_sub(8),
            })?),
            read_le_u64(bytes.get(16..24).ok_or_else(|| GgufError::UnexpectedEof {
                context: "GGUF metadata_kv_count",
                offset: 16,
                needed: 8,
                available: bytes.len().saturating_sub(16),
            })?),
        )
    } else {
        (
            read_u32_le(bytes, 8, "GGUF tensor_count")? as u64,
            read_u32_le(bytes, 12, "GGUF metadata_kv_count")? as u64,
        )
    };

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_kv_count,
    })
}

/// Determine the global alignment from `general.alignment`.
fn read_alignment(metadata: &[MetadataEntry]) -> Result<usize, GgufError> {
    let Some(entry) = metadata.iter().find(|entry| entry.key == KEY_ALIGNMENT) else {
        return Ok(DEFAULT_ALIGNMENT);
    };

    let raw = entry.value.as_usize().ok_or_else(|| {
        GgufError::InvalidFormat(format!(
            "{KEY_ALIGNMENT} must be a non-negative integer, found {}",
            entry.value.display_summary()
        ))
    })?;

    if raw == 0 || !raw.is_power_of_two() || raw < 8 {
        return Err(GgufError::Config(format!(
            "{KEY_ALIGNMENT} must be a power of two of at least 8, found {raw}"
        )));
    }
    Ok(raw)
}

/// Read one tensor descriptor.
fn read_tensor_info(bytes: &[u8], offset: usize) -> Result<(TensorInfo, usize), GgufError> {
    let mut cursor = offset;

    // The name is a length-prefixed string.
    let name_len = read_le_u64(slice(bytes, cursor, 8, "tensor name length")?);
    cursor += 8;
    let name_len =
        crate::base::check_limit("tensor name length", name_len, crate::base::MAX_STRING_LEN)?;
    let name_bytes = slice(bytes, cursor, name_len, "tensor name")?;
    let name = core::str::from_utf8(name_bytes)
        .map_err(|_| GgufError::InvalidUtf8 {
            context: "tensor name",
        })?
        .to_string();
    cursor += name_len;

    let n_dims = read_u32_le(bytes, cursor, "tensor dimension count")?;
    cursor += 4;
    let n_dims = crate::base::check_limit(
        "tensor dimension count",
        n_dims as u64,
        crate::base::MAX_TENSOR_DIMS,
    )?;
    if n_dims == 0 {
        return Err(GgufError::InvalidFormat(format!(
            "tensor '{name}' declares zero dimensions"
        )));
    }

    let mut dims = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        let dim = read_le_u64(slice(bytes, cursor, 8, "tensor dimension")?);
        cursor += 8;
        // A zero dimension would make the block count meaningless, and the shape itself
        // is never useful, so reject it here rather than deeper in.
        dims.push(
            usize::try_from(dim)
                .map_err(|_| GgufError::TensorSizeOverflow { name: name.clone() })?,
        );
    }

    let tensor_type = GgmlType::try_from(read_u32_le(bytes, cursor, "tensor type")?)?;
    cursor += 4;

    let offset_in_data = read_le_u64(slice(bytes, cursor, 8, "tensor data offset")?);
    cursor += 8;
    let offset_in_data = usize::try_from(offset_in_data)
        .map_err(|_| GgufError::TensorSizeOverflow { name: name.clone() })?;

    let info = TensorInfo::new(name, dims, tensor_type, offset_in_data)?;
    info.validate_block_layout()?;
    Ok((info, cursor))
}

/// Borrow `len` bytes at `offset`, reporting a descriptive EOF.
fn slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    len: usize,
    context: &'static str,
) -> Result<&'a [u8], GgufError> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| GgufError::InvalidFormat(format!("{context} overflows the file")))?;
    if end > bytes.len() {
        return Err(GgufError::UnexpectedEof {
            context,
            offset,
            needed: len,
            available: bytes.len().saturating_sub(offset),
        });
    }
    Ok(&bytes[offset..end])
}

/// Check that a tensor's data lies inside the file.
///
/// `start` and `len` are absolute, so they are compared against `total`, the size of the
/// whole file. Alignment is checked separately by [`validate_tensor_alignment`], since it
/// needs the file's own alignment rather than a default.
fn validate_tensor(info: &TensorInfo, data_offset: usize, total: usize) -> Result<(), GgufError> {
    let nbytes = info.nbytes()?;

    let start = data_offset.saturating_add(info.offset);
    let end = start.saturating_add(nbytes);
    if end > total {
        return Err(GgufError::TensorOutOfBounds {
            name: info.name.clone(),
            start,
            len: nbytes,
            section_len: total.saturating_sub(data_offset),
        });
    }
    Ok(())
}

/// Check that every tensor offset respects the file's alignment.
pub(crate) fn validate_tensor_alignment(
    tensors: &[TensorInfo],
    alignment: usize,
) -> Result<(), GgufError> {
    for info in tensors {
        if info.offset % alignment != 0 {
            return Err(GgufError::InvalidFormat(format!(
                "tensor '{}' has data offset {} which is not a multiple of the file \
                 alignment of {alignment}",
                info.name, info.offset
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{GgufWriter, TensorSpec};
    use alloc::vec;

    /// A two-tensor model with a couple of metadata entries.
    fn sample() -> GgufWriter {
        let mut writer = GgufWriter::new();
        writer.metadata("general.architecture", "llama");
        writer.metadata("general.alignment", MetadataValue::U32(32));
        writer.metadata("llama.block_count", MetadataValue::U32(2));
        writer.tensor(
            TensorSpec::new("blk.0.attn_q.weight", vec![4, 2], GgmlType::F32).with_f32(&[0.0; 8]),
        );
        writer.tensor(
            TensorSpec::new("output_norm.weight", vec![4], GgmlType::F32)
                .with_f32(&[1.0, 2.0, 3.0, 4.0]),
        );
        writer
    }

    fn parse(bytes: &[u8]) -> Result<GgufReader, GgufError> {
        GgufReader::from_bytes(Bytes::from_bytes_vec(bytes.to_vec()))
    }

    fn parse_ok(bytes: Vec<u8>) -> GgufReader {
        parse(&bytes).expect("the writer must produce a readable file")
    }

    #[test]
    fn should_parse_header_metadata_and_tensors() {
        let reader = parse_ok(sample().build());

        assert_eq!(reader.version(), GgufVersion::V3);
        assert_eq!(reader.alignment(), 32);
        assert_eq!(reader.architecture(), Some("llama"));
        assert_eq!(reader.metadata_usize("llama.block_count"), Some(2));
        assert_eq!(reader.tensors().len(), 2);

        let info = reader.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(info.dims, vec![4, 2]);
        assert_eq!(info.tensor_type, GgmlType::F32);
        // Reversed into Burn's row-major order.
        assert_eq!(info.burn_shape().as_slice(), &[2, 4]);
        assert_eq!(info.nbytes().unwrap(), 32);
        assert!(reader.tensor("does.not.exist").is_none());
    }

    #[test]
    fn should_reject_a_bad_magic() {
        let mut bytes = sample().build();
        bytes[0] = b'X';
        assert!(matches!(parse(&bytes), Err(GgufError::BadMagic { .. })));
    }

    #[test]
    fn should_reject_a_future_version() {
        let mut bytes = sample().build();
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(GgufError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn should_reject_a_truncated_header() {
        let bytes = sample().build();
        assert!(matches!(
            parse(&bytes[..10]),
            Err(GgufError::UnexpectedEof { .. })
        ));
        assert!(matches!(
            parse(b"".as_slice()),
            Err(GgufError::UnexpectedEof { .. })
        ));
        assert!(matches!(
            parse(b"GGU".as_slice()),
            Err(GgufError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn should_default_alignment_when_absent() {
        let mut writer = GgufWriter::new();
        writer.tensor(TensorSpec::new("w", vec![2], GgmlType::F32).with_f32(&[1.0, 2.0]));
        assert_eq!(parse_ok(writer.build()).alignment(), DEFAULT_ALIGNMENT);
    }

    #[test]
    fn should_reject_a_nonsensical_alignment() {
        // The writer will not emit an invalid alignment, so this exercises the reader's
        // validation directly.
        for bad in [0usize, 4, 3, 24] {
            let error = read_alignment(&[MetadataEntry::new(
                KEY_ALIGNMENT,
                MetadataValue::U32(bad as u32),
            )])
            .unwrap_err();
            assert!(matches!(error, GgufError::Config(_)), "{error:?}");
        }
    }

    #[test]
    fn should_reject_a_non_numeric_alignment() {
        let error = read_alignment(&[MetadataEntry::new(
            KEY_ALIGNMENT,
            MetadataValue::String("nope".into()),
        )])
        .unwrap_err();
        assert!(matches!(error, GgufError::InvalidFormat(_)), "{error:?}");
    }

    #[test]
    fn should_default_alignment_when_absent_or_valid() {
        assert_eq!(read_alignment(&[]).unwrap(), DEFAULT_ALIGNMENT);
        assert_eq!(
            read_alignment(&[MetadataEntry::new(KEY_ALIGNMENT, MetadataValue::U32(64))]).unwrap(),
            64
        );
    }

    #[test]
    fn should_honour_a_larger_alignment() {
        let mut writer = GgufWriter::new();
        // The writer sanitises a stray `general.alignment` entry, so the alignment has to
        // be set on the writer itself.
        writer.metadata("general.alignment", MetadataValue::U32(64));
        writer.alignment(64);
        writer.tensor(TensorSpec::new("w", vec![4], GgmlType::F32).with_f32(&[1.0, 2.0, 3.0, 4.0]));
        let reader = parse_ok(writer.build());
        assert_eq!(reader.alignment(), 64);
        assert_eq!(reader.data_offset() % 64, 0);
    }

    #[test]
    fn should_read_tensor_payloads() {
        let reader = parse_ok(sample().build());
        let info = reader.tensor("output_norm.weight").unwrap();

        let expected: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(&reader.tensor_data_bytes(info).unwrap()[..], expected);
        assert_eq!(reader.tensor_data_vec(info).unwrap(), expected);
    }

    #[test]
    fn should_expose_a_zero_copy_view_when_resident() {
        let reader = parse_ok(sample().build());
        assert!(reader.is_zero_copy());

        let info = reader.tensor("output_norm.weight").unwrap();
        let view = reader
            .tensor_data(info)
            .unwrap()
            .expect("resident data slices");
        assert_eq!(view.len(), 16);

        // The zero-copy view and the copy must agree.
        let copy = reader.tensor_data_vec(info).unwrap();
        assert_eq!(&view[..], copy.as_slice());
    }

    #[test]
    fn should_reject_tensor_data_past_the_end_of_the_file() {
        let mut writer = GgufWriter::new();
        // The descriptor claims 64 elements but only two are written.
        writer.tensor(TensorSpec::new("w", vec![64], GgmlType::F32).with_f32(&[1.0, 2.0]));
        assert!(matches!(
            parse(&writer.build()),
            Err(GgufError::TensorOutOfBounds { .. })
        ));
    }

    #[test]
    fn should_reject_a_shape_that_breaks_the_block_layout() {
        let mut writer = GgufWriter::new();
        // 30 is not a multiple of the 32 values in a Q8_0 block.
        writer.tensor(TensorSpec::new("w", vec![30], GgmlType::Q8_0));
        assert!(matches!(
            parse(&writer.build()),
            Err(GgufError::InvalidFormat(_))
        ));
    }

    #[test]
    fn should_read_v1_headers_with_narrow_counts() {
        let reader = parse_ok(sample().build_v1());
        assert_eq!(reader.version(), GgufVersion::V1);
        assert_eq!(reader.tensors().len(), 2);
        assert_eq!(reader.architecture(), Some("llama"));
    }

    #[test]
    fn should_read_v2_headers() {
        let reader = parse_ok(sample().build_with(GgufVersion::V2));
        assert_eq!(reader.version(), GgufVersion::V2);
        assert_eq!(reader.tensors().len(), 2);
    }

    #[test]
    fn should_report_the_dominant_type() {
        let mut writer = GgufWriter::new();
        writer.tensor(TensorSpec::new("a", vec![2], GgmlType::F32).with_f32(&[1.0, 2.0]));
        writer.tensor(TensorSpec::new("b", vec![2], GgmlType::F32).with_f32(&[1.0, 2.0]));
        writer.tensor(TensorSpec::new("c", vec![32], GgmlType::Q8_0));
        assert_eq!(
            parse_ok(writer.build()).dominant_type(),
            Some(GgmlType::F32)
        );
    }

    #[test]
    fn should_prefer_file_type_metadata_for_the_dominant_type() {
        let mut writer = GgufWriter::new();
        writer.metadata("general.file_type", MetadataValue::U32(14));
        writer.tensor(TensorSpec::new("a", vec![2], GgmlType::F32).with_f32(&[1.0, 2.0]));
        // 14 is MOSTLY_Q4_K_S in the file_type enum.
        assert_eq!(
            parse_ok(writer.build()).dominant_type(),
            Some(GgmlType::Q4K)
        );
    }

    #[test]
    fn should_have_no_dominant_type_when_there_are_no_tensors() {
        let mut writer = GgufWriter::new();
        writer.metadata("general.architecture", "llama");
        assert_eq!(parse_ok(writer.build()).dominant_type(), None);
    }

    #[test]
    fn should_enforce_alignment_on_tensor_offsets() {
        let info = TensorInfo::new("w".into(), vec![4], GgmlType::F32, 0).unwrap();
        assert!(validate_tensor_alignment(&[info], 32).is_ok());

        let misaligned = TensorInfo::new("w".into(), vec![4], GgmlType::F32, 4).unwrap();
        assert!(validate_tensor_alignment(std::slice::from_ref(&misaligned), 32).is_err());
        // The same offset is fine when the file only asks for 4-byte alignment.
        assert!(validate_tensor_alignment(&[misaligned], 4).is_ok());
    }

    #[test]
    fn should_reject_a_declared_count_beyond_the_limit() {
        let mut writer = GgufWriter::new();
        writer.tensor(TensorSpec::new("w", vec![2], GgmlType::F32).with_f32(&[1.0, 2.0]));
        let mut bytes = writer.build();
        // Claim an enormous tensor count in the header.
        bytes[8..16].copy_from_slice(&(MAX_TENSOR_COUNT as u64 + 1).to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(GgufError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn should_reject_an_absurd_metadata_count() {
        let mut writer = GgufWriter::new();
        writer.tensor(TensorSpec::new("w", vec![2], GgmlType::F32).with_f32(&[1.0, 2.0]));
        let mut bytes = writer.build();
        bytes[16..24].copy_from_slice(&(MAX_METADATA_KV as u64 + 1).to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(GgufError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn should_read_a_quantized_tensor_end_to_end() {
        // A Q8_0 tensor, which is the format that stays packed in Burn.
        let mut writer = GgufWriter::new();
        writer.metadata("general.architecture", "llama");
        writer.metadata("general.quantization_version", MetadataValue::U32(2));
        let block: Vec<u8> = (0..2)
            .flat_map(|index: u32| {
                let mut block = Vec::new();
                block.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
                block.extend(core::iter::repeat_n(index as u8, 32));
                block
            })
            .collect();
        writer.tensor(
            TensorSpec::new("blk.0.ffn_up.weight", vec![32, 2], GgmlType::Q8_0).with_bytes(&block),
        );

        let reader = parse_ok(writer.build());
        assert_eq!(reader.quantization_version(), Some(2));
        assert_eq!(reader.dominant_type(), Some(GgmlType::Q8_0));

        let info = reader.tensor("blk.0.ffn_up.weight").unwrap();
        assert_eq!(info.nbytes().unwrap(), 68);

        // The payload must come back byte for byte.
        let data = reader.tensor_data_bytes(info).unwrap();
        assert_eq!(&data[..], block.as_slice());
    }

    #[test]
    fn should_map_a_file_through_the_filesystem() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, sample().build()).unwrap();

        let reader = GgufReader::open(&path).unwrap();
        assert_eq!(reader.architecture(), Some("llama"));
        assert_eq!(reader.tensors().len(), 2);

        // With the default features the file is mapped, so payloads slice in place.
        #[cfg(feature = "memmap")]
        {
            assert!(reader.is_zero_copy());
            let info = reader.tensor("output_norm.weight").unwrap();
            let view = reader.tensor_data(info).unwrap().expect("mmap slices");
            assert_eq!(view.len(), 16);
        }
    }

    #[test]
    fn should_report_a_missing_file_clearly() {
        let error = GgufReader::open(std::path::Path::new("no/such/model.gguf")).unwrap_err();
        assert!(matches!(error, GgufError::Io(_)), "{error:?}");
        assert!(format!("{error}").contains("failed to open"), "{error}");
    }

    #[test]
    fn should_read_from_a_static_slice() {
        // The data is copied into a `Bytes` here because the caller owns a `Vec`; the
        // signature exists so callers with genuinely static data can opt in.
        let bytes = sample().build();
        let reader = GgufReader::from_vec(bytes).unwrap();
        assert_eq!(reader.tensors().len(), 2);
    }
}
