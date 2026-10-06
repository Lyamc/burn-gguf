//! GGUF container primitives: magic, version, limits and error handling.

use alloc::format;
use alloc::string::String;
use core::fmt;

/// The four leading bytes of every GGUF file, as a little-endian `u32`.
///
/// `b"GGUF"` read little-endian is `0x46554747`. Comparing the integer value lets the
/// endianness of the host cancel out, which is how `ggml` itself detects the magic.
pub const MAGIC: u32 = 0x4655_4747;

/// Default value for `general.alignment` when the metadata does not specify one.
pub const DEFAULT_ALIGNMENT: usize = 32;

/// Upper bound on the number of metadata key-value pairs.
///
/// Real models carry a few hundred entries at most (mostly the tokenizer vocabulary).
/// The cap exists so a corrupt or hostile `metadata_kv_count` cannot make us pre-allocate
/// an unbounded table.
pub const MAX_METADATA_KV: usize = 1 << 22;

/// Upper bound on the number of tensors in a file.
///
/// The largest models in circulation have on the order of a thousand tensors.
pub const MAX_TENSOR_COUNT: usize = 1 << 22;

/// Upper bound on the length in bytes of a single GGUF string (keys, names, strings).
pub const MAX_STRING_LEN: usize = 1 << 30;

/// Upper bound on the number of dimensions of a single tensor.
pub const MAX_TENSOR_DIMS: usize = 8;

/// Upper bound on how deeply metadata arrays may nest.
///
/// Arrays of arrays are legal in GGUF but only shallow nesting occurs in practice.
pub const MAX_ARRAY_DEPTH: usize = 8;

/// Upper bound on the total number of elements across one metadata value and its
/// nested arrays, so that a bogus array length cannot drive an unbounded allocation.
pub const MAX_ARRAY_ELEMENTS: usize = 1 << 26;

/// The version of the GGUF container format implemented by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GgufVersion {
    /// Version 1: `tensor_count` and `metadata_kv_count` are `u32`.
    V1,
    /// Version 2: counts widened to `u64`.
    V2,
    /// Version 3: as v2, and big-endian files become representable.
    ///
    /// The layout is identical to v2; v3 only adds a big-endian convention, and the
    /// container still gives us no way to detect which endianness a file uses, so this
    /// reader assumes little-endian as the specification instructs.
    V3,
}

impl GgufVersion {
    /// The version number as it appears in the header.
    pub const fn as_u32(self) -> u32 {
        match self {
            GgufVersion::V1 => 1,
            GgufVersion::V2 => 2,
            GgufVersion::V3 => 3,
        }
    }

    /// Whether the header's count fields are 64-bit.
    ///
    /// Widening the counts is the only structural difference between v1 and v2; v3 keeps
    /// the v2 layout.
    pub const fn counts_are_u64(self) -> bool {
        !matches!(self, GgufVersion::V1)
    }

    /// Whether the container might describe a big-endian model.
    ///
    /// Only v3 and later permit it. This reader cannot tell the two apart, so it is
    /// reported for diagnostics only and never used to decide how to decode a value.
    pub const fn may_be_big_endian(self) -> bool {
        matches!(self, GgufVersion::V3)
    }
}

impl TryFrom<u32> for GgufVersion {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(GgufVersion::V1),
            2 => Ok(GgufVersion::V2),
            3 => Ok(GgufVersion::V3),
            other => Err(GgufError::UnsupportedVersion(other)),
        }
    }
}

/// Errors produced while reading a GGUF container.
#[derive(Debug)]
pub enum GgufError {
    /// The leading bytes are not `GGUF`.
    BadMagic {
        /// The four bytes that were found.
        found: [u8; 4],
    },
    /// The container version is newer than this reader understands.
    UnsupportedVersion(u32),
    /// A read ran past the end of the mapped data.
    ///
    /// Always the sign of a truncated or malformed file, since every read is bounds
    /// checked against the known length of the mapping.
    UnexpectedEof {
        /// What was being read, for the message.
        context: &'static str,
        /// Byte offset the read started at.
        offset: usize,
        /// Number of bytes the read wanted.
        needed: usize,
        /// Number of bytes actually available.
        available: usize,
    },
    /// The header or a descriptor is structurally invalid.
    InvalidFormat(String),
    /// A declared count exceeds the corresponding safety limit.
    LimitExceeded {
        /// What was being counted.
        what: &'static str,
        /// The value found in the file.
        found: u64,
        /// The limit it exceeded.
        limit: usize,
    },
    /// A string was not valid UTF-8.
    InvalidUtf8 {
        /// What the string was being read for.
        context: &'static str,
    },
    /// A metadata key or value used a type tag outside the defined range.
    UnknownMetadataValueType(u32),
    /// A tensor declared a `ggml_type` outside the defined range.
    UnknownTensorType(u32),
    /// A tensor's byte size computed from its shape and type overflowed.
    TensorSizeOverflow {
        /// The tensor's name.
        name: String,
    },
    /// A tensor's data range does not lie inside the tensor data section.
    TensorOutOfBounds {
        /// The tensor's name.
        name: String,
        /// The absolute byte offset the tensor data starts at.
        start: usize,
        /// The number of bytes the tensor claims to occupy.
        len: usize,
        /// The size of the tensor data section.
        section_len: usize,
    },
    /// The file could not be opened or mapped.
    Io(String),
    /// A caller-supplied configuration was inconsistent.
    Config(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic { found } => write!(
                f,
                "not a GGUF file: expected magic 0x{MAGIC:08X} ('GGUF'), found 0x{:02X}{:02X}{:02X}{:02X} ({:?})",
                found[3],
                found[2],
                found[1],
                found[0],
                String::from_utf8_lossy(found)
            ),
            Self::UnsupportedVersion(version) => write!(
                f,
                "unsupported GGUF version {version}, this reader implements 1, 2 and 3"
            ),
            Self::UnexpectedEof {
                context,
                offset,
                needed,
                available,
            } => write!(
                f,
                "unexpected end of data while reading {context} at offset {offset}: \
                 needed {needed} bytes but only {available} are available"
            ),
            Self::InvalidFormat(message) => write!(f, "invalid GGUF format: {message}"),
            Self::LimitExceeded { what, found, limit } => {
                write!(f, "{what} of {found} exceeds the limit of {limit}")
            }
            Self::InvalidUtf8 { context } => write!(f, "{context} was not valid UTF-8"),
            Self::UnknownMetadataValueType(value) => {
                write!(f, "unknown GGUF metadata value type {value}")
            }
            Self::UnknownTensorType(value) => write!(f, "unknown ggml type {value}"),
            Self::TensorSizeOverflow { name } => {
                write!(f, "tensor '{name}' byte size overflowed usize")
            }
            Self::TensorOutOfBounds {
                name,
                start,
                len,
                section_len,
            } => write!(
                f,
                "tensor '{name}' spans bytes {start}..{} of a {section_len} byte tensor data section",
                start.saturating_add(*len)
            ),
            Self::Io(message) => write!(f, "GGUF I/O error: {message}"),
            Self::Config(message) => write!(f, "invalid GGUF store configuration: {message}"),
        }
    }
}

impl core::error::Error for GgufError {}

/// Round `offset` up to the next multiple of `alignment`.
///
/// This is GGUF's `align_offset`. Alignment must be a power of two and is required by the
/// specification to be a multiple of 8.
pub fn align_offset(offset: usize, alignment: usize) -> Result<usize, GgufError> {
    if !alignment.is_power_of_two() || alignment == 0 {
        return Err(GgufError::Config(format!(
            "general.alignment must be a non-zero power of two, got {alignment}"
        )));
    }
    if alignment < 8 {
        return Err(GgufError::Config(format!(
            "general.alignment must be at least 8, got {alignment}"
        )));
    }
    // `alignment - 1` masks off the low bits, and the `+ alignment - 1` before the mask
    // rounds up. This cannot overflow for any offset we can hold.
    Ok((offset + alignment - 1) & !(alignment - 1))
}

/// The fixed part of a GGUF header.
///
/// Field widths depend on the version: see [`GgufVersion::counts_are_u64`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufHeader {
    /// Container version.
    pub version: GgufVersion,
    /// Number of tensor descriptors that follow the metadata.
    pub tensor_count: u64,
    /// Number of metadata key-value pairs.
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    /// Byte length of the header for a given version.
    ///
    /// v1 is 16 bytes (`u32` counts), v2 and v3 are 24 bytes (`u64` counts).
    pub const fn size(version: GgufVersion) -> usize {
        if version.counts_are_u64() {
            4 + 4 + 8 + 8
        } else {
            4 + 4 + 4 + 4
        }
    }
}

/// Validate a declared count against its safety limit, reporting the limit on failure.
pub(crate) fn check_limit(
    what: &'static str,
    found: u64,
    limit: usize,
) -> Result<usize, GgufError> {
    if found > limit as u64 {
        return Err(GgufError::LimitExceeded { what, found, limit });
    }
    Ok(found as usize)
}

/// Read a fixed-width little-endian unsigned integer from a byte slice.
pub(crate) fn read_le_u64(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(buf)
}

/// Decode `len` bytes at `offset` as UTF-8, with bounds checking.
pub(crate) fn read_string<'a>(
    bytes: &'a [u8],
    offset: usize,
    len: u64,
    context: &'static str,
) -> Result<&'a str, GgufError> {
    let len = check_limit("string length", len, MAX_STRING_LEN)?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| GgufError::InvalidFormat(format!("{context} length overflows the file")))?;
    if end > bytes.len() {
        return Err(GgufError::UnexpectedEof {
            context,
            offset,
            needed: len,
            available: bytes.len().saturating_sub(offset),
        });
    }
    core::str::from_utf8(&bytes[offset..end]).map_err(|_| GgufError::InvalidUtf8 { context })
}

/// Read `bytes[..4]` as a little-endian `u32`.
pub(crate) fn read_u32_le(
    bytes: &[u8],
    offset: usize,
    context: &'static str,
) -> Result<u32, GgufError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| GgufError::InvalidFormat(format!("{context} offset overflows the file")))?;
    if end > bytes.len() {
        return Err(GgufError::UnexpectedEof {
            context,
            offset,
            needed: 4,
            available: bytes.len().saturating_sub(offset),
        });
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[offset..end]);
    Ok(u32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_constant_is_gguf_bytes() {
        // `b"GGUF"` interpreted little-endian.
        assert_eq!(MAGIC, u32::from_le_bytes(*b"GGUF"));
    }

    #[test]
    fn should_align_offset_to_next_multiple() {
        assert_eq!(align_offset(0, 32).unwrap(), 0);
        assert_eq!(align_offset(1, 32).unwrap(), 32);
        assert_eq!(align_offset(31, 32).unwrap(), 32);
        assert_eq!(align_offset(32, 32).unwrap(), 32);
        assert_eq!(align_offset(33, 32).unwrap(), 64);
    }

    #[test]
    fn should_align_offset_with_smaller_alignment() {
        assert_eq!(align_offset(1, 8).unwrap(), 8);
        assert_eq!(align_offset(7, 8).unwrap(), 8);
        assert_eq!(align_offset(8, 8).unwrap(), 8);
    }

    #[test]
    fn should_reject_invalid_alignment() {
        // Not a power of two.
        assert!(align_offset(0, 24).is_err());
        // Power of two but below the specification's minimum of 8.
        assert!(align_offset(0, 4).is_err());
        assert!(align_offset(0, 0).is_err());
    }

    #[test]
    fn should_size_header_by_version() {
        assert_eq!(GgufHeader::size(GgufVersion::V1), 16);
        assert_eq!(GgufHeader::size(GgufVersion::V2), 24);
        assert_eq!(GgufHeader::size(GgufVersion::V3), 24);
    }

    #[test]
    fn should_reject_unknown_version() {
        assert_eq!(GgufVersion::try_from(3).unwrap(), GgufVersion::V3);
        assert!(matches!(
            GgufVersion::try_from(4),
            Err(GgufError::UnsupportedVersion(4))
        ));
        assert!(GgufVersion::try_from(0).is_err());
    }

    #[test]
    fn should_enforce_limits() {
        assert!(check_limit("test", 10, 10).is_ok());
        assert!(check_limit("test", 11, 10).is_err());
    }

    #[test]
    fn should_read_le_integers() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(read_u32_le(&bytes, 0, "test").unwrap(), 0x0403_0201);
        assert_eq!(read_le_u64(&bytes), 0x0807_0605_0403_0201);
    }

    #[test]
    fn should_report_eof_instead_of_panicking() {
        let bytes = [0u8; 2];
        assert!(matches!(
            read_u32_le(&bytes, 0, "test"),
            Err(GgufError::UnexpectedEof { .. })
        ));
        // Even with an offset near `usize::MAX`, checked arithmetic must not wrap.
        assert!(read_u32_le(&bytes, usize::MAX, "test").is_err());
    }

    #[test]
    fn should_read_utf8_string() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(b"abc");
        // `read_string` borrows `len` bytes at `offset`; the length prefix sits before it.
        assert_eq!(read_string(&bytes, 8, 3, "test").unwrap(), "abc");
        assert!(read_string(&bytes, 8, 99, "test").is_err());
    }

    #[test]
    fn should_reject_invalid_utf8_string() {
        let bytes = [0xFFu8, 0xFE];
        assert!(matches!(
            read_string(&bytes, 0, 2, "test"),
            Err(GgufError::InvalidUtf8 { .. })
        ));
    }

    #[test]
    fn should_display_bad_magic_helpfully() {
        let error = GgufError::BadMagic { found: *b"NOPE" };
        let message = alloc::format!("{error}");
        assert!(message.contains("not a GGUF file"), "{message}");
        assert!(message.contains("NOPE"), "{message}");
    }
}
