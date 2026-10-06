//! GGUF metadata key-value pairs.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::base::{
    GgufError, MAX_ARRAY_DEPTH, MAX_ARRAY_ELEMENTS, check_limit, read_le_u64, read_string,
    read_u32_le,
};

/// The type tag of a GGUF metadata value.
///
/// Values are tagged inline: unlike tensors, metadata carries no separate dtype, so a
/// reader must read the tag before it can know how many bytes the value occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MetadataValueType {
    /// 8-bit unsigned integer.
    U8 = 0,
    /// 8-bit signed integer.
    I8 = 1,
    /// 16-bit unsigned integer.
    U16 = 2,
    /// 16-bit signed integer.
    I16 = 3,
    /// 32-bit unsigned integer.
    U32 = 4,
    /// 32-bit signed integer.
    I32 = 5,
    /// 32-bit IEEE 754 float.
    F32 = 6,
    /// Boolean, encoded as one byte where 0 is false and 1 is true.
    Bool = 7,
    /// Length-prefixed, non-null-terminated UTF-8 string.
    String = 8,
    /// A tagged, length-prefixed array of values, possibly nested.
    Array = 9,
    /// 64-bit unsigned integer.
    U64 = 10,
    /// 64-bit signed integer.
    I64 = 11,
    /// 64-bit IEEE 754 float.
    F64 = 12,
}

impl TryFrom<u32> for MetadataValueType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => return Err(GgufError::UnknownMetadataValueType(other)),
        })
    }
}

impl fmt::Display for MetadataValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Array => "array",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        };
        f.write_str(name)
    }
}

/// A decoded GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// 8-bit unsigned integer.
    U8(u8),
    /// 8-bit signed integer.
    I8(i8),
    /// 16-bit unsigned integer.
    U16(u16),
    /// 16-bit signed integer.
    I16(i16),
    /// 32-bit unsigned integer.
    U32(u32),
    /// 32-bit signed integer.
    I32(i32),
    /// 32-bit IEEE 754 float.
    F32(f32),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Array of values. May nest.
    Array(Vec<MetadataValue>),
    /// 64-bit unsigned integer.
    U64(u64),
    /// 64-bit signed integer.
    I64(i64),
    /// 64-bit IEEE 754 float.
    F64(f64),
}

impl MetadataValue {
    /// The type tag this value was stored with.
    pub const fn value_type(&self) -> MetadataValueType {
        match self {
            Self::U8(_) => MetadataValueType::U8,
            Self::I8(_) => MetadataValueType::I8,
            Self::U16(_) => MetadataValueType::U16,
            Self::I16(_) => MetadataValueType::I16,
            Self::U32(_) => MetadataValueType::U32,
            Self::I32(_) => MetadataValueType::I32,
            Self::F32(_) => MetadataValueType::F32,
            Self::Bool(_) => MetadataValueType::Bool,
            Self::String(_) => MetadataValueType::String,
            Self::Array(_) => MetadataValueType::Array,
            Self::U64(_) => MetadataValueType::U64,
            Self::I64(_) => MetadataValueType::I64,
            Self::F64(_) => MetadataValueType::F64,
        }
    }

    /// Read a boolean, or `None` if this value is not a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Read the value as a UTF-8 string, or `None` if it is not a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// Read the value as an array, or `None` if it is not an array.
    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    /// Read any integer value as `u64`.
    ///
    /// GGUF is loose about integer widths: the specification asks writers to prefer
    /// `u64` for counts but notes that some models use `u32`, so readers are expected to
    /// accept both. Negative values are rejected rather than reinterpreted, since a
    /// negative count is always a malformed file.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Self::U8(value) => *value as u64,
            Self::U16(value) => *value as u64,
            Self::U32(value) => *value as u64,
            Self::U64(value) => *value,
            Self::I8(value) => u8::try_from(*value).ok()? as u64,
            Self::I16(value) => u16::try_from(*value).ok()? as u64,
            Self::I32(value) => u32::try_from(*value).ok()? as u64,
            Self::I64(value) => u64::try_from(*value).ok()?,
            _ => return None,
        })
    }

    /// Read any integer value as `usize`, saturating at `usize::MAX`.
    ///
    /// Use this for tensor shapes and layer counts, where a value that cannot fit in
    /// `usize` on this host is better treated as "very large" than as a hard error; the
    /// subsequent size computation will fail its own bounds check.
    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64()
            .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
    }

    /// Read any integer or float value as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        Some(match self {
            Self::F32(value) => *value as f64,
            Self::F64(value) => *value,
            other => other.as_u64()? as f64,
        })
    }

    /// Read the value as `f32`, or `None` if it is not numeric.
    pub fn as_f32(&self) -> Option<f32> {
        self.as_f64().map(|value| value as f32)
    }

    /// The scalar rendered as a string, for diagnostics and the inspect example.
    ///
    /// Arrays are summarised by length rather than expanded, since the tokenizer
    /// vocabulary makes them far too large to print.
    pub fn display_summary(&self) -> String {
        match self {
            Self::U8(value) => alloc::format!("{value}"),
            Self::I8(value) => alloc::format!("{value}"),
            Self::U16(value) => alloc::format!("{value}"),
            Self::I16(value) => alloc::format!("{value}"),
            Self::U32(value) => alloc::format!("{value}"),
            Self::I32(value) => alloc::format!("{value}"),
            Self::U64(value) => alloc::format!("{value}"),
            Self::I64(value) => alloc::format!("{value}"),
            Self::F32(value) => alloc::format!("{value}"),
            Self::F64(value) => alloc::format!("{value}"),
            Self::Bool(value) => alloc::format!("{value}"),
            Self::String(value) => value.clone(),
            Self::Array(values) => {
                let inner = values
                    .first()
                    .map(MetadataValue::value_type)
                    .map(|ty| alloc::format!("{ty}"))
                    .unwrap_or_else(|| "?".into());
                alloc::format!("[{}; {}]", inner, values.len())
            }
        }
    }
}

/// A metadata key and its value.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataEntry {
    /// The key, which the specification requires to be hierarchical `lower_snake_case`
    /// separated by `.`, for example `llama.attention.head_count`.
    pub key: String,
    /// The value.
    pub value: MetadataValue,
}

impl MetadataEntry {
    /// Build an entry from a key and value.
    pub fn new(key: impl Into<String>, value: MetadataValue) -> Self {
        Self {
            key: key.into(),
            value,
        }
    }
}

impl From<&str> for MetadataValue {
    fn from(value: &str) -> Self {
        Self::String(String::from(value))
    }
}

impl From<String> for MetadataValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<u32> for MetadataValue {
    fn from(value: u32) -> Self {
        Self::U32(value)
    }
}

impl From<u64> for MetadataValue {
    fn from(value: u64) -> Self {
        Self::U64(value)
    }
}

impl From<bool> for MetadataValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<f32> for MetadataValue {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

/// Tracks how many array elements have been decoded while reading one metadata entry.
///
/// A single corrupt length field would otherwise let one entry allocate an unbounded
/// `Vec`, so every element decoded through a [`MetadataReader`] is counted against a
/// budget shared by that entry and all of its nested arrays.
#[derive(Debug)]
pub(crate) struct MetadataReader<'a> {
    bytes: &'a [u8],
    offset: usize,
    remaining: usize,
}

impl<'a> MetadataReader<'a> {
    pub(crate) fn new(bytes: &'a [u8], offset: usize) -> Self {
        Self {
            bytes,
            offset,
            remaining: MAX_ARRAY_ELEMENTS,
        }
    }

    pub(crate) const fn offset(&self) -> usize {
        self.offset
    }

    /// Consume one element from this entry's budget.
    fn take_element(&mut self) -> Result<(), GgufError> {
        self.remaining = self
            .remaining
            .checked_sub(1)
            .ok_or_else(|| GgufError::LimitExceeded {
                what: "metadata array element count",
                found: MAX_ARRAY_ELEMENTS as u64 + 1,
                limit: MAX_ARRAY_ELEMENTS,
            })?;
        Ok(())
    }

    /// Read a length-prefixed string, returning it and the new offset.
    pub(crate) fn read_string(
        &mut self,
        context: &'static str,
    ) -> Result<(String, usize), GgufError> {
        let start = self.offset;
        let len = read_le_u64(self.slice(8, context)?);
        self.offset = start + 8;
        let value = read_string(self.bytes, self.offset, len, context)?;
        let value = String::from(value);
        self.offset = self.offset.checked_add(len as usize).ok_or_else(|| {
            GgufError::InvalidFormat(alloc::format!("{context} overflows the file"))
        })?;
        Ok((value, start))
    }

    /// Borrow `len` bytes at the current offset, advancing past them.
    pub(crate) fn slice(
        &mut self,
        len: usize,
        context: &'static str,
    ) -> Result<&'a [u8], GgufError> {
        let end = self.offset.checked_add(len).ok_or_else(|| {
            GgufError::InvalidFormat(alloc::format!("{context} overflows the file"))
        })?;
        if end > self.bytes.len() {
            return Err(GgufError::UnexpectedEof {
                context,
                offset: self.offset,
                needed: len,
                available: self.bytes.len().saturating_sub(self.offset),
            });
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    /// Read a fixed-width scalar of `size` bytes at the current offset.
    fn read_scalar(&mut self, size: usize, context: &'static str) -> Result<u64, GgufError> {
        let slice = self.slice(size, context)?;
        let mut buf = [0u8; 8];
        buf[..size].copy_from_slice(slice);
        Ok(u64::from_le_bytes(buf))
    }

    /// Read one metadata value of the given type.
    pub(crate) fn read_value(
        &mut self,
        ty: MetadataValueType,
        context: &'static str,
    ) -> Result<MetadataValue, GgufError> {
        self.read_value_at_depth(ty, context, 0)
    }

    /// Read one metadata value, tracking how deeply arrays nest.
    ///
    /// The depth has to be threaded through rather than reset per element, otherwise a
    /// chain of arrays looks infinitely shallow to the limit check.
    fn read_value_at_depth(
        &mut self,
        ty: MetadataValueType,
        context: &'static str,
        depth: usize,
    ) -> Result<MetadataValue, GgufError> {
        let value = match ty {
            MetadataValueType::U8 => MetadataValue::U8(self.read_scalar(1, context)? as u8),
            MetadataValueType::I8 => MetadataValue::I8(self.read_scalar(1, context)? as u8 as i8),
            MetadataValueType::U16 => MetadataValue::U16(self.read_scalar(2, context)? as u16),
            MetadataValueType::I16 => {
                MetadataValue::I16(self.read_scalar(2, context)? as u16 as i16)
            }
            MetadataValueType::U32 => MetadataValue::U32(self.read_scalar(4, context)? as u32),
            MetadataValueType::I32 => {
                MetadataValue::I32(self.read_scalar(4, context)? as u32 as i32)
            }
            MetadataValueType::U64 => MetadataValue::U64(self.read_scalar(8, context)?),
            MetadataValueType::I64 => MetadataValue::I64(self.read_scalar(8, context)? as i64),
            MetadataValueType::F32 => {
                MetadataValue::F32(f32::from_bits(self.read_scalar(4, context)? as u32))
            }
            MetadataValueType::F64 => {
                MetadataValue::F64(f64::from_bits(self.read_scalar(8, context)?))
            }
            MetadataValueType::Bool => {
                let raw = self.read_scalar(1, context)? as u8;
                // The specification requires exactly 0 or 1; anything else means the
                // model or the reader is broken, and silently rounding would hide it.
                match raw {
                    0 => MetadataValue::Bool(false),
                    1 => MetadataValue::Bool(true),
                    other => {
                        return Err(GgufError::InvalidFormat(alloc::format!(
                            "{context}: bool must be 0 or 1, found {other}"
                        )));
                    }
                }
            }
            MetadataValueType::String => {
                let (value, _) = self.read_string(context)?;
                MetadataValue::String(value)
            }
            MetadataValueType::Array => self.read_array(depth, context)?,
        };
        Ok(value)
    }

    /// Read an array, recursing for nested arrays.
    fn read_array(
        &mut self,
        depth: usize,
        context: &'static str,
    ) -> Result<MetadataValue, GgufError> {
        if depth >= MAX_ARRAY_DEPTH {
            return Err(GgufError::LimitExceeded {
                what: "metadata array nesting depth",
                found: depth as u64 + 1,
                limit: MAX_ARRAY_DEPTH,
            });
        }

        // The element type is a tag, then a length in elements (not bytes).
        let element_type =
            MetadataValueType::try_from(read_u32_le(self.bytes, self.offset, context)?)?;
        self.offset += 4;
        let len = read_le_u64(self.slice(8, context)?);
        let len = check_limit("metadata array length", len, MAX_ARRAY_ELEMENTS)?;

        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            self.take_element()?;
            values.push(self.read_value_at_depth(element_type, context, depth + 1)?);
        }

        Ok(MetadataValue::Array(values))
    }
}

/// Read a metadata entry starting at `offset`, returning it and the offset just past it.
pub(crate) fn read_entry(bytes: &[u8], offset: usize) -> Result<(MetadataEntry, usize), GgufError> {
    let mut reader = MetadataReader::new(bytes, offset);
    let (key, _) = reader.read_string("metadata key")?;
    let value_type_offset = reader.offset();
    let value_type = MetadataValueType::try_from(read_u32_le(
        bytes,
        value_type_offset,
        "metadata value type",
    )?)?;
    reader.offset = value_type_offset + 4;
    let value = reader.read_value(value_type, "metadata value")?;
    Ok((MetadataEntry { key, value }, reader.offset()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::vec;

    /// Read one value of the given type from the start of `bytes`.
    fn read_value_at(
        bytes: &[u8],
        offset: usize,
        ty: MetadataValueType,
    ) -> Result<(MetadataValue, usize), GgufError> {
        let mut reader = MetadataReader::new(bytes, offset);
        let value = reader.read_value(ty, "test value")?;
        Ok((value, reader.offset()))
    }

    #[test]
    fn should_map_every_value_type_tag() {
        let expected = [
            (0u32, MetadataValueType::U8),
            (1, MetadataValueType::I8),
            (2, MetadataValueType::U16),
            (3, MetadataValueType::I16),
            (4, MetadataValueType::U32),
            (5, MetadataValueType::I32),
            (6, MetadataValueType::F32),
            (7, MetadataValueType::Bool),
            (8, MetadataValueType::String),
            (9, MetadataValueType::Array),
            (10, MetadataValueType::U64),
            (11, MetadataValueType::I64),
            (12, MetadataValueType::F64),
        ];
        for (tag, ty) in expected {
            assert_eq!(MetadataValueType::try_from(tag).unwrap(), ty);
            assert_eq!(ty as u32, tag);
        }
        assert!(matches!(
            MetadataValueType::try_from(13),
            Err(GgufError::UnknownMetadataValueType(13))
        ));
    }

    #[test]
    fn should_read_all_scalar_types() {
        let cases: &[(MetadataValueType, &[u8], MetadataValue)] = &[
            (MetadataValueType::U8, &[0xFF], MetadataValue::U8(255)),
            (MetadataValueType::I8, &[0xFF], MetadataValue::I8(-1)),
            (
                MetadataValueType::U16,
                &[0xFF, 0x7F],
                MetadataValue::U16(0x7FFF),
            ),
            (
                MetadataValueType::I16,
                &[0x00, 0x80],
                MetadataValue::I16(i16::MIN),
            ),
            (
                MetadataValueType::U32,
                &[0x01, 0x00, 0x00, 0x80],
                MetadataValue::U32(0x8000_0001),
            ),
            (
                MetadataValueType::I32,
                &[0xFF, 0xFF, 0xFF, 0xFF],
                MetadataValue::I32(-1),
            ),
            (
                MetadataValueType::U64,
                &[1, 0, 0, 0, 0, 0, 0, 0],
                MetadataValue::U64(1),
            ),
            (
                MetadataValueType::I64,
                &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                MetadataValue::I64(-1),
            ),
        ];

        for (ty, bytes, expected) in cases {
            let (value, _) = read_value_at(bytes, 0, *ty).unwrap();
            assert_eq!(&value, expected, "for {ty}");
            // The reader must consume exactly the scalar width.
            let (_, offset) = read_value_at(bytes, 0, *ty).unwrap();
            assert_eq!(offset, bytes.len(), "for {ty}");
        }
    }

    #[test]
    fn should_read_floats() {
        let (value, _) = read_value_at(&1.5f32.to_le_bytes(), 0, MetadataValueType::F32).unwrap();
        assert_eq!(value, MetadataValue::F32(1.5));

        let (value, _) =
            read_value_at(&(-2.25f64).to_le_bytes(), 0, MetadataValueType::F64).unwrap();
        assert_eq!(value, MetadataValue::F64(-2.25));
    }

    #[test]
    fn should_read_string() {
        let mut bytes = vec![5u8, 0, 0, 0, 0, 0, 0, 0];
        bytes.extend_from_slice(b"hello");

        let (value, offset) = read_value_at(&bytes, 0, MetadataValueType::String).unwrap();
        assert_eq!(value, MetadataValue::String("hello".into()));
        assert_eq!(offset, bytes.len());
    }

    #[test]
    fn should_read_bool_and_reject_other_bytes() {
        for (raw, expected) in [(0u8, false), (1, true)] {
            let (value, _) = read_value_at(&[raw], 0, MetadataValueType::Bool).unwrap();
            assert_eq!(value, MetadataValue::Bool(expected));
        }
        assert!(matches!(
            read_value_at(&[2], 0, MetadataValueType::Bool),
            Err(GgufError::InvalidFormat(_))
        ));
    }

    #[test]
    fn should_read_array_of_scalars() {
        // type tag u32, length 3, then three u32 values.
        let mut bytes = vec![4u8, 0, 0, 0];
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&20u32.to_le_bytes());
        bytes.extend_from_slice(&30u32.to_le_bytes());

        let (value, offset) = read_value_at(&bytes, 0, MetadataValueType::Array).unwrap();
        assert_eq!(
            value,
            MetadataValue::Array(vec![
                MetadataValue::U32(10),
                MetadataValue::U32(20),
                MetadataValue::U32(30),
            ])
        );
        assert_eq!(offset, bytes.len());
    }

    #[test]
    fn should_read_nested_array() {
        // `array<array<u8>>` with two inner arrays of two elements each. Every array is a
        // self-describing struct, so each inner array repeats its own type and length.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&9u32.to_le_bytes()); // outer element type = Array
        bytes.extend_from_slice(&2u64.to_le_bytes()); // outer length = 2
        // Inner array 0: type U8 (tag 0), length 2, elements 1 and 2.
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.push(1u8);
        bytes.push(2u8);
        // Inner array 1: type U8 (tag 0), length 2, elements 3 and 4.
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.push(3u8);
        bytes.push(4u8);

        let (value, offset) = read_value_at(&bytes, 0, MetadataValueType::Array).unwrap();
        assert_eq!(
            value,
            MetadataValue::Array(vec![
                MetadataValue::Array(vec![MetadataValue::U8(1), MetadataValue::U8(2)]),
                MetadataValue::Array(vec![MetadataValue::U8(3), MetadataValue::U8(4)]),
            ])
        );
        assert_eq!(offset, bytes.len());
    }

    #[test]
    fn should_reject_array_longer_than_the_file() {
        let mut bytes = vec![4u8, 0, 0, 0];
        bytes.extend_from_slice(&1_000_000u64.to_le_bytes());
        assert!(matches!(
            read_value_at(&bytes, 0, MetadataValueType::Array),
            Err(GgufError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn should_reject_absurd_array_length_against_the_limit() {
        let mut bytes = vec![4u8, 0, 0, 0];
        bytes.extend_from_slice(&(MAX_ARRAY_ELEMENTS as u64 + 1).to_le_bytes());
        assert!(matches!(
            read_value_at(&bytes, 0, MetadataValueType::Array),
            Err(GgufError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn should_reject_excessive_nesting() {
        // Build an array nested one level deeper than the limit.
        let mut bytes = Vec::new();
        for _ in 0..MAX_ARRAY_DEPTH + 1 {
            bytes.extend_from_slice(&9u32.to_le_bytes()); // element type = Array
            bytes.extend_from_slice(&1u64.to_le_bytes()); // length 1
        }
        assert!(matches!(
            read_value_at(&bytes, 0, MetadataValueType::Array),
            Err(GgufError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn should_read_entry_with_key_and_value() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&6u64.to_le_bytes()); // "a.b.cd" is six bytes
        bytes.extend_from_slice(b"a.b.cd");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // value type U32
        bytes.extend_from_slice(&77u32.to_le_bytes());

        let (entry, offset) = read_entry(&bytes, 0).unwrap();
        assert_eq!(entry.key, "a.b.cd");
        assert_eq!(entry.value, MetadataValue::U32(77));
        assert_eq!(offset, bytes.len());
    }

    #[test]
    fn should_widen_integer_widths_when_reading_counts() {
        // A count stored as u32 must still be readable as usize/u64.
        let value = MetadataValue::U32(4096);
        assert_eq!(value.as_u64(), Some(4096));
        assert_eq!(value.as_usize(), Some(4096));
        assert_eq!(value.as_f64(), Some(4096.0));

        // Negative integers must not wrap into huge unsigned counts.
        assert_eq!(MetadataValue::I32(-1).as_u64(), None);
        assert_eq!(MetadataValue::I64(-1).as_u64(), None);

        // The one non-negative range that fits still converts.
        assert_eq!(MetadataValue::I32(5).as_u64(), Some(5));
    }

    #[test]
    fn should_saturate_usize_instead_of_failing() {
        let value = MetadataValue::U64(u64::MAX);
        assert_eq!(value.as_usize(), Some(usize::MAX));
    }

    #[test]
    fn should_refuse_to_widen_non_numeric_values() {
        assert_eq!(MetadataValue::Bool(true).as_u64(), None);
        assert_eq!(MetadataValue::String("x".into()).as_u64(), None);
        assert_eq!(MetadataValue::Bool(true).as_f64(), None);
    }

    #[test]
    fn should_summarise_arrays_without_expanding_them() {
        let value = MetadataValue::Array(vec![MetadataValue::F32(1.0); 5]);
        assert_eq!(value.display_summary(), "[f32; 5]");
        assert_eq!(MetadataValue::Array(vec![]).display_summary(), "[?; 0]");
        assert_eq!(MetadataValue::String("hi".into()).display_summary(), "hi");
        assert_eq!(MetadataValue::Bool(false).display_summary(), "false");
    }

    #[test]
    fn should_format_errors() {
        let error = read_value_at(&[2], 0, MetadataValueType::Bool).unwrap_err();
        assert!(format!("{error}").contains("bool must be 0 or 1"));
    }
}
