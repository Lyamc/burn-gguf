//! A minimal GGUF writer.
//!
//! This exists so the crate can be tested against real files without checking in
//! hundreds of megabytes of fixtures, and so a quantized Burn tensor can be written back
//! out in GGUF form when that becomes useful. It is deliberately not a general-purpose
//! converter: it emits exactly what the reader understands and nothing more.

use alloc::string::String;
use alloc::vec::Vec;

use crate::base::{DEFAULT_ALIGNMENT, GgufVersion, MAGIC, align_offset};
use crate::kv::{MetadataEntry, MetadataValue, MetadataValueType};

/// A tensor to be written.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Tensor name.
    pub name: String,
    /// Dimensions in GGUF order, index 0 varying fastest.
    pub dims: Vec<usize>,
    /// Element type.
    pub tensor_type: crate::tensor::GgmlType,
    /// Already-encoded tensor bytes, exactly `type_size` bytes per block.
    pub data: Vec<u8>,
}

impl TensorSpec {
    /// Describe a tensor whose payload is already in GGUF's packed block format.
    pub fn new(
        name: impl Into<String>,
        dims: Vec<usize>,
        tensor_type: crate::tensor::GgmlType,
    ) -> Self {
        let elements = dims.iter().product::<usize>().max(1);
        let block_elements = tensor_type.block_elements();
        let blocks = elements.div_ceil(block_elements);
        Self {
            name: name.into(),
            dims,
            tensor_type,
            data: alloc::vec![0u8; blocks * tensor_type.type_size()],
        }
    }

    /// Replace the payload with raw bytes.
    pub fn with_data(mut self, data: &[u8]) -> Self {
        self.data = data.to_vec();
        self
    }

    /// Replace the payload with the little-endian encoding of `values`.
    ///
    /// Only for the unquantized types, where the in-memory and on-disk layouts agree.
    pub fn with_f32(mut self, values: &[f32]) -> Self {
        self.data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self
    }

    /// Replace the payload with the little-endian encoding of `values`.
    pub fn with_f64(mut self, values: &[f64]) -> Self {
        self.data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self
    }

    /// Replace the payload with the little-endian encoding of `values`.
    pub fn with_i32(mut self, values: &[i32]) -> Self {
        self.data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self
    }

    /// Replace the payload with raw bytes, for the quantized types.
    pub fn with_bytes(self, data: &[u8]) -> Self {
        self.with_data(data)
    }
}

/// Builds a GGUF byte stream.
#[derive(Debug, Default)]
pub struct GgufWriter {
    metadata: Vec<MetadataEntry>,
    tensors: Vec<TensorSpec>,
    alignment: usize,
}

impl GgufWriter {
    /// A writer with the default alignment of 32.
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
        }
    }

    /// Add a metadata entry.
    ///
    /// The value is generic so callers can write `writer.metadata("k", 3u32)` or
    /// `writer.metadata("k", "llama")` directly.
    pub fn metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> &mut Self {
        self.metadata.push(MetadataEntry::new(key, value.into()));
        self
    }

    /// Set the global alignment, defaulting to 32 when never called.
    pub fn alignment(&mut self, alignment: usize) -> &mut Self {
        self.alignment = alignment;
        self
    }

    /// Add a tensor.
    ///
    /// The payload comes from the spec, so a caller that built one with
    /// [`TensorSpec::with_f32`] and one that built one with
    /// [`TensorSpec::with_bytes`] both go through here.
    pub fn tensor(&mut self, spec: TensorSpec) -> &mut Self {
        self.tensors.push(spec);
        self
    }

    /// Serialize to a version 3 file.
    pub fn build(&self) -> Vec<u8> {
        self.build_with(GgufVersion::V3)
    }

    /// Serialize to a version 1 file, whose counts are 32 bits wide.
    pub fn build_v1(&self) -> Vec<u8> {
        self.build_with(GgufVersion::V1)
    }

    /// Serialize to the given version.
    pub fn build_with(&self, version: GgufVersion) -> Vec<u8> {
        // The reader reads `general.alignment` out of the metadata, so the writer has to
        // agree with itself: fold the chosen alignment into the entry list when it is not
        // the default, and drop any caller-supplied duplicate.
        let mut metadata: Vec<MetadataEntry> = self
            .metadata
            .iter()
            .filter(|entry| entry.key != crate::reader::KEY_ALIGNMENT)
            .cloned()
            .collect();
        if self.alignment != DEFAULT_ALIGNMENT {
            metadata.push(MetadataEntry::new(
                crate::reader::KEY_ALIGNMENT,
                MetadataValue::U32(self.alignment as u32),
            ));
        }

        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&version.as_u32().to_le_bytes());
        if version.counts_are_u64() {
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        } else {
            out.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
            out.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        }

        for entry in &metadata {
            write_entry(&mut out, entry);
        }

        // Tensor offsets are relative to the start of the data section, so they can only
        // be computed once the descriptor size is known.
        let descriptors_start = out.len();
        let mut offset = 0usize;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for spec in &self.tensors {
            offsets.push(offset);
            offset = align_offset(offset + spec.data.len(), self.alignment)
                .unwrap_or(offset + spec.data.len());
        }
        let descriptors_end =
            descriptors_start + self.tensors.iter().map(descriptor_len).sum::<usize>();

        for (spec, offset) in self.tensors.iter().zip(offsets) {
            write_descriptor(&mut out, spec, offset);
        }

        // Pad to the alignment, then lay out the payloads in descriptor order.
        let aligned = align_offset(descriptors_end, self.alignment).unwrap_or(descriptors_end);
        out.resize(aligned, 0);
        for spec in &self.tensors {
            out.extend_from_slice(&spec.data);
            let padded = align_offset(out.len(), self.alignment).unwrap_or(out.len());
            out.resize(padded, 0);
        }

        out
    }
}

/// Byte length of a tensor descriptor, including the name.
fn descriptor_len(spec: &TensorSpec) -> usize {
    // u64 length + name + u32 dims + dims * u64 + u32 type + u64 offset.
    8 + spec.name.len() + 4 + spec.dims.len() * 8 + 4 + 8
}

fn write_entry(out: &mut Vec<u8>, entry: &MetadataEntry) {
    out.extend_from_slice(&(entry.key.len() as u64).to_le_bytes());
    out.extend_from_slice(entry.key.as_bytes());
    out.extend_from_slice(&(entry.value.value_type() as u32).to_le_bytes());
    write_value(out, &entry.value);
}

fn write_value(out: &mut Vec<u8>, value: &MetadataValue) {
    match value {
        MetadataValue::U8(v) => out.push(*v),
        MetadataValue::I8(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::Bool(v) => out.push(u8::from(*v)),
        MetadataValue::String(v) => {
            out.extend_from_slice(&(v.len() as u64).to_le_bytes());
            out.extend_from_slice(v.as_bytes());
        }
        MetadataValue::Array(values) => {
            let element_type = values
                .first()
                .map(MetadataValue::value_type)
                .unwrap_or(MetadataValueType::U8);
            out.extend_from_slice(&(element_type as u32).to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                write_value(out, value);
            }
        }
    }
}

fn write_descriptor(out: &mut Vec<u8>, spec: &TensorSpec, offset: usize) {
    out.extend_from_slice(&(spec.name.len() as u64).to_le_bytes());
    out.extend_from_slice(spec.name.as_bytes());
    out.extend_from_slice(&(spec.dims.len() as u32).to_le_bytes());
    for dim in &spec.dims {
        out.extend_from_slice(&(*dim as u64).to_le_bytes());
    }
    out.extend_from_slice(&(spec.tensor_type as u32).to_le_bytes());
    out.extend_from_slice(&(offset as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::MetadataValue;
    use burn_tensor::Bytes;

    fn reader(bytes: Vec<u8>) -> crate::reader::GgufReader {
        crate::reader::GgufReader::from_bytes(Bytes::from_bytes_vec(bytes))
            .expect("the writer must produce a readable file")
    }

    #[test]
    fn should_size_a_spec_payload_for_its_shape() {
        let spec = TensorSpec::new("w", vec![64], crate::tensor::GgmlType::Q8_0);
        // Two 34-byte blocks.
        assert_eq!(spec.data.len(), 68);

        let spec = TensorSpec::new("w", vec![2, 3], crate::tensor::GgmlType::F32);
        assert_eq!(spec.data.len(), 24);
    }

    #[test]
    fn should_replace_the_payload() {
        let spec = TensorSpec::new("w", vec![2], crate::tensor::GgmlType::F32)
            .with_data(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(spec.data, vec![1u8, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn should_encode_scalar_payloads() {
        let spec =
            TensorSpec::new("w", vec![2], crate::tensor::GgmlType::F32).with_f32(&[1.0, -2.0]);
        assert_eq!(spec.data, vec![0x00u8, 0x00, 0x80, 0x3F, 0, 0, 0, 0xC0]);

        let spec = TensorSpec::new("w", vec![1], crate::tensor::GgmlType::I32).with_i32(&[-1]);
        assert_eq!(spec.data, vec![0xFFu8, 0xFF, 0xFF, 0xFF]);

        let spec = TensorSpec::new("w", vec![1], crate::tensor::GgmlType::F64).with_f64(&[0.5]);
        assert_eq!(spec.data.len(), 8);
        assert_eq!(spec.data[7], 0x3F);
    }

    #[test]
    fn should_emit_the_magic_and_version() {
        let bytes = GgufWriter::new().build();
        assert_eq!(&bytes[..4], b"GGUF");
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            3
        );
        // And the reader agrees.
        assert_eq!(reader(bytes).version(), crate::base::GgufVersion::V3);
    }

    #[test]
    fn should_fold_a_custom_alignment_into_the_metadata() {
        let mut writer = GgufWriter::new();
        writer.tensor(
            TensorSpec::new("w", vec![2], crate::tensor::GgmlType::F32).with_f32(&[1.0, 2.0]),
        );
        writer.alignment(64);
        assert_eq!(reader(writer.build()).alignment(), 64);
    }

    #[test]
    fn should_drop_a_caller_supplied_duplicate_alignment() {
        // A stray `general.alignment` of 999 is not a power of two. If the writer let it
        // through, the reader would reject the file it just produced.
        let mut writer = GgufWriter::new();
        writer.metadata("general.alignment", 999u32);
        writer.tensor(
            TensorSpec::new("w", vec![2], crate::tensor::GgmlType::F32).with_f32(&[1.0, 2.0]),
        );
        writer.alignment(64);
        assert_eq!(reader(writer.build()).alignment(), 64);
    }

    #[test]
    fn should_align_tensor_offsets() {
        let mut writer = GgufWriter::new();
        // An F16 tensor of 1 element is 2 bytes, so the next one must start at 32.
        writer
            .tensor(TensorSpec::new("a", vec![1], crate::tensor::GgmlType::F16).with_data(&[1, 2]));
        writer
            .tensor(TensorSpec::new("b", vec![1], crate::tensor::GgmlType::F16).with_data(&[3, 4]));
        let bytes = writer.build();
        assert_eq!(bytes.len() % 32, 0);

        let reader = reader(bytes);
        let a = reader.tensor("a").unwrap();
        let b = reader.tensor("b").unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 32);
        // And the payloads really are at those offsets.
        assert_eq!(&reader.tensor_data_bytes(a).unwrap()[..], &[1u8, 2]);
        assert_eq!(&reader.tensor_data_bytes(b).unwrap()[..], &[3u8, 4]);
    }

    #[test]
    fn should_round_trip_tensor_names_and_shapes() {
        let mut writer = GgufWriter::new();
        writer.tensor(
            TensorSpec::new(
                "blk.3.attn_q.weight",
                vec![8, 2],
                crate::tensor::GgmlType::F32,
            )
            .with_f32(&[1.0; 16]),
        );
        let reader = reader(writer.build());
        let info = reader.tensor("blk.3.attn_q.weight").unwrap();
        assert_eq!(info.dims, vec![8, 2]);
        assert_eq!(info.burn_shape().as_slice(), &[2, 8]);
        assert!(reader.tensor("nope").is_none());
    }

    #[test]
    fn should_round_trip_v1_headers() {
        let mut writer = GgufWriter::new();
        writer.metadata("general.architecture", "llama");
        writer.tensor(
            TensorSpec::new("w", vec![2], crate::tensor::GgmlType::F32).with_f32(&[1.0, 2.0]),
        );
        let reader = reader(writer.build_v1());
        assert_eq!(reader.version(), crate::base::GgufVersion::V1);
        assert_eq!(reader.architecture(), Some("llama"));
        assert_eq!(reader.tensors().len(), 1);
    }

    #[test]
    fn should_round_trip_every_metadata_value_type() {
        use alloc::vec;

        let mut writer = GgufWriter::new();
        let cases = vec![
            ("k.u8", MetadataValue::U8(7)),
            ("k.i8", MetadataValue::I8(-7)),
            ("k.u16", MetadataValue::U16(70000u32 as u16)),
            ("k.i16", MetadataValue::I16(-300)),
            ("k.u32", MetadataValue::U32(4_000_000_000)),
            ("k.i32", MetadataValue::I32(-2_000_000_000)),
            ("k.u64", MetadataValue::U64(u64::MAX)),
            ("k.i64", MetadataValue::I64(i64::MIN)),
            ("k.f32", MetadataValue::F32(0.5)),
            ("k.f64", MetadataValue::F64(-0.25)),
            ("k.bool", MetadataValue::Bool(true)),
            ("k.string", MetadataValue::String("hello".into())),
            (
                "k.array",
                MetadataValue::Array(vec![MetadataValue::F32(1.0), MetadataValue::F32(2.0)]),
            ),
        ];
        for (key, value) in &cases {
            writer.metadata(*key, value.clone());
        }
        writer.tensor(
            TensorSpec::new("w", vec![2], crate::tensor::GgmlType::F32).with_f32(&[1.0, 2.0]),
        );

        let reader = reader(writer.build());
        for (key, value) in &cases {
            assert_eq!(reader.metadata_value(key), Some(value), "for {key}");
        }
    }
}
