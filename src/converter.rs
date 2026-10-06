//! Converting a GGUF file into dequantized SafeTensors.
//!
//! This exists for validation, not as a loading strategy. GGUF stays the better way to
//! *load* a model: it keeps the weights packed, which is the entire reason to use it. But
//! SafeTensors is what the rest of the toolchain understands, so being able to expand a
//! file lets you check a GGUF code path against a plain float one, or hand a model to
//! something that only reads SafeTensors.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use burn_tensor::DType;

use crate::base::GgufError;
use crate::quant::dequantize_checked;
use crate::reader::GgufReader;
use crate::tensor::GgmlType;

/// Errors produced by the converter.
#[derive(Debug)]
pub enum ConvertError {
    /// The GGUF file could not be read.
    Read(GgufError),
    /// A tensor could not be dequantized.
    Dequantize {
        /// The tensor's name.
        name: String,
        /// Its GGML type.
        tensor_type: GgmlType,
        /// What went wrong.
        message: String,
    },
    /// SafeTensors rejected the output.
    Serialize(String),
    /// A file operation failed.
    Io(String),
}

impl core::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Read(error) => write!(f, "{error}"),
            Self::Dequantize {
                name,
                tensor_type,
                message,
            } => write!(
                f,
                "failed to dequantize '{name}' ({tensor_type}): {message}"
            ),
            Self::Serialize(message) => write!(f, "failed to serialize SafeTensors: {message}"),
            Self::Io(message) => write!(f, "I/O error: {message}"),
        }
    }
}

impl core::error::Error for ConvertError {}

impl From<GgufError> for ConvertError {
    fn from(error: GgufError) -> Self {
        Self::Read(error)
    }
}

/// A tensor ready to be written to SafeTensors.
#[derive(Debug, Clone)]
pub struct ConvertedTensor {
    /// The tensor name, unchanged from GGUF.
    pub name: String,
    /// The shape in Burn's row-major order.
    pub shape: Vec<usize>,
    /// The dtype of the expanded values.
    pub dtype: DType,
    /// The expanded values.
    pub values: Vec<f32>,
    /// The GGUF type the values came from, so a caller can see what was unpacked.
    pub source_type: GgmlType,
}

/// The result of expanding a whole file.
#[derive(Debug, Clone)]
pub struct ConvertedModel {
    /// Every tensor, dequantized and in GGUF's own naming.
    pub tensors: Vec<ConvertedTensor>,
    /// The file's metadata, flattened to strings for the SafeTensors header.
    pub metadata: Vec<(String, String)>,
}

impl ConvertedModel {
    /// Total number of elements across every tensor.
    pub fn num_elements(&self) -> usize {
        self.tensors.iter().map(|tensor| tensor.values.len()).sum()
    }

    /// Which GGUF types the file contained, for a summary line.
    pub fn expanded_types(&self) -> Vec<GgmlType> {
        let mut types: Vec<GgmlType> = self
            .tensors
            .iter()
            .map(|tensor| tensor.source_type)
            .collect();
        types.sort_by_key(|ty| ty.name());
        types.dedup();
        types
    }
}

/// Expand every tensor in a GGUF file.
///
/// The output dtype is always `f32`, because that is what the dequantizer produces and
/// narrowing it would lose information the comparison is meant to preserve.
pub fn convert(reader: &GgufReader) -> Result<ConvertedModel, ConvertError> {
    let mut tensors = Vec::with_capacity(reader.tensors().len());

    for info in reader.tensors() {
        let num_elements = info.num_elements()?;
        let bytes = reader.tensor_data_bytes(info)?;
        let values = if info.tensor_type.is_quantized() {
            dequantize_checked(info.tensor_type, &bytes, num_elements).map_err(|error| {
                ConvertError::Dequantize {
                    name: info.name.clone(),
                    tensor_type: info.tensor_type,
                    message: format!("{error}"),
                }
            })?
        } else {
            read_plain(info.tensor_type, &bytes, num_elements).map_err(|error| {
                ConvertError::Dequantize {
                    name: info.name.clone(),
                    tensor_type: info.tensor_type,
                    message: format!("{error}"),
                }
            })?
        };

        let mut shape = info.dims.clone();
        shape.reverse();
        tensors.push(ConvertedTensor {
            name: info.name.clone(),
            shape,
            dtype: DType::F32,
            values,
            source_type: info.tensor_type,
        });
    }

    let metadata = reader
        .metadata()
        .iter()
        .map(|entry| {
            let value = entry.value.display_summary();
            // Truncate the tokenizer vocabulary, which would otherwise dominate the
            // header and make the file unreadable.
            let value = if value.len() > 256 {
                let mut truncated = value.chars().take(253).collect::<String>();
                truncated.push_str("...");
                truncated
            } else {
                value
            };
            (entry.key.clone(), value)
        })
        .collect();

    Ok(ConvertedModel { tensors, metadata })
}

/// Expand a GGUF file and write the result as SafeTensors.
#[cfg(feature = "std")]
pub fn convert_to_safetensors(
    reader: &GgufReader,
    path: impl AsRef<std::path::Path>,
) -> Result<ConvertedModel, ConvertError> {
    let model = convert(reader)?;
    write_safetensors(&model, path)?;
    Ok(model)
}

/// Write a converted model out as SafeTensors.
#[cfg(feature = "std")]
pub fn write_safetensors(
    model: &ConvertedModel,
    path: impl AsRef<std::path::Path>,
) -> Result<(), ConvertError> {
    use safetensors::tensor::{Dtype, TensorView, serialize};

    let mut tensors: Vec<(&str, TensorView<'_>)> = Vec::with_capacity(model.tensors.len());
    for tensor in &model.tensors {
        // SafeTensors stores raw little-endian bytes, so the `f32` values go over as
        // their own bit pattern rather than as a list of numbers.
        let data: &[u8] = bytemuck::cast_slice(&tensor.values);
        let view = TensorView::new(Dtype::F32, tensor.shape.clone(), data)
            .map_err(|error| ConvertError::Serialize(error.to_string()))?;
        tensors.push((tensor.name.as_str(), view));
    }

    let metadata: std::collections::HashMap<String, String> =
        model.metadata.iter().cloned().collect();
    let metadata = (!metadata.is_empty()).then_some(metadata);

    let bytes =
        serialize(tensors, metadata).map_err(|error| ConvertError::Serialize(error.to_string()))?;
    let path = path.as_ref();
    std::fs::write(path, bytes).map_err(|error| {
        ConvertError::Io(format!("failed to write {}: {error}", path.display()))
    })?;
    Ok(())
}

/// Read an unquantized tensor as `f32`.
fn read_plain(
    tensor_type: GgmlType,
    bytes: &[u8],
    num_elements: usize,
) -> Result<Vec<f32>, GgufError> {
    // The unquantized formats are plain scalars, so the dequantizer handles them too; this
    // wrapper exists to make the intent explicit at the call site.
    crate::quant::dequantize(tensor_type, bytes, num_elements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{GgufWriter, TensorSpec};
    use alloc::vec;

    fn reader(bytes: Vec<u8>) -> GgufReader {
        GgufReader::from_bytes(burn_tensor::Bytes::from_bytes_vec(bytes))
            .expect("the writer must produce a readable file")
    }

    #[test]
    fn should_expand_an_f32_tensor_unchanged() {
        let mut writer = GgufWriter::new();
        writer.tensor(TensorSpec::new("w", vec![4], GgmlType::F32).with_f32(&[1.0, 2.0, 3.0, 4.0]));
        let model = convert(&reader(writer.build())).unwrap();
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(model.tensors[0].shape, vec![4]);
        assert_eq!(model.tensors[0].dtype, DType::F32);
        assert_eq!(model.num_elements(), 4);
    }

    #[test]
    fn should_reverse_the_shape_into_burn_order() {
        let mut writer = GgufWriter::new();
        writer.tensor(TensorSpec::new("w", vec![4, 2], GgmlType::F32).with_f32(&[0.0; 8]));
        let model = convert(&reader(writer.build())).unwrap();
        // GGUF `[4, 2]` is Burn `[2, 4]`.
        assert_eq!(model.tensors[0].shape, vec![2, 4]);
    }

    #[test]
    fn should_expand_a_quantized_tensor() {
        let mut writer = GgufWriter::new();
        let mut payload = Vec::new();
        payload.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        payload.extend((0..32).map(|i| (i - 16) as i8 as u8));
        writer.tensor(TensorSpec::new("w", vec![32], GgmlType::Q8_0).with_bytes(&payload));

        let model = convert(&reader(writer.build())).unwrap();
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].values.len(), 32);
        for (index, value) in model.tensors[0].values.iter().enumerate() {
            let expected = (index as f32 - 16.0) * 0.5;
            assert!((value - expected).abs() < 1e-6, "at {index}: {value}");
        }
        assert_eq!(model.expanded_types(), vec![GgmlType::Q8_0]);
    }

    #[test]
    fn should_keep_tensor_names_as_they_are_in_the_file() {
        let mut writer = GgufWriter::new();
        writer.tensor(
            TensorSpec::new("blk.3.attn_q.weight", vec![2], GgmlType::F32).with_f32(&[1.0, 2.0]),
        );
        let model = convert(&reader(writer.build())).unwrap();
        // The converter is a compatibility path, so it does not rename anything; the store
        // does that when loading into a module.
        assert_eq!(model.tensors[0].name, "blk.3.attn_q.weight");
    }

    #[test]
    fn should_carry_metadata_as_strings() {
        let mut writer = GgufWriter::new();
        writer.metadata("general.architecture", "llama");
        writer.metadata("llama.block_count", 32u32);
        writer.tensor(TensorSpec::new("w", vec![1], GgmlType::F32).with_f32(&[0.0]));
        let model = convert(&reader(writer.build())).unwrap();
        let architecture = model
            .metadata
            .iter()
            .find(|(key, _)| key == "general.architecture")
            .map(|(_, value)| value.clone());
        assert_eq!(architecture.as_deref(), Some("llama"));
    }

    #[test]
    fn should_truncate_a_long_metadata_value() {
        let mut writer = GgufWriter::new();
        let long = "x".repeat(1000);
        writer.metadata(
            "tokenizer.ggml.tokens",
            crate::kv::MetadataValue::Array(
                (0..400)
                    .map(|index| crate::kv::MetadataValue::String(format!("{index}{long}")))
                    .collect(),
            ),
        );
        writer.tensor(TensorSpec::new("w", vec![1], GgmlType::F32).with_f32(&[0.0]));
        let model = convert(&reader(writer.build())).unwrap();
        let value = model
            .metadata
            .iter()
            .find(|(key, _)| key == "tokenizer.ggml.tokens")
            .map(|(_, value)| value.clone())
            .expect("the key is present");
        // The array summary is short, but a long string value must still be bounded.
        assert!(value.len() <= 256, "{} bytes", value.len());
    }

    #[test]
    fn should_report_a_format_it_cannot_expand() {
        let mut writer = GgufWriter::new();
        writer.tensor(TensorSpec::new("w", vec![256], GgmlType::Iq2Xxs));
        let error = convert(&reader(writer.build())).unwrap_err();
        assert!(
            matches!(error, ConvertError::Dequantize { .. }),
            "{error:?}"
        );
        let message = format!("{error}");
        assert!(message.contains("IQ2_XXS"), "{message}");
        assert!(message.contains("w"), "{message}");
    }

    #[test]
    fn should_reject_a_file_that_is_not_gguf() {
        // Three bytes is not a GGUF file, so the reader rejects it before conversion runs.
        let error =
            GgufReader::from_bytes(burn_tensor::Bytes::from_bytes_vec(vec![1, 2, 3])).unwrap_err();
        assert!(
            matches!(error, GgufError::UnexpectedEof { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn should_convert_a_file_with_no_tensors() {
        let mut writer = GgufWriter::new();
        writer.metadata("general.architecture", "llama");
        let model = convert(&reader(writer.build())).unwrap();
        assert!(model.tensors.is_empty());
        assert_eq!(model.num_elements(), 0);
        assert!(model.expanded_types().is_empty());
        assert_eq!(model.metadata.len(), 1);
    }
}
