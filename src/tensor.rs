//! `ggml_type` and the tensor descriptor.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use burn_tensor::DType;
use burn_tensor::quantization::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};

use crate::base::{GgufError, MAX_TENSOR_DIMS};

/// Super-block size used by every `K`-quant type.
pub const QK_K: usize = 256;

/// Number of bytes in a `Q4K`/`Q5K` sub-block scale and minimum table.
pub const K_SCALE_SIZE: usize = 12;

/// A GGUF tensor element type, mirroring `ggml_type`.
///
/// The discriminants are part of the file format and must not be reordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    /// 32-bit IEEE 754 float.
    F32 = 0,
    /// 16-bit IEEE 754 float.
    F16 = 1,
    /// 4-bit, one `f16` scale per 32 elements.
    Q4_0 = 2,
    /// 4-bit, `f16` scale and minimum per 32 elements.
    Q4_1 = 3,
    /// 5-bit, one `f16` scale per 32 elements.
    Q5_0 = 6,
    /// 5-bit, `f16` scale and minimum per 32 elements.
    Q5_1 = 7,
    /// 8-bit, one `f16` scale per 32 elements.
    Q8_0 = 8,
    /// 8-bit with a redundant sum term; intermediate use only.
    Q8_1 = 9,
    /// 2-bit `K`-quant: 16 sub-blocks of 16, 4-bit scale and minimum.
    Q2K = 10,
    /// 3-bit `K`-quant: 16 sub-blocks of 16, 6-bit scales.
    Q3K = 11,
    /// 4-bit `K`-quant: 8 sub-blocks of 32, 6-bit scale and minimum.
    Q4K = 12,
    /// 5-bit `K`-quant: 8 sub-blocks of 32, 6-bit scale and minimum.
    Q5K = 13,
    /// 6-bit `K`-quant: 16 sub-blocks of 16, 8-bit scales.
    Q6K = 14,
    /// 8-bit `K`-quant with per-group sums; intermediate use only.
    Q8K = 15,
    /// i-quant, 2.0625 bits per weight.
    Iq2Xxs = 16,
    /// i-quant, 2.3125 bits per weight.
    Iq2Xs = 17,
    /// i-quant, 3.0625 bits per weight.
    Iq3Xxs = 18,
    /// i-quant, 1.5625 bits per weight.
    Iq1S = 19,
    /// i-quant with a non-linear 4-bit codebook, 32 elements per block.
    Iq4Nl = 20,
    /// i-quant, 3.4375 bits per weight.
    Iq3S = 21,
    /// i-quant, 2.5625 bits per weight.
    Iq2S = 22,
    /// i-quant, 4.25 bits per weight.
    Iq4Xs = 23,
    /// 8-bit signed integer.
    I8 = 24,
    /// 16-bit signed integer.
    I16 = 25,
    /// 32-bit signed integer.
    I32 = 26,
    /// 64-bit signed integer.
    I64 = 27,
    /// 64-bit IEEE 754 float.
    F64 = 28,
    /// i-quant, 1.75 bits per weight.
    Iq1M = 29,
    /// bfloat16.
    Bf16 = 30,
    /// Ternary quant, 1.6875 bits per weight.
    Tq1_0 = 34,
    /// Ternary quant, 2.0625 bits per weight.
    Tq2_0 = 35,
    /// 4-bit with a shared `E8M0` exponent per 32 elements.
    MxFp4 = 39,
}

impl TryFrom<u32> for GgmlType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            39 => Self::MxFp4,
            other => return Err(GgufError::UnknownTensorType(other)),
        })
    }
}

impl GgmlType {
    /// Every type the format defines, including the ones removed from support.
    ///
    /// Tags 4, 5 and 31 to 33 once existed and were withdrawn; they are reported as
    /// unknown rather than guessed at.
    pub const ALL: &'static [Self] = &[
        Self::F32,
        Self::F16,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Q8K,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq3Xxs,
        Self::Iq1S,
        Self::Iq4Nl,
        Self::Iq3S,
        Self::Iq2S,
        Self::Iq4Xs,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F64,
        Self::Iq1M,
        Self::Bf16,
        Self::Tq1_0,
        Self::Tq2_0,
        Self::MxFp4,
    ];

    /// The number of logical elements one block of this type encodes.
    ///
    /// This is `1` for the unquantized types and `ggml_blck_size` for the quantized ones.
    pub const fn block_elements(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 | Self::F64 => 1,
            Self::I8 => 1,
            Self::I16 => 1,
            Self::I32 => 1,
            Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Iq4Nl => 32,
            Self::MxFp4 => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq4Xs
            | Self::Iq1M
            | Self::Tq1_0
            | Self::Tq2_0 => QK_K,
        }
    }

    /// The size in bytes of one block of this type.
    ///
    /// This is `ggml_type_size`. For the quantized types it includes the embedded
    /// scale, minimum and high-bit planes.
    pub const fn type_size(self) -> usize {
        match self {
            // Unquantized: the scalar width.
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::I8 => 1,

            // Legacy blocks: `f16 d` plus the packed quants, and for the `_1` variants
            // an extra `f16 m`.
            Self::Q4_0 | Self::Iq4Nl => 2 + 16,
            Self::Q4_1 => 2 + 2 + 16,
            Self::Q5_0 => 2 + 4 + 16,
            Self::Q5_1 => 2 + 2 + 4 + 16,
            Self::Q8_0 => 2 + 32,
            Self::Q8_1 => 2 + 2 + 32,

            // `K` super-blocks of 256 elements.
            Self::Q2K => 4 + (QK_K / 16) + (QK_K / 4),
            Self::Q3K => 2 + (QK_K / 8) + (QK_K / 4) + K_SCALE_SIZE,
            Self::Q4K => 4 + K_SCALE_SIZE + (QK_K / 2),
            Self::Q5K => 4 + K_SCALE_SIZE + (QK_K / 8) + (QK_K / 2),
            Self::Q6K => 2 + (QK_K / 16) + (QK_K / 2) + (QK_K / 4),
            Self::Q8K => 4 + QK_K + (QK_K / 16) * 2,

            // i-quants and ternary quants.
            Self::Iq2Xxs => 2 + QK_K / 8 * 2,
            Self::Iq2Xs => 2 + QK_K / 8 * 2 + QK_K / 32,
            Self::Iq3Xxs => 2 + 3 * (QK_K / 8),
            Self::Iq1S => 2 + QK_K / 8 + QK_K / 32 * 2,
            Self::Iq3S => 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + (QK_K / 64),
            Self::Iq2S => 2 + QK_K / 4 + QK_K / 32 + QK_K / 32,
            Self::Iq4Xs => 2 + 2 + (QK_K / 64) + (QK_K / 2),
            Self::Iq1M => QK_K / 8 + QK_K / 16 + QK_K / 32,
            Self::Tq1_0 => 2 + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5,
            Self::Tq2_0 => 2 + QK_K / 4,

            Self::MxFp4 => 1 + 16,
        }
    }

    /// Whether this type stores values in a quantized block format.
    pub const fn is_quantized(self) -> bool {
        self.block_elements() > 1
    }

    /// The Burn dtype for an unquantized type.
    ///
    /// Returns `None` for quantized types, which have no plain dtype.
    pub const fn dtype(self) -> Option<DType> {
        match self {
            Self::F32 => Some(DType::F32),
            Self::F16 => Some(DType::F16),
            Self::Bf16 => Some(DType::BF16),
            Self::F64 => Some(DType::F64),
            Self::I8 => Some(DType::I8),
            Self::I16 => Some(DType::I16),
            Self::I32 => Some(DType::I32),
            Self::I64 => Some(DType::I64),
            _ => None,
        }
    }

    /// The Burn quantization scheme this GGUF type converts into, if one exists.
    ///
    /// The mapping is deliberately conservative: a type only gets a scheme when the
    /// conversion is exact, with no approximation of the stored weights. Two of the
    /// legacy formats happen to line up perfectly with schemes Burn already has:
    ///
    /// * [`GgmlType::Q8_0`] stores `x = d * q` with an `f16` `d` every 32 values, which is
    ///   [`QuantValue::Q8F`] blocked by 32 with `f16` parameters.
    /// * [`GgmlType::Q4_0`] stores `x = d * (q - 8)` with an `f16` `d` every 32 values.
    ///   [`QuantValue::Q4F`] spans `-8..=7`, and Burn's `dequantize` kernel converts a
    ///   4-bit raw value `n` to `n - 16` when `n >= 8`, which is exactly `q - 8`. The
    ///   nibbles still need repacking; see [`crate::quant`].
    ///
    /// The `K`-quants and the asymmetric legacy formats need a scheme with a per-block
    /// offset and, for `Q6_K`, a 6-bit value width. Burn's `QuantMode` is symmetric
    /// only and its widest integer `QuantValue` is 8-bit, so they return `None` here and
    /// are dequantized instead. See the crate documentation.
    pub fn to_quant_scheme(self) -> Option<QuantScheme> {
        match self {
            Self::Q8_0 => Some(
                QuantScheme::default()
                    .with_value(QuantValue::Q8F)
                    // `Q8_0` is already one `i8` per value with no packing to undo.
                    .with_store(QuantStore::Native)
                    .with_level(QuantLevel::block([32]))
                    .with_param(QuantParam::F16),
            ),
            Self::Q4_0 => Some(
                QuantScheme::default()
                    .with_value(QuantValue::Q4F)
                    // Burn packs 8 four-bit values per `u32`, GGUF packs 2 per byte.
                    .with_store(QuantStore::PackedU32(0))
                    .with_level(QuantLevel::block([32]))
                    .with_param(QuantParam::F16),
            ),
            _ => None,
        }
    }

    /// Why this type cannot be executed natively in quantized form, when it cannot.
    ///
    /// Only meaningful for quantized types that [`GgmlType::to_quant_scheme`] rejects.
    pub const fn native_quant_gap(self) -> Option<&'static str> {
        match self {
            Self::Q8_0 | Self::Q4_0 => None,
            Self::Q4_1 | Self::Q5_1 => {
                Some("needs a per-block offset (zero point); Burn's QuantMode is symmetric only")
            }
            Self::Q5_0 => Some("needs a 5-bit QuantValue; Burn's integer widths are 2, 4 and 8"),
            Self::Q2K | Self::Q4K | Self::Q5K => Some(
                "K-quant needs both a per-sub-block offset and a two-level scale \
                 (f16 super-block scale times a 6-bit sub-block scale)",
            ),
            Self::Q6K => Some(
                "Q6_K needs a 6-bit QuantValue and a 16-element sub-block; \
                 Burn's integer widths are 2, 4 and 8",
            ),
            Self::Q3K => {
                Some("Q3K reconstructs quants from a popcount mask and needs custom packing")
            }
            Self::Q8_1 => {
                Some("Q8_1 stores a redundant per-block sum and is for intermediate use only")
            }
            Self::Q8K => {
                Some("Q8K is for intermediate quantization only and has no GGUF dequantizer")
            }
            Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq1M => {
                Some("i-quant encodes values through a codebook, not a scale and an integer")
            }
            Self::Iq4Nl => Some("i-quant uses a non-linear 4-bit codebook"),
            Self::Iq4Xs => Some("i-quant needs a two-level scale and a 4-bit codebook"),
            Self::Tq1_0 | Self::Tq2_0 => Some("ternary quant needs a shift-based reconstruction"),
            Self::MxFp4 => Some(
                "MXFP4 shares one E8M0 exponent per block, which is not a multiplicative scale",
            ),
            // The unquantized types are not quantized at all.
            _ => Some("not a quantized type"),
        }
    }

    /// The canonical `ggml` name of this type, as used in file names and `file_type`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq1S => "IQ1_S",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq2S => "IQ2_S",
            Self::Iq4Xs => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::Iq1M => "IQ1_M",
            Self::Bf16 => "BF16",
            Self::Tq1_0 => "TQ1_0",
            Self::Tq2_0 => "TQ2_0",
            Self::MxFp4 => "MXFP4",
        }
    }
}

impl core::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// A tensor descriptor from a GGUF file's tensor info table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// The tensor name, for example `blk.0.attn_q.weight`.
    pub name: String,
    /// Dimensions in GGUF order, where index 0 varies fastest.
    ///
    /// Burn is row-major with the last dimension varying fastest, so the shape handed to
    /// a Burn tensor is this vector reversed. See [`TensorInfo::burn_shape`].
    pub dims: Vec<usize>,
    /// The element type.
    pub tensor_type: GgmlType,
    /// Byte offset of this tensor's data relative to the start of the tensor data section.
    pub offset: usize,
}

impl TensorInfo {
    /// Build a descriptor, validating the dimension count.
    pub fn new(
        name: String,
        dims: Vec<usize>,
        tensor_type: GgmlType,
        offset: usize,
    ) -> Result<Self, GgufError> {
        if dims.is_empty() {
            return Err(GgufError::InvalidFormat(format!(
                "tensor '{name}' declares zero dimensions"
            )));
        }
        if dims.len() > MAX_TENSOR_DIMS {
            return Err(GgufError::LimitExceeded {
                what: "tensor dimension count",
                found: dims.len() as u64,
                limit: MAX_TENSOR_DIMS,
            });
        }
        if dims.contains(&0) {
            return Err(GgufError::InvalidFormat(format!(
                "tensor '{name}' has a zero-sized dimension: {dims:?}"
            )));
        }
        Ok(Self {
            name,
            dims,
            tensor_type,
            offset,
        })
    }

    /// The shape in Burn's row-major order.
    ///
    /// GGUF writes dimensions with the fastest-varying one first, the reverse of Burn,
    /// so the vector is reversed rather than copied.
    pub fn burn_shape(&self) -> burn_tensor::Shape {
        let mut dims = self.dims.clone();
        dims.reverse();
        burn_tensor::Shape::from(dims)
    }

    /// Total number of logical elements.
    pub fn num_elements(&self) -> Result<usize, GgufError> {
        self.dims.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| GgufError::TensorSizeOverflow {
                    name: self.name.clone(),
                })
        })
    }

    /// Size in bytes of this tensor's data, computed from its shape and type.
    pub fn nbytes(&self) -> Result<usize, GgufError> {
        let elements = self.num_elements()?;
        let block_elements = self.tensor_type.block_elements();
        let blocks = elements.div_ceil(block_elements);
        blocks
            .checked_mul(self.tensor_type.type_size())
            .ok_or_else(|| GgufError::TensorSizeOverflow {
                name: self.name.clone(),
            })
    }

    /// Validate that the shape is compatible with this type's block layout.
    ///
    /// Quantized types encode a whole number of values per block along the fastest
    /// dimension, so `dims[0]` must be a multiple of the block size.
    pub fn validate_block_layout(&self) -> Result<(), GgufError> {
        let block_elements = self.tensor_type.block_elements();
        if block_elements > 1 {
            let innermost = self.dims[0];
            if !innermost.is_multiple_of(block_elements) {
                return Err(GgufError::InvalidFormat(format!(
                    "tensor '{}' has {} as its innermost dimension, which is not a multiple \
                     of the {} elements per {} block",
                    self.name, innermost, block_elements, self.tensor_type,
                )));
            }
        }
        Ok(())
    }

    /// The number of quantization parameter entries this tensor needs.
    ///
    /// Only meaningful when [`GgmlType::to_quant_scheme`] returns a scheme; the count is
    /// the number of blocks, since every scheme Burn supports is blocked.
    pub fn num_blocks(&self) -> Result<usize, GgufError> {
        Ok(self
            .num_elements()?
            .div_ceil(self.tensor_type.block_elements()))
    }
}

/// The Burn representation a GGUF tensor is loaded into.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BurnLayout {
    /// A plain float tensor.
    Float(DType),
    /// A plain integer tensor.
    Int(DType),
    /// A tensor in Burn's packed quantized form, executed without dequantizing first.
    Quantized(QuantScheme),
}

impl BurnLayout {
    /// The dtype of the resulting Burn tensor.
    pub const fn dtype(self) -> DType {
        match self {
            Self::Float(dtype) | Self::Int(dtype) => dtype,
            Self::Quantized(scheme) => DType::QFloat(scheme),
        }
    }

    /// Whether this layout keeps the weights packed.
    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::Quantized(_))
    }
}

/// How a tensor type should be loaded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadStrategy {
    /// Keep the weights packed and run the backend's quantized kernels with this scheme.
    NativeQuantized(burn_tensor::quantization::QuantScheme),
    /// Expand the weights to a plain float tensor of this dtype.
    Dequantized(DType),
}

impl GgmlType {
    /// Decide how tensors of this type should be loaded.
    ///
    /// Quantized types with an exact scheme stay packed. Everything else, including every
    /// format Burn cannot execute natively, is expanded to `f32` so that loading always
    /// produces the right numbers.
    pub fn load_strategy(self, dequantized: DType) -> LoadStrategy {
        match self.to_quant_scheme() {
            Some(scheme) => LoadStrategy::NativeQuantized(scheme),
            None if self.is_quantized() => LoadStrategy::Dequantized(dequantized),
            None => LoadStrategy::Dequantized(self.dtype().unwrap_or(dequantized)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::vec;

    #[test]
    fn should_round_trip_every_defined_type_tag() {
        for ty in GgmlType::ALL {
            assert_eq!(GgmlType::try_from(*ty as u32).unwrap(), *ty, "for {ty}");
        }
    }

    #[test]
    fn should_reject_withdrawn_and_undefined_tags() {
        // 4, 5 and 31..=33 were withdrawn from the format.
        for tag in [4, 5, 31, 32, 33, 36, 37, 38, 40, 100, u32::MAX] {
            assert!(
                matches!(
                    GgmlType::try_from(tag),
                    Err(GgufError::UnknownTensorType(_))
                ),
                "tag {tag} should be rejected"
            );
        }
    }

    #[test]
    fn should_use_the_documented_block_sizes() {
        // Cross-checked against the `static_assert`s in `ggml-common.h`.
        assert_eq!(GgmlType::Q4_0.type_size(), 18);
        assert_eq!(GgmlType::Q4_1.type_size(), 20);
        assert_eq!(GgmlType::Q5_0.type_size(), 22);
        assert_eq!(GgmlType::Q5_1.type_size(), 24);
        assert_eq!(GgmlType::Q8_0.type_size(), 34);
        assert_eq!(GgmlType::Q8_1.type_size(), 36);
        assert_eq!(GgmlType::Q2K.type_size(), 84);
        assert_eq!(GgmlType::Q3K.type_size(), 110);
        assert_eq!(GgmlType::Q4K.type_size(), 144);
        assert_eq!(GgmlType::Q5K.type_size(), 176);
        assert_eq!(GgmlType::Q6K.type_size(), 210);
        assert_eq!(GgmlType::Q8K.type_size(), 292);
        assert_eq!(GgmlType::MxFp4.type_size(), 17);
    }

    #[test]
    fn should_use_the_documented_block_element_counts() {
        assert_eq!(GgmlType::F32.block_elements(), 1);
        for ty in [
            GgmlType::Q4_0,
            GgmlType::Q8_0,
            GgmlType::Iq4Nl,
            GgmlType::MxFp4,
        ] {
            assert_eq!(ty.block_elements(), 32, "for {ty}");
        }
        for ty in [
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Q2K,
            GgmlType::Iq2Xs,
            GgmlType::Tq2_0,
        ] {
            assert_eq!(ty.block_elements(), 256, "for {ty}");
        }
    }

    #[test]
    fn should_compute_bytes_per_weight() {
        // The published bits-per-weight figures, to one decimal place.
        let cases = [
            (GgmlType::Q4_0, 4.5),
            (GgmlType::Q4K, 4.5),
            (GgmlType::Q5K, 5.5),
            (GgmlType::Q6K, 6.5625),
            (GgmlType::Q8_0, 8.5),
            (GgmlType::Q2K, 2.625),
            (GgmlType::Q3K, 3.4375),
        ];
        for (ty, expected) in cases {
            let actual = ty.type_size() as f64 * 8.0 / ty.block_elements() as f64;
            assert!(
                (actual - expected).abs() < 0.001,
                "{ty}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn should_map_plain_types_to_dtypes() {
        assert_eq!(GgmlType::F32.dtype(), Some(DType::F32));
        assert_eq!(GgmlType::F16.dtype(), Some(DType::F16));
        assert_eq!(GgmlType::Bf16.dtype(), Some(DType::BF16));
        assert_eq!(GgmlType::F64.dtype(), Some(DType::F64));
        assert_eq!(GgmlType::I8.dtype(), Some(DType::I8));
        assert_eq!(GgmlType::I32.dtype(), Some(DType::I32));
        // Quantized types have no plain dtype.
        assert_eq!(GgmlType::Q8_0.dtype(), None);
        assert_eq!(GgmlType::Q4K.dtype(), None);
    }

    #[test]
    fn should_map_q8_0_and_q4_0_to_native_schemes() {
        let q8 = GgmlType::Q8_0.to_quant_scheme().expect("Q8_0 must map");
        assert_eq!(q8.value, QuantValue::Q8F);
        assert_eq!(q8.store, QuantStore::Native);
        assert_eq!(q8.level, QuantLevel::block([32]));
        assert_eq!(q8.param, QuantParam::F16);

        let q4 = GgmlType::Q4_0.to_quant_scheme().expect("Q4_0 must map");
        assert_eq!(q4.value, QuantValue::Q4F);
        assert_eq!(q4.store, QuantStore::PackedU32(0));
        assert_eq!(q4.level, QuantLevel::block([32]));
        assert_eq!(q4.param, QuantParam::F16);
    }

    #[test]
    fn should_report_a_reason_for_every_unmapped_quantized_type() {
        for ty in GgmlType::ALL.iter().filter(|ty| ty.is_quantized()) {
            if ty.to_quant_scheme().is_some() {
                assert_eq!(ty.native_quant_gap(), None, "{ty} maps natively");
                continue;
            }
            assert!(
                ty.native_quant_gap().is_some(),
                "{ty} has no scheme and must explain why"
            );
        }
    }

    #[test]
    fn should_reverse_dimensions_into_burn_shape() {
        // A llama embedding is GGUF `[n_embd, n_vocab]` and Burn `[n_vocab, n_embd]`.
        let info =
            TensorInfo::new("token_embd.weight".into(), vec![8, 32], GgmlType::Q4K, 0).unwrap();
        assert_eq!(info.burn_shape().as_slice(), &[32, 8]);
        assert_eq!(info.num_elements().unwrap(), 256);
    }

    #[test]
    fn should_compute_tensor_nbytes_from_shape_and_type() {
        let info = TensorInfo::new("w".into(), vec![32, 2], GgmlType::Q8_0, 0).unwrap();
        // Two rows of 32 values, one 34-byte block each.
        assert_eq!(info.nbytes().unwrap(), 68);

        let info = TensorInfo::new("w".into(), vec![64, 1], GgmlType::Q4_0, 0).unwrap();
        // Two 18-byte blocks.
        assert_eq!(info.nbytes().unwrap(), 36);

        let info = TensorInfo::new("w".into(), vec![256], GgmlType::Q6K, 0).unwrap();
        assert_eq!(info.nbytes().unwrap(), 210);
    }

    #[test]
    fn should_reject_shape_incompatible_with_the_block_layout() {
        // 30 is not a multiple of the 32-element Q8_0 block.
        let info = TensorInfo::new("w".into(), vec![30, 2], GgmlType::Q8_0, 0).unwrap();
        assert!(info.validate_block_layout().is_err());

        let info = TensorInfo::new("w".into(), vec![32, 2], GgmlType::Q8_0, 0).unwrap();
        assert!(info.validate_block_layout().is_ok());

        // Unquantized types impose no constraint.
        let info = TensorInfo::new("w".into(), vec![7, 2], GgmlType::F32, 0).unwrap();
        assert!(info.validate_block_layout().is_ok());
    }

    #[test]
    fn should_reject_degenerate_shapes() {
        assert!(TensorInfo::new("w".into(), vec![], GgmlType::F32, 0).is_err());
        assert!(TensorInfo::new("w".into(), vec![0, 2], GgmlType::F32, 0).is_err());
        assert!(TensorInfo::new("w".into(), vec![2; 9], GgmlType::F32, 0).is_err());
    }

    #[test]
    fn should_detect_shape_overflow() {
        let info = TensorInfo::new(
            "w".into(),
            vec![usize::MAX, usize::MAX, usize::MAX],
            GgmlType::F32,
            0,
        )
        .unwrap();
        assert!(matches!(
            info.num_elements(),
            Err(GgufError::TensorSizeOverflow { .. })
        ));
    }

    #[test]
    fn should_count_blocks() {
        let info = TensorInfo::new("w".into(), vec![64, 3], GgmlType::Q8_0, 0).unwrap();
        assert_eq!(info.num_blocks().unwrap(), 6);
    }

    #[test]
    fn should_keep_native_formats_packed_and_expand_the_rest() {
        assert!(matches!(
            GgmlType::Q8_0.load_strategy(DType::F32),
            LoadStrategy::NativeQuantized(_)
        ));
        assert!(matches!(
            GgmlType::Q4K.load_strategy(DType::F32),
            LoadStrategy::Dequantized(DType::F32)
        ));
        // Unquantized types carry their own dtype through unchanged.
        assert_eq!(
            GgmlType::F16.load_strategy(DType::F32),
            LoadStrategy::Dequantized(DType::F16)
        );
    }

    #[test]
    fn should_expose_layout_dtype_and_kind() {
        let scheme = GgmlType::Q8_0.to_quant_scheme().unwrap();
        let layout = BurnLayout::Quantized(scheme);
        assert_eq!(layout.dtype(), DType::QFloat(scheme));
        assert!(layout.is_quantized());

        let layout = BurnLayout::Float(DType::BF16);
        assert_eq!(layout.dtype(), DType::BF16);
        assert!(!layout.is_quantized());

        let layout = BurnLayout::Int(DType::I32);
        assert_eq!(layout.dtype(), DType::I32);
    }

    #[test]
    fn should_format_type_names() {
        assert_eq!(format!("{}", GgmlType::Q6K), "Q6_K");
        assert_eq!(format!("{}", GgmlType::Iq4Nl), "IQ4_NL");
        assert_eq!(format!("{}", GgmlType::Bf16), "BF16");
    }

    #[test]
    fn should_cover_every_type_in_the_all_table() {
        // Guards against adding an `ALL` entry without matching the other arms.
        let mut names: Vec<&str> = GgmlType::ALL.iter().map(|ty| ty.name()).collect();
        assert_eq!(names.len(), 32);
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 32, "type names must be unique");
    }
}
