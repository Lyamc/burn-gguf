//! Adapting GGUF tensors to the names and layouts Burn modules expect.

use alloc::boxed::Box;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_store::{TensorSnapshot, TensorSnapshotError};
use burn_tensor::DType;

use crate::base::GgufError;
use crate::quant::{dequantize_checked, repack_to_burn};
use crate::reader::GgufReader;
use crate::tensor::{GgmlType, LoadStrategy, TensorInfo};

/// How a GGUF tensor should be turned into Burn tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conversion {
    /// Keep the weights packed in Burn's quantized layout.
    ///
    /// Only the formats with an exact scheme mapping take this path; see
    /// [`GgmlType::to_quant_scheme`].
    NativeQuantized,
    /// Expand to a float tensor of this dtype.
    Dequantized(DType),
}

impl Conversion {
    /// Choose how to load a tensor of this type.
    ///
    /// `strict` decides what happens when a format cannot stay packed. Left off, the
    /// weights are expanded so that loading always produces the right numbers; turned
    /// on, it is an error, which is what a caller who needs the memory saving wants.
    pub fn for_type(
        tensor_type: GgmlType,
        dequantized: DType,
        strict: bool,
    ) -> Result<Self, GgufError> {
        match tensor_type.load_strategy(dequantized) {
            LoadStrategy::NativeQuantized(_) => Ok(Self::NativeQuantized),
            LoadStrategy::Dequantized(dtype) => {
                if strict && tensor_type.is_quantized() {
                    let reason = tensor_type.native_quant_gap().unwrap_or("unsupported");
                    return Err(GgufError::InvalidFormat(format!(
                        "{tensor_type} cannot stay quantized because it {reason}"
                    )));
                }
                Ok(Self::Dequantized(dtype))
            }
        }
    }

    /// The dtype the resulting Burn tensor will have.
    pub fn dtype(self, tensor_type: GgmlType) -> DType {
        match self {
            Self::NativeQuantized => DType::QFloat(tensor_type.to_quant_scheme().expect("checked")),
            Self::Dequantized(dtype) => dtype,
        }
    }
}

/// Renames GGUF tensors onto the paths a Burn module visits, and converts their payload.
///
/// # What belongs here and what does not
///
/// Only the mechanical part of architecture mapping lives here. GGUF's standardized
/// names follow a single convention across transformer architectures:
///
/// ```text
/// token_embd.weight       blk.N.attn_norm.weight
/// output.weight           blk.N.attn_q.weight
/// output_norm.weight      blk.N.ffn_gate.weight
/// ```
///
/// and so does the naming Burn's own modules generate, just with different words for the
/// same thing. Translating between the two vocabularies is renaming, which is a per-tensor
/// string substitution and belongs in the store.
///
/// Deciding that `architecture = llama` means `blk.N.attn_q` is a query projection with
/// `n_head` heads is *not* mechanical: it depends on the architecture and on how the model
/// was written, so it belongs to a `Llama` implementation above the parser. This crate
/// deliberately stops short of that. [`GgufReader::architecture`] exposes the key that
/// selects such an implementation.
#[derive(Debug, Clone)]
pub struct GgufAdapter {
    /// Extra path rewrites, applied in order after the built-in rules.
    ///
    /// The first element of each pair is a regular expression and the second is its
    /// replacement. Use this for architecture-specific names the convention does not
    /// cover.
    remap: Vec<(regex::Regex, String)>,
    /// Fail rather than expanding a quantized tensor that cannot stay packed.
    strict: bool,
    /// The dtype expanded weights are produced in.
    dequantized_dtype: DType,
}

impl Default for GgufAdapter {
    fn default() -> Self {
        Self {
            remap: Vec::new(),
            strict: false,
            dequantized_dtype: DType::F32,
        }
    }
}

impl GgufAdapter {
    /// An adapter that expands any quantized weight it cannot keep packed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Require every quantized tensor to stay packed.
    ///
    /// Without this, a `Q4_K` model loads correctly but as `f32`, which defeats the point
    /// of loading it. Turning this on turns that situation into an error instead.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Set the dtype that expanded weights are produced in.
    ///
    /// `f16` is the usual choice for inference, halving the memory an expanded model
    /// occupies.
    pub fn dequantized_dtype(mut self, dtype: DType) -> Self {
        self.dequantized_dtype = dtype;
        self
    }

    /// Add a path rewrite, applied after the built-in rules.
    ///
    /// Takes the pattern as a string so callers do not have to depend on `regex`
    /// themselves; an invalid pattern is reported rather than panicking.
    pub fn with_remap(
        mut self,
        pattern: &str,
        replacement: impl Into<String>,
    ) -> Result<Self, regex::Error> {
        self.remap
            .push((regex::Regex::new(pattern)?, replacement.into()));
        Ok(self)
    }

    /// How a tensor of this type should be converted.
    pub fn conversion(&self, tensor_type: GgmlType) -> Result<Conversion, GgufError> {
        Conversion::for_type(tensor_type, self.dequantized_dtype, self.strict)
    }

    /// Translate a GGUF tensor name into the path a Burn module visits.
    ///
    /// Returns `None` when no rule matches, which leaves the original name in place. That
    /// is the right default for a model whose parameter names already match, and it means
    /// an unknown architecture degrades to a name mismatch rather than to silence.
    pub fn rename(&self, gguf_name: &str, architecture: Option<&str>) -> Option<String> {
        // The standardized names are shared across transformer architectures, so nothing
        // here branches on `architecture` yet. It is accepted so that a caller can pass
        // the key that will eventually select architecture-specific rules, rather than
        // silently ignoring it.
        let _ = architecture;

        let renamed = rewrite_layer_prefix(gguf_name)?;
        for (pattern, replacement) in &self.remap {
            if pattern.is_match(&renamed) {
                return Some(pattern.replace(&renamed, replacement.as_str()).into_owned());
            }
        }
        Some(renamed)
    }

    /// Build the snapshot a Burn module will consume for one GGUF tensor.
    ///
    /// The payload is produced inside the closure so a tensor the module never visits is
    /// never decoded. For a quantized model that is the difference between reading a few
    /// thousand descriptors and expanding the whole file.
    ///
    /// `reader` is shared rather than borrowed because a [`TensorSnapshot`] holds a
    /// `'static` closure, and a mapped file is worth keeping alive rather than copying.
    pub(crate) fn snapshot(
        &self,
        reader: &Rc<GgufReader>,
        info: &TensorInfo,
    ) -> Result<TensorSnapshot, GgufError> {
        let conversion = self.conversion(info.tensor_type)?;
        let dtype = conversion.dtype(info.tensor_type);
        let shape = info.burn_shape();
        let num_elements = info.num_elements()?;
        let path = self
            .rename(&info.name, reader.architecture())
            .unwrap_or_else(|| info.name.clone());
        let path_stack: Vec<String> = path.split('.').map(String::from).collect();

        let reader = Rc::clone(reader);
        let name = info.name.clone();
        let tensor_type = info.tensor_type;
        let closure_shape = shape.clone();
        let data_fn = Rc::new(
            move || -> Result<burn_tensor::TensorData, TensorSnapshotError> {
                let info = reader.tensor(&name).ok_or_else(|| {
                    TensorSnapshotError::DataError(format!("tensor '{name}' is no longer present"))
                })?;
                let bytes = reader
                    .tensor_bytes(info)
                    .map_err(|error| TensorSnapshotError::IoError(format!("{error}")))?;

                match conversion {
                    Conversion::NativeQuantized => {
                        let scheme = tensor_type.to_quant_scheme().ok_or_else(|| {
                            TensorSnapshotError::DataError(format!(
                                "{tensor_type} has no packed conversion"
                            ))
                        })?;
                        let packed = repack_to_burn(tensor_type, &bytes, scheme, num_elements)
                            .map_err(|error| TensorSnapshotError::DataError(format!("{error}")))?;
                        // The dtype carries the scheme, and it is what tells the backend where
                        // to split this blob into values and scales, so it is passed straight
                        // through rather than restated here.
                        Ok(burn_tensor::TensorData::from_bytes_vec(
                            packed,
                            closure_shape.clone(),
                            dtype,
                        ))
                    }
                    Conversion::Dequantized(_) => {
                        let values = dequantize_checked(tensor_type, &bytes, num_elements)
                            .map_err(|error| TensorSnapshotError::DataError(format!("{error}")))?;
                        Ok(burn_tensor::TensorData::new(values, closure_shape.clone()))
                    }
                }
            },
        );

        Ok(TensorSnapshot::from_closure(
            data_fn,
            dtype,
            shape,
            path_stack,
            Vec::new(),
            burn_core::module::ParamId::new(),
        ))
    }
}

/// Rewrite the standardized GGUF layer naming onto Burn's vocabulary.
///
/// The layer index and the projection name are the only parts that differ:
///
/// | GGUF | Burn |
/// |---|---|
/// | `token_embd.weight` | `embedding.weight` |
/// | `output.weight` | `lm_head.weight` |
/// | `output_norm.weight` | `norm.weight` |
/// | `blk.N.attn_norm.weight` | `layers.N.attention_norm.weight` |
/// | `blk.N.attn_q.weight` | `layers.N.self_attn.q_proj.weight` |
/// | `blk.N.attn_k.weight` | `layers.N.self_attn.k_proj.weight` |
/// | `blk.N.attn_v.weight` | `layers.N.self_attn.v_proj.weight` |
/// | `blk.N.attn_output.weight` | `layers.N.self_attn.o_proj.weight` |
/// | `blk.N.attn_norm_2.weight` | `layers.N.post_attention_norm.weight` |
/// | `blk.N.ffn_gate.weight` | `layers.N.feed_forward.gate_proj.weight` |
/// | `blk.N.ffn_up.weight` | `layers.N.feed_forward.up_proj.weight` |
/// | `blk.N.ffn_down.weight` | `layers.N.feed_forward.down_proj.weight` |
///
/// Names outside that table are returned unchanged, which covers the `ssm_*` and `ffn_*_exp`
/// families as well as anything a future architecture introduces.
fn rewrite_layer_prefix(name: &str) -> Option<String> {
    let Some((prefix, rest)) = name.split_once('.') else {
        return Some(name.to_string());
    };

    match prefix {
        "token_embd" => Some(format!("embedding.{rest}")),
        "output_norm" => Some(format!("norm.{rest}")),
        "output" => Some(format!("lm_head.{rest}")),
        "blk" => {
            let (index, rest) = rest.split_once('.')?;
            let suffix = match rest {
                "attn_norm.weight" => "attention_norm.weight",
                "attn_norm_2.weight" | "attn_norm_2.bias" => "post_attention_norm.weight",
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "ffn_norm.weight" => "post_ffw_norm.weight",
                "ffn_gate.weight" => "feed_forward.gate_proj.weight",
                "ffn_up.weight" => "feed_forward.up_proj.weight",
                "ffn_down.weight" => "feed_forward.down_proj.weight",
                // Anything else in a layer keeps its name, so the expert and state-space
                // families are not folded into the dense vocabulary.
                other => return Some(format!("layers.{index}.{other}")),
            };
            Some(format!("layers.{index}.{suffix}"))
        }
        // Not part of the standardized naming, so leave it for the caller's own rules.
        _ => Some(name.to_string()),
    }
}

/// A [`ModuleAdapter`](burn_store::ModuleAdapter) wrapper, for composing with other stores.
///
/// This is mostly useful for documentation and for callers that want to plug a GGUF load
/// into a chain that also renames tensors; [`GgufStore`](crate::GgufStore) applies this
/// adapter itself, so most callers never need it.
impl burn_store::ModuleAdapter for GgufAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        // Nothing to do: the conversion and renaming happen in `snapshot`, which already
        // knows the tensor's GGUF type. This method only exists to satisfy the trait.
        snapshot.clone()
    }

    fn clone_box(&self) -> Box<dyn burn_store::ModuleAdapter> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantScheme, QuantValue};

    #[test]
    fn should_map_the_standardized_names() {
        let adapter = GgufAdapter::new();
        let cases = [
            ("token_embd.weight", "embedding.weight"),
            ("output_norm.weight", "norm.weight"),
            ("output.weight", "lm_head.weight"),
            ("blk.0.attn_norm.weight", "layers.0.attention_norm.weight"),
            (
                "blk.3.attn_norm_2.weight",
                "layers.3.post_attention_norm.weight",
            ),
            ("blk.3.attn_q.weight", "layers.3.self_attn.q_proj.weight"),
            ("blk.12.attn_k.weight", "layers.12.self_attn.k_proj.weight"),
            ("blk.12.attn_v.weight", "layers.12.self_attn.v_proj.weight"),
            (
                "blk.12.attn_output.weight",
                "layers.12.self_attn.o_proj.weight",
            ),
            (
                "blk.12.ffn_gate.weight",
                "layers.12.feed_forward.gate_proj.weight",
            ),
            (
                "blk.12.ffn_up.weight",
                "layers.12.feed_forward.up_proj.weight",
            ),
            (
                "blk.12.ffn_down.weight",
                "layers.12.feed_forward.down_proj.weight",
            ),
        ];
        for (from, to) in cases {
            assert_eq!(
                adapter.rename(from, Some("llama")).as_deref(),
                Some(to),
                "for {from}"
            );
        }
    }

    #[test]
    fn should_keep_the_layer_index_but_not_rename_an_unknown_projection() {
        let adapter = GgufAdapter::new();
        // The layer prefix is standardized, so it is rewritten whatever the projection is
        // called. Only the standardized projections themselves are renamed.
        for name in [
            "blk.0.ssm_in.weight",
            "blk.0.ffn_gate_exp.weight",
            "blk.0.something.new.weight",
            "blk.0.attn_q.bias",
        ] {
            let expected = name.replace("blk.0.", "layers.0.");
            assert_eq!(
                adapter.rename(name, Some("llama")).as_deref(),
                Some(expected.as_str()),
                "for {name}"
            );
        }

        // Names outside a layer keep every character.
        for name in ["rope_freqs", "a_bare_name", "custom.thing.weight"] {
            assert_eq!(
                adapter.rename(name, Some("llama")).as_deref(),
                Some(name),
                "for {name}"
            );
        }
    }

    #[test]
    fn should_keep_moe_names_distinct() {
        let adapter = GgufAdapter::new();
        // Expert tensors must not be folded into the dense feed-forward names.
        assert_eq!(
            adapter.rename("blk.0.ffn_gate_exp.weight", None).as_deref(),
            Some("layers.0.ffn_gate_exp.weight")
        );
        assert_eq!(
            adapter.rename("blk.0.ffn_gate_inp.weight", None).as_deref(),
            Some("layers.0.ffn_gate_inp.weight")
        );
    }

    #[test]
    fn should_apply_a_caller_supplied_rule_after_the_builtin_ones() {
        let adapter = GgufAdapter::new()
            .with_remap(r"^layers\.0\.custom\.weight$", "special.head")
            .unwrap();
        assert_eq!(
            adapter.rename("blk.0.custom.weight", None).as_deref(),
            Some("special.head")
        );
        // And it does not disturb the rest.
        assert_eq!(
            adapter.rename("blk.0.attn_q.weight", None).as_deref(),
            Some("layers.0.self_attn.q_proj.weight")
        );
    }

    #[test]
    fn should_reject_an_invalid_remap_pattern() {
        assert!(GgufAdapter::new().with_remap("(", "x").is_err());
        assert!(GgufAdapter::new().with_remap("[a-", "x").is_err());
    }

    #[test]
    fn should_keep_supported_formats_packed() {
        let adapter = GgufAdapter::new();
        assert_eq!(
            adapter.conversion(GgmlType::Q8_0).unwrap(),
            Conversion::NativeQuantized
        );
        assert_eq!(
            adapter.conversion(GgmlType::Q4_0).unwrap(),
            Conversion::NativeQuantized
        );
    }

    #[test]
    fn should_expand_unsupported_formats_by_default() {
        let adapter = GgufAdapter::new();
        for ty in [GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K, GgmlType::Q3K] {
            assert_eq!(
                adapter.conversion(ty).unwrap(),
                Conversion::Dequantized(DType::F32),
                "for {ty}"
            );
        }
    }

    #[test]
    fn should_error_in_strict_mode_instead_of_expanding() {
        let adapter = GgufAdapter::new().strict(true);
        // The formats Burn can execute stay packed.
        assert_eq!(
            adapter.conversion(GgmlType::Q8_0).unwrap(),
            Conversion::NativeQuantized
        );
        // The rest become an error explaining what is missing.
        for ty in [GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
            let error = adapter.conversion(ty).unwrap_err();
            let message = format!("{error}");
            assert!(message.contains(ty.name()), "{message}");
            assert!(
                message.contains("offset") || message.contains("6-bit"),
                "the reason should name the missing capability: {message}"
            );
        }
    }

    #[test]
    fn should_still_load_unquantized_types_in_strict_mode() {
        let adapter = GgufAdapter::new().strict(true);
        assert_eq!(
            adapter.conversion(GgmlType::F32).unwrap(),
            Conversion::Dequantized(DType::F32)
        );
        assert_eq!(
            adapter.conversion(GgmlType::F16).unwrap(),
            Conversion::Dequantized(DType::F16)
        );
    }

    #[test]
    fn should_report_the_dtype_a_conversion_produces() {
        assert_eq!(
            Conversion::NativeQuantized.dtype(GgmlType::Q8_0),
            DType::QFloat(
                QuantScheme::default()
                    .with_value(QuantValue::Q8F)
                    .with_store(burn_tensor::quantization::QuantStore::Native)
                    .with_level(burn_tensor::quantization::QuantLevel::block([32]))
                    .with_param(burn_tensor::quantization::QuantParam::F16)
            )
        );
        assert_eq!(
            Conversion::Dequantized(DType::BF16).dtype(GgmlType::F32),
            DType::BF16
        );
    }
}
