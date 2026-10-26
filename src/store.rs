//! Loading GGUF files into Burn modules.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use burn_store::burn_pack::Tensor as PackTensor;
use burn_store::{ApplyResult, ModuleSnapshot, ModuleStore, PathFilter};
use burn_tensor::DType;

use crate::adapter::{Conversion, GgufAdapter};
use crate::base::GgufError;
use crate::reader::GgufReader;
use crate::tensor::TensorInfo;

/// Errors produced by [`GgufStore`].
#[derive(Debug)]
pub enum GgufStoreError {
    /// The container could not be read.
    Read(GgufError),
    /// A tensor could not be converted.
    Conversion {
        /// The tensor's GGUF name.
        name: String,
        /// What went wrong.
        message: String,
    },
    /// A configuration was inconsistent.
    Config(String),
}

impl core::fmt::Display for GgufStoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Read(error) => write!(f, "{error}"),
            Self::Conversion { name, message } => {
                write!(f, "failed to convert tensor '{name}': {message}")
            }
            Self::Config(message) => write!(f, "invalid GGUF store configuration: {message}"),
        }
    }
}

impl core::error::Error for GgufStoreError {}

impl From<GgufError> for GgufStoreError {
    fn from(error: GgufError) -> Self {
        Self::Read(error)
    }
}

/// A read-only store backed by a GGUF file.
///
/// Loading works like SafeTensors: build the module, hand it a store, and every parameter
/// the store knows about is replaced in place. What differs is what the payloads are.
///
/// ```no_run
/// # use burn_gguf::GgufStore;
/// # fn load(
/// #     model: &mut burn_nn::Linear,
/// #     path: &str,
/// # ) -> Result<(), burn_gguf::GgufStoreError> {
/// let mut store = GgufStore::from_file(path)?;
/// store.load_into(model).map(|_| ())
/// # }
/// ```
///
/// # Memory
///
/// Descriptors are parsed eagerly, which costs microseconds for a model with thousands of
/// tensors. Payloads are produced inside the tensor's data closure, so a parameter the
/// module never visits is never decoded. With the default `memmap` feature the bytes a
/// tensor *does* need are a subslice of the mapping rather than a copy.
#[derive(Debug)]
pub struct GgufStore {
    reader: Arc<GgufReader>,
    adapter: GgufAdapter,
    /// Which parameters to touch. `None` means all of them.
    filter: PathFilter,
    /// Whether a parameter with no matching tensor is an error.
    allow_partial: bool,
    /// Whether dtype or shape disagreements count as failures.
    validate: bool,
    /// Descriptors, built on first use and reused afterwards.
    snapshots: Option<BTreeMap<String, PackTensor>>,
}

impl GgufStore {
    /// Open a GGUF file, memory-mapping it when available.
    #[cfg(feature = "std")]
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, GgufStoreError> {
        Ok(Self::new(GgufReader::open(path)?))
    }

    /// Open an already-parsed GGUF file.
    pub fn from_reader(reader: GgufReader) -> Result<Self, GgufStoreError> {
        Ok(Self::new(reader))
    }

    /// Open a GGUF file held in memory.
    #[cfg(feature = "std")]
    pub fn from_bytes(data: burn_tensor::Bytes) -> Result<Self, GgufStoreError> {
        Ok(Self::new(GgufReader::from_bytes(data)?))
    }

    /// Wrap a reader with default settings.
    pub fn new(reader: GgufReader) -> Self {
        Self {
            reader: Arc::new(reader),
            adapter: GgufAdapter::new(),
            filter: PathFilter::none(),
            allow_partial: true,
            validate: false,
            snapshots: None,
        }
    }

    /// Restrict loading to the paths a filter matches.
    ///
    /// Loading one layer at a time is the main reason to reach for this: an adapter or a
    /// fine-tune usually touches a handful of tensors in an otherwise large file.
    pub fn with_filter(mut self, filter: PathFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Load every parameter, ignoring any the file does not contain.
    ///
    /// This is the default, because a GGUF file usually holds only the weights and not the
    /// buffers a model needs at run time.
    pub fn with_allow_partial(mut self, allow_partial: bool) -> Self {
        self.allow_partial = allow_partial;
        self
    }

    /// Fail the load if any parameter is missing.
    pub fn require_complete(self) -> Self {
        self.with_allow_partial(false)
    }

    /// Treat dtype and shape disagreements as failures.
    ///
    /// Off by default, matching the SafeTensors store.
    pub fn with_validate(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }

    /// Use a custom adapter, for architecture-specific naming or a different dtype for
    /// expanded weights.
    pub fn with_adapter(mut self, adapter: GgufAdapter) -> Self {
        self.adapter = adapter;
        self
    }

    /// Require every quantized tensor to stay packed.
    pub fn strict(mut self) -> Self {
        self.adapter = self.adapter.clone().strict(true);
        self
    }

    /// The underlying reader, for metadata and raw access.
    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    /// The architecture this file declares, if any.
    pub fn architecture(&self) -> Option<&str> {
        self.reader.architecture()
    }

    /// How many tensors will be exposed to the module.
    pub fn len(&self) -> usize {
        self.reader.tensors().len()
    }

    /// Whether the file contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.reader.tensors().is_empty()
    }

    /// Build every descriptor, reporting which ones cannot be converted.
    fn ensure_snapshots(&mut self) -> Result<(), GgufStoreError> {
        if self.snapshots.is_some() {
            return Ok(());
        }

        let mut snapshots = BTreeMap::new();
        for info in self.reader.tensors() {
            let tensor = self.adapter.tensor(&self.reader, info).map_err(|error| {
                GgufStoreError::Conversion {
                    name: info.name.clone(),
                    message: format!("{error}"),
                }
            })?;
            let name = tensor.name.clone();
            snapshots.insert(name, tensor);
        }
        self.snapshots = Some(snapshots);
        Ok(())
    }

    /// Load the store into a module.
    ///
    /// Equivalent to [`ModuleStore::load_from`], spelled out so callers do not have to
    /// import the trait for the common case.
    pub fn load_into<M>(&mut self, module: &mut M) -> Result<ApplyResult, GgufStoreError>
    where
        M: ModuleSnapshot,
    {
        ModuleStore::apply_to(self, module)
    }

    /// Add a tensor path rewrite, applied after the built-in GGUF naming rules.
    ///
    /// This is the hook for architecture-specific names the standardized convention does not
    /// cover. Invalid patterns are reported rather than panicking.
    pub fn with_remap(mut self, pattern: &str, replacement: &str) -> Result<Self, regex::Error> {
        self.adapter = self.adapter.clone().with_remap(pattern, replacement)?;
        self.snapshots = None;
        Ok(self)
    }

    /// Describe one tensor, without converting it.
    ///
    /// Useful for deciding what a filter should cover before building one.
    pub fn describe(&self, name: &str) -> Option<(&TensorInfo, Conversion, DType, String)> {
        let info = self.reader.tensor(name)?;
        let conversion = self.adapter.conversion(info.tensor_type).ok()?;
        let dtype = conversion.dtype(info.tensor_type);
        let path = self
            .adapter
            .rename(name, self.reader.architecture())
            .unwrap_or_else(|| name.to_string());
        Some((info, conversion, dtype, path))
    }

    /// Every tensor name, together with the path a module would look it up under.
    pub fn describe_all(&self) -> Vec<(&str, String, Conversion, DType)> {
        self.reader
            .tensors()
            .iter()
            .filter_map(|info| {
                let conversion = self.adapter.conversion(info.tensor_type).ok()?;
                let path = self
                    .adapter
                    .rename(&info.name, self.reader.architecture())
                    .unwrap_or_else(|| info.name.clone());
                Some((
                    info.name.as_str(),
                    path,
                    conversion,
                    conversion.dtype(info.tensor_type),
                ))
            })
            .collect()
    }
}

impl ModuleStore for GgufStore {
    type Error = GgufStoreError;

    /// Always fails: GGUF is an interchange format, not somewhere Burn writes models.
    fn collect_from<M: ModuleSnapshot>(&mut self, _module: &M) -> Result<(), Self::Error> {
        Err(GgufStoreError::Config(
            "GGUF is a read-only interchange format; use burn-store to save a model".into(),
        ))
    }

    /// Replace the module's parameters with the file's tensors.
    fn apply_to<M: ModuleSnapshot>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        self.ensure_snapshots()?;
        let snapshots: Vec<PackTensor> = self
            .snapshots
            .as_ref()
            .expect("just populated")
            .values()
            .cloned()
            .collect();
        let filter = self.filter.clone();
        let filter = if filter.is_empty() {
            None
        } else {
            Some(filter)
        };
        // GGUF carries no parameter identities, so there is nothing to gain from
        // descending into enum variants; a model tree holds one variant of each type.
        let result = module.apply(snapshots, filter, None, true);

        if self.validate && !result.errors.is_empty() {
            return Err(GgufStoreError::Config(format!(
                "the module did not match the file: {result}"
            )));
        }
        if !self.allow_partial && !result.missing.is_empty() {
            return Err(GgufStoreError::Config(format!(
                "the file is missing parameters of the module: {result}"
            )));
        }
        Ok(result)
    }

    fn get_tensor(&mut self, name: &str) -> Result<Option<&PackTensor>, Self::Error> {
        self.ensure_snapshots()?;
        Ok(self.snapshots.as_ref().and_then(|all| all.get(name)))
    }

    fn get_all_tensors(&mut self) -> Result<&BTreeMap<String, PackTensor>, Self::Error> {
        self.ensure_snapshots()?;
        Ok(self
            .snapshots
            .as_ref()
            .expect("ensure_snapshots either populates or errors"))
    }

    fn keys(&mut self) -> Result<Vec<String>, Self::Error> {
        self.ensure_snapshots()?;
        Ok(self
            .snapshots
            .as_ref()
            .expect("just populated")
            .keys()
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::MetadataValue;
    use crate::tensor::GgmlType;

    use crate::writer::{GgufWriter, TensorSpec};
    use alloc::vec;

    fn store_from(bytes: Vec<u8>) -> GgufStore {
        GgufStore::from_bytes(burn_tensor::Bytes::from_bytes_vec(bytes))
            .expect("the writer must produce a loadable file")
    }

    /// A file with one F32 tensor named as a normalized GGUF tensor.
    fn sample() -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.metadata("general.architecture", "llama");
        writer.tensor(
            TensorSpec::new("output_norm.weight", vec![4], GgmlType::F32)
                .with_f32(&[1.0, 2.0, 3.0, 4.0]),
        );
        writer.build()
    }

    #[test]
    fn should_expose_metadata_and_counts() {
        let store = store_from(sample());
        assert_eq!(store.architecture(), Some("llama"));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert_eq!(store.reader().tensors().len(), 1);
    }

    #[test]
    fn should_expose_renamed_paths() {
        let store = store_from(sample());
        let described = store.describe_all();
        assert_eq!(described.len(), 1);
        assert_eq!(described[0].0, "output_norm.weight");
        assert_eq!(described[0].1, "norm.weight");
        assert_eq!(described[0].2, Conversion::Dequantized(DType::F32));
        assert_eq!(described[0].3, DType::F32);
    }

    #[test]
    fn should_describe_a_single_tensor() {
        let store = store_from(sample());
        let (info, conversion, dtype, path) = store
            .describe("output_norm.weight")
            .expect("the tensor exists");
        assert_eq!(info.dims, vec![4]);
        assert_eq!(conversion, Conversion::Dequantized(DType::F32));
        assert_eq!(dtype, DType::F32);
        assert_eq!(path, "norm.weight");
        assert!(store.describe("nope").is_none());
    }

    #[test]
    fn should_list_keys_under_the_burn_paths() {
        let mut store = store_from(sample());
        assert_eq!(
            store.keys().unwrap(),
            vec![alloc::string::String::from("norm.weight")]
        );
    }

    #[test]
    fn should_refuse_to_save() {
        let mut store = store_from(sample());
        // A real module, so the `ModuleStore` bounds are satisfied the way a caller would
        // satisfy them.
        let linear = burn_nn::LinearConfig::new(4, 4).init(&Default::default());
        let error = store.collect_from(&linear).unwrap_err();
        assert!(matches!(error, GgufStoreError::Config(_)), "{error:?}");
        assert!(format!("{error}").contains("read-only"), "{error}");
    }

    #[test]
    fn should_report_a_read_failure_for_a_malformed_file() {
        let error =
            GgufStore::from_bytes(burn_tensor::Bytes::from_bytes_vec(vec![1, 2, 3])).unwrap_err();
        assert!(matches!(error, GgufStoreError::Read(_)), "{error:?}");
    }

    #[test]
    fn should_reject_a_strict_load_of_a_format_it_cannot_pack() {
        let mut writer = GgufWriter::new();
        writer.tensor(TensorSpec::new("w", vec![256], GgmlType::Q4K));
        let mut store = store_from(writer.build()).strict();
        let error = store.ensure_snapshots().unwrap_err();
        assert!(
            matches!(error, GgufStoreError::Conversion { .. }),
            "{error:?}"
        );
        assert!(format!("{error}").contains("Q4_K"), "{error}");
    }

    #[test]
    fn should_expand_the_same_format_when_not_strict() {
        let mut writer = GgufWriter::new();
        writer.tensor(TensorSpec::new("w", vec![256], GgmlType::Q4K));
        let mut store = store_from(writer.build());
        assert!(store.ensure_snapshots().is_ok());
    }

    #[test]
    fn should_defer_conversion_until_a_tensor_is_read() {
        // `IQ2_XXS` has no dequantizer, so the snapshot can be built but its payload
        // cannot. That is exactly the shape of work worth deferring: a descriptor is cheap,
        // and a tensor the module never visits should cost nothing.
        let mut writer = GgufWriter::new();
        writer.metadata("general.architecture", "llama");
        writer.tensor(TensorSpec::new("w", vec![256], GgmlType::Iq2Xxs));
        let mut store = store_from(writer.build());

        // The descriptor builds fine, with a plain float dtype already decided.
        store.ensure_snapshots().unwrap();
        let snapshot = store.get_tensor("w").unwrap().unwrap();
        assert_eq!(snapshot.shape.as_slice(), &[256]);
        assert_eq!(snapshot.dtype, DType::F32);

        // Reading the data fails, proving the work was deferred to that point.
        assert!(burn_store::bridge::to_data(snapshot).is_err());
    }

    #[test]
    fn should_expose_a_quantized_dtype_for_a_packed_format() {
        let mut writer = GgufWriter::new();
        writer.tensor(TensorSpec::new(
            "blk.0.attn_q.weight",
            vec![32],
            GgmlType::Q8_0,
        ));
        let mut store = store_from(writer.build());
        let snapshot = store
            .get_tensor("layers.0.self_attn.q_proj.weight")
            .unwrap()
            .unwrap();
        assert!(
            matches!(snapshot.dtype, DType::QFloat(_)),
            "expected a quantized dtype, got {:?}",
            snapshot.dtype
        );
    }

    #[test]
    fn packed_weights_must_reproduce_the_dequantized_values() {
        // One Q8_0 block, loaded packed through the store, must agree with what the
        // dequantizer produces from the same bytes. This is the invariant that makes the
        // packed path usable at all: Burn's kernel will unpack the blob we hand it, and the
        // numbers it gets back have to be the ones the file described.
        let mut writer = GgufWriter::new();
        let mut payload = Vec::new();
        payload.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        payload.extend((0..32).map(|i| (i - 16) as i8 as u8));
        writer.tensor(TensorSpec::new("w", vec![32], GgmlType::Q8_0).with_bytes(&payload));
        let bytes = writer.build();

        let expected =
            crate::quant::dequantize(GgmlType::Q8_0, &payload, 32).expect("a full block decodes");
        assert_eq!(expected.len(), 32);

        let mut store = store_from(bytes);
        let snapshot = store.get_tensor("w").unwrap().unwrap();
        assert!(
            matches!(snapshot.dtype, DType::QFloat(_)),
            "Q8_0 should stay packed, got {:?}",
            snapshot.dtype
        );
        let packed = burn_store::bridge::to_data(snapshot).unwrap();
        let blob = packed.as_bytes();

        // The blob is `[values][scales]`: 32 bytes of `i8`, then 2 bytes of `f16`.
        assert_eq!(blob.len(), 32 + 2);
        let scale = half::f16::from_le_bytes([blob[32], blob[33]]).to_f32();
        assert_eq!(scale, 0.5);

        for index in 0..32 {
            let value = blob[index] as i8 as f32 * scale;
            assert!(
                (value - expected[index]).abs() < 1e-6,
                "at {index}: {value} vs {}",
                expected[index]
            );
        }
    }

    #[test]
    fn packed_q4_0_weights_must_reproduce_the_dequantized_values() {
        // The same invariant for the sub-byte format, which additionally exercises the
        // nibble reshuffle and the sign-bit flip.
        let mut writer = GgufWriter::new();
        let mut payload = Vec::new();
        payload.extend_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
        for byte in 0..16u8 {
            payload.push(byte | ((15 - byte) << 4));
        }
        writer.tensor(TensorSpec::new("w", vec![32], GgmlType::Q4_0).with_bytes(&payload));
        let bytes = writer.build();

        let expected =
            crate::quant::dequantize(GgmlType::Q4_0, &payload, 32).expect("a full block decodes");

        let mut store = store_from(bytes);
        let snapshot = store.get_tensor("w").unwrap().unwrap();
        assert!(matches!(snapshot.dtype, DType::QFloat(_)));
        let packed = burn_store::bridge::to_data(snapshot).unwrap();
        let blob = packed.as_bytes();

        // 32 four-bit values become 16 bytes, followed by a 2-byte `f16` scale.
        assert_eq!(blob.len(), 16 + 2);
        let scale = half::f16::from_le_bytes([blob[16], blob[17]]).to_f32();
        assert_eq!(scale, 0.25);

        for index in 0..32 {
            // Burn reads value `j` from bit `j * 4` of the word holding it, and treats the
            // raw nibble as two's complement.
            let word = u32::from_le_bytes([
                blob[index / 8 * 4],
                blob[index / 8 * 4 + 1],
                blob[index / 8 * 4 + 2],
                blob[index / 8 * 4 + 3],
            ]);
            let raw = (word >> ((index % 8) * 4)) & 0x0F;
            let signed = raw as i32 - if raw >= 8 { 16 } else { 0 };
            let value = signed as f32 * scale;
            assert!(
                (value - expected[index]).abs() < 1e-6,
                "at {index}: {value} vs {}",
                expected[index]
            );
        }
    }

    #[test]
    fn should_use_the_metadata_it_was_given() {
        let store = store_from(sample());
        assert_eq!(store.reader().metadata_usize("general.alignment"), None);
        assert_eq!(store.reader().architecture(), Some("llama"));
        assert_eq!(
            store.reader().metadata_value("general.architecture"),
            Some(&MetadataValue::String("llama".into()))
        );
    }
}
