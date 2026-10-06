//! Native GGUF loading for Burn.
//!
//! This crate reads [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
//! model files and hands their weights to Burn tensors. It is a loader, not a model
//! implementation: it parses the container and converts the weights, and leaves the
//! decision of what a given tensor *means* to the caller.
//!
//! # What this crate does not do
//!
//! GGUF is not Burn's canonical format, and nothing here changes that. Burn's own formats
//! remain the ones it writes and trains with. The relationship is one-way:
//!
//! ```text
//! Burn model
//!   ├── SafeTensors / burnpack   (canonical serialization)
//!   └── GGUF importer            (this crate)
//! ```
//!
//! A model built in Burn can be converted to GGUF for a converter tool or a debugging
//! path, and a GGUF file can be loaded back in, but GGUF is never what Burn produces on
//! its own.
//!
//! # Layers
//!
//! The three concerns are kept apart, because they change for different reasons.
//!
//! * **The container.** [`GgufReader`] parses the header, the metadata key-value pairs and
//!   the tensor descriptors, and memory-maps the file. Nothing here knows about any model
//!   architecture.
//! * **Architecture mapping.** Deciding that `blk.3.attn_q.weight` is the query
//!   projection of layer three is the job of the model, not the parser. The crate
//!   exposes [`GgufReader::architecture`] and the raw tensor names so that a
//!   `Llama` or `Qwen2` implementation can do the mapping, and [`GgufAdapter`] provides
//!   the renaming hook for the mechanical part.
//! * **Quantization.** [`repack_to_burn`] and [`dequantize`] are where the format is
//!   actually hard. See below.
//!
//! # Quantization
//!
//! Parsing a header is easy. Executing a packed format is not, and the two are separated
//! here on purpose.
//!
//! GGUF stores weights in block-packed formats that differ from each other in bit width,
//! scale precision, and how many levels of scale they carry. Two of them map exactly onto
//! quantization schemes Burn already has:
//!
//! | GGUF type | Burn scheme | native execution |
//! |---|---|---|
//! | `Q8_0` | `Q8F`, blocked by 32, `f16` parameters | yes |
//! | `Q4_0` | `Q4F`, blocked by 32, `PackedU32`, `f16` parameters | yes |
//! | `Q4_1`, `Q5_0`, `Q5_1` | needs a per-block offset | no, dequantized |
//! | `Q2_K`, `Q4_K`, `Q5_K` | needs a per-block offset *and* a two-level scale | no, dequantized |
//! | `Q6_K` | needs a 6-bit value width | no, dequantized |
//! | `Q3_K`, the `IQ*` and `TQ*` families | needs format-specific packing | no, dequantized |
//!
//! The two exact matches are not a coincidence. Burn's `dequantize` kernel turns a 4-bit
//! raw value `n` into `n - 16` when bit 3 is set, which is precisely the `-8` bias
//! `Q4_0` applies after doubling; and `Q8_0`'s `f16` scale per 32 values is what Burn's
//! blocked scheme expects. Converting them is a de-interleave plus, for 4-bit, a nibble
//! shuffle.
//!
//! The `K` formats cannot be expressed, because Burn's `QuantMode` is symmetric only and
//! its `QuantValue` widths are 2, 4 and 8 bits. `Q4_K` needs a per-sub-block offset and a
//! scale that is the product of an `f16` super-block scale and a 6-bit sub-block scale;
//! `Q6_K` needs six-bit values. Loading one of those produces correct numbers, but they
//! arrive as `f32` and the memory advantage is gone. Making them native means widening
//! Burn's quantization vocabulary, which is a change to the core tensor representation
//! rather than to this crate.
//!
//! # The other half of the story
//!
//! Two mechanical details are worth knowing because they are where a loader silently
//! produces wrong numbers.
//!
//! **Dimension order.** GGUF writes dimensions with the fastest-varying one first, the
//! reverse of Burn's row-major layout, so shapes must be reversed. See
//! [`TensorInfo::burn_shape`].
//!
//! **Bit order.** Burn's kernels read sub-byte value `j` from bit `j * 4` of a `u32`.
//! GGUF packs two values per byte with the low nibble first, putting value `j` at bit
//! `((j / 2) * 8) + (j % 2) * 4`. The orderings agree only for `j < 2`, so every
//! sub-byte conversion is a real shuffle.
//!
//! # Loading
//!
//! ```no_run
//! use burn_gguf::{GgufReader, GgufStore};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let reader = GgufReader::open("model-Q4_K_M.gguf")?;
//! println!("{:?} with {} tensors", reader.architecture(), reader.tensors().len());
//!
//! // Reading a tensor by hand.
//! if let Some(info) = reader.tensor("output_norm.weight") {
//!     let bytes = reader.tensor_data_bytes(info)?;
//!     println!("{} bytes", bytes.len());
//! }
//!
//! // Or through a module, the same way SafeTensors is loaded.
//! let mut store = GgufStore::from_reader(reader)?;
//! for (name, path, conversion, dtype) in store.describe_all() {
//!     println!("{name} -> {path} as {:?} ({conversion:?})", dtype);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Feature flags
//!
//! * `std` — filesystem access. Off by default only in `no_std` builds.
//! * `memmap` (default) — memory-map files, so tensor payloads are read without copying.
//! * `converter` — [`converter`], which expands a GGUF file into a SafeTensors file for
//!   comparison against a non-GGUF code path. This is a convenience for validation, not a
//!   loading strategy.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

mod adapter;
mod base;
mod kv;
mod quant;
mod reader;
mod storage;
mod store;
mod tensor;
mod writer;

#[cfg(feature = "converter")]
mod converter;

pub use adapter::{Conversion, GgufAdapter};
pub use base::{
    DEFAULT_ALIGNMENT, GgufError, GgufHeader, GgufVersion, MAGIC, MAX_ARRAY_DEPTH,
    MAX_ARRAY_ELEMENTS, MAX_METADATA_KV, MAX_STRING_LEN, MAX_TENSOR_COUNT, MAX_TENSOR_DIMS,
};
pub use kv::{MetadataEntry, MetadataValue, MetadataValueType};
pub use quant::{dequantize, dequantize_checked, dequantize_is_supported, repack_to_burn};
pub use reader::{
    GgufFileType, GgufReader, KEY_ALIGNMENT, KEY_ARCHITECTURE, KEY_FILE_TYPE,
    KEY_QUANTIZATION_VERSION,
};
pub use store::{GgufStore, GgufStoreError};
pub use tensor::{BurnLayout, GgmlType, K_SCALE_SIZE, LoadStrategy, QK_K, TensorInfo};
pub use writer::{GgufWriter, TensorSpec};

#[cfg(feature = "converter")]
#[cfg_attr(docsrs, doc(cfg(feature = "converter")))]
pub use converter::{
    ConvertError, ConvertedModel, ConvertedTensor, convert, convert_to_safetensors,
    write_safetensors,
};
