//! End-to-end loading of a synthetic GGUF file into real Burn modules.
//!
//! These tests exist because the interesting failures are integration failures: a file
//! that parses perfectly and a tensor that dequantizes perfectly can still end up in the
//! wrong parameter, under a name the module never visits, with a shape the module rejects.

use burn::module::{Module, Param};
use burn::tensor::{Device, Tensor, TensorData, Tolerance};
use burn_gguf::{GgmlType, GgufAdapter, GgufStore, GgufWriter, TensorSpec};
use burn_nn::{Embedding, EmbeddingConfig};

/// Hidden size, which has to be a multiple of 32 for the quantized formats.
const D_IN: usize = 32;
/// Vocabulary size.
const D_OUT: usize = 6;

/// A model whose field names match what [`GgufAdapter`] produces.
///
/// GGUF calls the embedding `token_embd` and the final norm `output_norm`. The first maps
/// onto this field's `weight` without help; the second needs a rule because a Burn norm
/// parameter is named `gamma` and a GGUF one is named `weight`.
#[derive(Module, Debug)]
struct Tiny {
    embedding: Embedding,
    norm: Param<Tensor<1>>,
    /// Never present in the file, so the loader's handling of a missing parameter can be
    /// checked without a second fixture.
    extra: Param<Tensor<1>>,
}

impl Tiny {
    fn init(device: &Device) -> Self {
        Self {
            // GGUF stores `token_embd.weight` as `[n_embd, n_vocab]`, which reverses into
            // `[n_vocab, n_embd]`, the layout an embedding weight already uses.
            embedding: EmbeddingConfig::new(D_OUT, D_IN).init(device),
            norm: Param::from_tensor(Tensor::ones([D_IN], device)),
            extra: Param::from_tensor(Tensor::ones([D_IN], device)),
        }
    }
}

/// The adapter that maps this model's GGUF names.
fn adapter() -> GgufAdapter {
    GgufAdapter::new()
        .with_remap(r"^norm\.weight$", "norm")
        .expect("the pattern is valid")
}
/// The quant pattern each block uses, so the expected values can be recomputed here
/// rather than taken from the crate under test.
fn quant_at(index: usize) -> i8 {
    ((index % 31) as i8) - 15
}

fn q8_0_block(scale: f32, first: usize) -> Vec<u8> {
    let mut block = Vec::new();
    block.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
    block.extend((0..32).map(|i| quant_at(first + i) as u8));
    block
}

/// Write a file whose tensors are named the way GGUF names them.
fn write_gguf(path: &std::path::Path) {
    let mut writer = GgufWriter::new();
    writer.metadata("general.architecture", "llama");
    writer.metadata("general.alignment", 32u32);
    writer.metadata("llama.block_count", 1u32);
    writer.metadata("llama.embedding_length", D_IN as u32);

    // Six `Q8_0` blocks: 32 by 6, which reverses to the embedding's `[6, 32]`.
    let mut embedding = Vec::new();
    for block in 0..6usize {
        // A different scale per block, so a misordered de-interleave cannot pass.
        embedding.extend(q8_0_block(0.5 + block as f32 * 0.125, block * 32));
    }
    writer.tensor(
        TensorSpec::new("token_embd.weight", vec![D_IN, D_OUT], GgmlType::Q8_0)
            .with_bytes(&embedding),
    );

    // The norm stays in F32, so one load exercises both dtypes.
    let norm: Vec<f32> = (0..D_IN)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    writer.tensor(TensorSpec::new("output_norm.weight", vec![D_IN], GgmlType::F32).with_f32(&norm));

    // A tensor the module does not visit, to check it is reported rather than fatal.
    writer.tensor(
        TensorSpec::new("blk.0.attn_q.weight", vec![D_IN], GgmlType::F32).with_f32(&[0.0; D_IN]),
    );

    std::fs::write(path, writer.build()).expect("the temporary path is writable");
}

/// The embedding values the file describes, recomputed independently.
fn expected_embedding() -> Vec<f32> {
    let mut expected = Vec::with_capacity(D_IN * D_OUT);
    for block in 0..6usize {
        let scale = 0.5 + block as f32 * 0.125;
        for i in 0..32usize {
            expected.push(quant_at(block * 32 + i) as f32 * scale);
        }
    }
    expected
}

fn load(path: &std::path::Path) -> GgufStore {
    GgufStore::from_file(path)
        .expect("the file loads")
        .with_adapter(adapter())
}

#[test]
fn should_load_a_gguf_file_into_a_module() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    write_gguf(&path);

    let device = Default::default();
    let mut model: Tiny = Tiny::init(&device);
    let before = model.embedding.weight.val().into_data();

    let mut store = load(&path);
    assert_eq!(store.architecture(), Some("llama"));
    assert_eq!(
        store.reader().metadata_usize("llama.embedding_length"),
        Some(D_IN)
    );

    let result = store
        .load_into(&mut model)
        .expect("applying the store succeeds");

    assert!(
        result.applied.contains(&"embedding.weight".to_string()),
        "{:?}",
        result.applied
    );
    assert!(
        result.applied.contains(&"norm".to_string()),
        "{:?}",
        result.applied
    );

    // The weights really changed.
    assert!(!before.eq(&model.embedding.weight.val().into_data()));
}

#[test]
fn should_load_a_quantized_weight_with_the_right_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    write_gguf(&path);

    let device = Default::default();
    let mut model: Tiny = Tiny::init(&device);
    load(&path)
        .load_into(&mut model)
        .expect("applying the store succeeds");

    let weight = model.embedding.weight.val();
    // GGUF `[32, 6]` reversed is the embedding's `[6, 32]`.
    assert_eq!(weight.dims(), [D_OUT, D_IN]);

    // The weight arrived packed, so comparing it to floats means asking the backend to
    // unpack it. This is the step that would go wrong if the scales were sliced at the
    // wrong offset, and it is the same call `Linear` and `Embedding` make internally.
    let expected = Tensor::<2>::from_data(
        TensorData::new(expected_embedding(), [D_OUT, D_IN]),
        &device,
    );
    weight
        .dequantize()
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
}

#[test]
fn should_load_an_unquantized_weight_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    write_gguf(&path);

    let device = Default::default();
    let mut model: Tiny = Tiny::init(&device);
    load(&path)
        .load_into(&mut model)
        .expect("applying the store succeeds");

    let expected: Vec<f32> = (0..D_IN)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    model
        .norm
        .val()
        .into_data()
        .assert_approx_eq::<f32>(&TensorData::new(expected, [D_IN]), Tolerance::default());
}

#[test]
fn should_report_a_tensor_the_module_does_not_use() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    write_gguf(&path);

    let device = Default::default();
    let mut model: Tiny = Tiny::init(&device);
    let result = load(&path)
        .load_into(&mut model)
        .expect("applying the store succeeds");

    // The extra layer is not an error by default, but it is reported so a caller can see
    // that the file holds more than the model asked for. It has still been renamed onto
    // the Burn vocabulary, which is what a caller would print to work out what is there.
    assert!(
        result
            .unused
            .contains(&"layers.0.self_attn.q_proj.weight".to_string()),
        "{:?}",
        result.unused
    );
}

#[test]
fn should_report_a_parameter_the_file_does_not_contain() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    write_gguf(&path);

    let device = Default::default();
    let mut model: Tiny = Tiny::init(&device);
    let result = load(&path)
        .load_into(&mut model)
        .expect("a missing parameter is not fatal by default");

    assert!(
        result.missing.iter().any(|(name, _)| name == "extra"),
        "{:?}",
        result.missing
    );
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn should_fail_when_every_parameter_is_required_and_one_is_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    write_gguf(&path);

    let device = Default::default();
    let mut model: Tiny = Tiny::init(&device);
    let error = load(&path)
        .require_complete()
        .load_into(&mut model)
        .expect_err("a complete load rejects a file that is missing a parameter");
    let message = format!("{error}");
    assert!(message.contains("missing"), "{message}");
    assert!(message.contains("extra"), "{message}");
}

#[test]
fn should_only_touch_the_paths_a_filter_selects() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    write_gguf(&path);

    let device = Default::default();
    let mut model: Tiny = Tiny::init(&device);
    let before = model.norm.val().into_data();

    let mut store =
        load(&path).with_filter(burn_store::PathFilter::new().with_full_path("embedding.weight"));
    let result = store
        .load_into(&mut model)
        .expect("a filtered load succeeds");

    assert_eq!(result.applied, vec!["embedding.weight".to_string()]);
    // The norm was not visited, so it still holds whatever it was initialized to.
    assert!(before.eq(&model.norm.val().into_data()));
}

#[test]
fn should_expand_a_format_burn_cannot_keep_packed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4k.gguf");
    let mut writer = GgufWriter::new();
    writer.metadata("general.architecture", "llama");
    writer.tensor(TensorSpec::new(
        "token_embd.weight",
        vec![256, D_OUT],
        GgmlType::Q4K,
    ));
    writer.tensor(
        TensorSpec::new("output_norm.weight", vec![256], GgmlType::F32).with_f32(&vec![0.0; 256]),
    );
    std::fs::write(&path, writer.build()).unwrap();

    // `Q4_K` needs an embedding of 256 inputs to fill a super-block.
    let device = Default::default();
    let mut model: Tiny = Tiny {
        embedding: EmbeddingConfig::new(D_OUT, 256).init(&device),
        norm: Param::from_tensor(Tensor::ones([256], &device)),
        extra: Param::from_tensor(Tensor::ones([256], &device)),
    };

    load(&path)
        .load_into(&mut model)
        .expect("an expanded format still loads");

    // The shape was still honoured, which is the part that can silently go wrong.
    assert_eq!(model.embedding.weight.val().dims(), [D_OUT, 256]);
}

#[test]
fn should_refuse_an_expanded_format_in_strict_mode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4k.gguf");
    let mut writer = GgufWriter::new();
    writer.tensor(TensorSpec::new(
        "token_embd.weight",
        vec![256, D_OUT],
        GgmlType::Q4K,
    ));
    std::fs::write(&path, writer.build()).unwrap();

    let device = Default::default();
    let mut model: Tiny = Tiny {
        embedding: EmbeddingConfig::new(D_OUT, 256).init(&device),
        norm: Param::from_tensor(Tensor::ones([256], &device)),
        extra: Param::from_tensor(Tensor::ones([256], &device)),
    };

    let error = GgufStore::from_file(&path)
        .expect("the file loads")
        .with_adapter(adapter())
        .strict()
        .load_into(&mut model)
        .expect_err("strict mode rejects a format it cannot keep packed");
    let message = format!("{error}");
    assert!(message.contains("Q4_K"), "{message}");
    // The message should say what is missing, not just that something went wrong.
    assert!(message.contains("offset"), "{message}");
}

#[test]
fn should_reject_a_shape_that_cannot_fill_a_block() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.gguf");
    let mut writer = GgufWriter::new();
    // 30 inputs is not a multiple of the 32 values in a Q8_0 block.
    writer.tensor(TensorSpec::new(
        "token_embd.weight",
        vec![30, D_OUT],
        GgmlType::Q8_0,
    ));
    std::fs::write(&path, writer.build()).unwrap();

    let error = GgufStore::from_file(&path).expect_err("the block layout check rejects this");
    let message = format!("{error}");
    assert!(message.contains("innermost"), "{message}");
    assert!(message.contains("32"), "{message}");
}
