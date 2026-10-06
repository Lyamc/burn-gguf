//! Build a small GGUF file to exercise the examples and the store.
//!
//! ```text
//! cargo run --example make_sample -p burn-gguf
//! cargo run --example gguf_inspect -p burn-gguf -- target/demo.gguf
//! ```

use burn_gguf::{GgmlType, GgufWriter, TensorSpec};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/demo.gguf".to_string());

    let mut writer = GgufWriter::new();
    writer.metadata("general.architecture", "llama");
    writer.metadata("general.name", "gguf-example");
    writer.metadata("general.alignment", 32u32);
    writer.metadata("general.quantization_version", 2u32);
    writer.metadata("llama.block_count", 2u32);
    writer.metadata("llama.embedding_length", 64u32);
    writer.metadata("llama.feed_forward_length", 128u32);
    writer.metadata("llama.attention.head_count", 8u32);

    // A `Q8_0` embedding, which stays packed in Burn.
    let mut embedding = Vec::new();
    for block in 0..(64 * 4 / 32) {
        embedding
            .extend_from_slice(&half::f16::from_f32(0.01 + block as f32 * 0.001).to_le_bytes());
        embedding.extend((0..32).map(|i| ((i + block) % 61) as u8));
    }
    writer.tensor(
        TensorSpec::new("token_embd.weight", vec![64, 4], GgmlType::Q8_0).with_bytes(&embedding),
    );

    // A `Q4_K` layer, which Burn cannot keep packed and will expand. A `Q4_K` block
    // covers 256 values, so the innermost dimension has to be a multiple of that.
    let mut q4k = vec![0u8; 144];
    q4k[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
    q4k[2..4].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
    // Sub-block scales and minima, all one.
    q4k[4..16].fill(1);
    q4k[16..144].fill(0x11);
    writer.tensor(
        TensorSpec::new("blk.0.attn_q.weight", vec![256, 1], GgmlType::Q4K).with_bytes(&q4k),
    );

    // A full-precision bias, the kind that survives quantization.
    writer.tensor(
        TensorSpec::new("blk.0.ffn_gate.bias", vec![128], GgmlType::F32).with_f32(&vec![0.0; 128]),
    );
    writer.tensor(
        TensorSpec::new("output_norm.weight", vec![64], GgmlType::F32).with_f32(&vec![1.0; 64]),
    );
    writer.tensor(TensorSpec::new(
        "blk.1.attn_k.weight",
        vec![256, 2],
        GgmlType::Q4K,
    ));

    std::fs::write(&path, writer.build()).expect("the output path is writable");
    println!("wrote {path}");
}
