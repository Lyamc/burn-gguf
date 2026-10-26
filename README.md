# burn-gguf

Native [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) loader for [Burn](https://github.com/tracel-ai/burn). It parses the container, renames tensors, and loads weights into Burn modules. It is not a model implementation.

This crate was split out of the [Lyamc/burn](https://github.com/Lyamc/burn) fork. It targets Burn **0.22**.

`Q8_0` and `Q4_0` stay packed, using Burn's `Q8F` and `Q4F` schemes with `f16` scales (`ScaleDtype::F16`, one scale per 32 values). The K-quant, IQ, and TQ types are dequantized to `f32`.

```toml
[dependencies]
burn-gguf = "0.1"
```

```rust
use burn_gguf::GgufReader;

let reader = GgufReader::open("model-Q4_K_M.gguf")?;
```

Examples:

```text
cargo run --example gguf_inspect -- model-Q4_K_M.gguf
cargo run --example make_sample
cargo run --example gguf_to_safetensors --features converter -- model.gguf model.safetensors
```

## License

MIT OR Apache-2.0
