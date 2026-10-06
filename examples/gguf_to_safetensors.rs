//! Expand a GGUF file into a SafeTensors file.
//!
//! This is a validation and compatibility path, not a loading strategy: GGUF keeps the
//! weights packed, which is the reason to use it, and the output here does not. The point
//! is to produce something the rest of a toolchain can read so a GGUF code path can be
//! checked against a plain float one.
//!
//! ```text
//! cargo run --example gguf_to_safetensors -p burn-gguf --features converter -- \
//!     model-Q4_K_M.gguf model.safetensors
//! ```

use burn_gguf::{GgufReader, convert_to_safetensors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(source) = args.next() else {
        eprintln!("usage: gguf_to_safetensors <input.gguf> [output.safetensors]");
        eprintln!();
        eprintln!("The output defaults to the input path with a .safetensors extension.");
        std::process::exit(2);
    };
    let destination = args.next().unwrap_or_else(|| {
        std::path::Path::new(&source)
            .with_extension("safetensors")
            .to_string_lossy()
            .into_owned()
    });

    let reader = GgufReader::open(&source)?;
    let types: Vec<String> = reader
        .dominant_type()
        .map(|ty| ty.name().to_string())
        .into_iter()
        .collect();

    let model = convert_to_safetensors(&reader, &destination)?;

    println!("read     {source}");
    println!("wrote    {destination}");
    println!(
        "types    {}",
        if types.is_empty() {
            "none".to_string()
        } else {
            types.join(", ")
        }
    );
    println!("tensors  {}", model.tensors.len());
    println!("values   {}", model.num_elements());
    for ty in model.expanded_types() {
        println!("  expanded {ty}");
    }

    Ok(())
}
