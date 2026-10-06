//! Print what a GGUF file contains.
//!
//! ```text
//! cargo run --example gguf_inspect -p burn-gguf -- model-Q4_K_M.gguf
//! ```

use std::collections::BTreeMap;

use burn_gguf::{Conversion, GgmlType, GgufReader, GgufStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: gguf_inspect <file.gguf>");
        std::process::exit(2);
    };

    let reader = GgufReader::open(&path)?;

    let payload: usize = reader
        .tensors()
        .iter()
        .filter_map(|info| info.nbytes().ok())
        .sum();

    println!("file           {path}");
    println!("version        {}", reader.version().as_u32());
    println!("alignment      {}", reader.alignment());
    println!(
        "data section   {payload} bytes, starting at offset {}",
        reader.data_offset()
    );
    println!(
        "payloads       {}",
        if reader.is_zero_copy() {
            "read in place from the mapping"
        } else {
            "copied out of the file"
        }
    );
    println!();

    println!("metadata ({} entries)", reader.metadata().len());
    for entry in reader.metadata() {
        // The tokenizer vocabulary can be enormous, so long values are elided.
        let value = entry.value.display_summary();
        let value = if value.chars().count() > 72 {
            let mut truncated: String = value.chars().take(69).collect();
            truncated.push_str("...");
            truncated
        } else {
            value
        };
        println!("  {:<48} {}", entry.key, value);
    }
    println!();

    if let Some(ty) = reader.dominant_type() {
        println!("dominant type  {ty}");
        if let Some(gap) = ty.native_quant_gap() {
            println!("                {gap}");
        }
    }

    let store = GgufStore::from_reader(reader)?;
    let described = store.describe_all();

    // Group by how each tensor will be loaded, so a model with thousands of tensors
    // produces a short summary rather than a wall of text.
    let mut summary: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for (name, _, conversion, _) in &described {
        let nbytes = store
            .reader()
            .tensor(name)
            .and_then(|info| info.nbytes().ok())
            .unwrap_or(0);
        let key = match conversion {
            Conversion::NativeQuantized => "kept packed",
            Conversion::Dequantized(_) => "expanded to f32",
        };
        let entry = summary.entry(key).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += nbytes;
    }

    println!();
    println!("tensors ({} total)", store.len());
    for (kind, (count, bytes)) in summary {
        println!("  {count:>6} {kind:<20} {bytes:>12} bytes on disk");
    }

    println!();
    println!("first 20 tensors, with the path a Burn module would look them up under:");
    for (name, path, conversion, dtype) in described.iter().take(20) {
        println!("  {name}");
        println!("    -> {path} as {dtype:?} ({conversion:?})");
    }
    if store.len() > 20 {
        println!("  ... and {} more", store.len() - 20);
    }

    // A short note on the formats this build can execute without expanding.
    let packed: Vec<GgmlType> = GgmlType::ALL
        .iter()
        .filter(|ty| ty.to_quant_scheme().is_some())
        .copied()
        .collect();
    println!();
    println!(
        "native quantized execution is available for {}",
        packed
            .iter()
            .map(|ty| ty.name())
            .collect::<Vec<_>>()
            .join(", ")
    );

    Ok(())
}
