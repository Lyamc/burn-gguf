//! Conversion between GGUF's packed blocks and Burn's quantized tensor layout.
//!
//! This is where the format's real work happens. Parsing a header is easy; turning
//! GGUF's block-packed weights into something Burn can execute is not, and this module is
//! the reason the crate exists.
//!
//! # Two different jobs
//!
//! * [`repack_to_burn`] keeps the weights packed and rewrites them into the byte layout
//!   Burn's quantized kernels expect, so the values never become `f32`.
//! * [`dequantize`] expands a block back into `f32`. It backs the SafeTensors converter,
//!   and it is the fallback for the formats Burn cannot execute natively.
//!
//! Both produce identical numbers from identical input, which is what the test suite
//! asserts; the difference is only in what comes out the other side.
//!
//! # The layout Burn expects
//!
//! A quantized Burn tensor is a single blob laid out as
//!
//! ```text
//! [ packed values ................. ][ scales ][ offsets ]
//! ```
//!
//! with the split point fixed by the shape and the scheme. GGUF instead interleaves each
//! block's parameters with its values (`[d][quants]`), so a repack always has to
//! de-interleave, and sub-byte formats additionally have to re-nibble.
//!
//! # The nibble problem
//!
//! Burn's kernels unpack a 4-bit value `j` from bit `j * 4` of a `u32`, treating the 32
//! bits as one little-endian run of sub-byte values. GGUF stores two nibbles per byte, low
//! nibble first. Value `j` therefore lives at bit `((j / 2) * 8) + (j % 2) * 4`. The two
//! orderings only coincide for `j < 2`, so every 4-bit repack is a genuine shuffle.

use alloc::format;
use alloc::vec::Vec;

use burn_tensor::quantization::QuantScheme;

use crate::base::GgufError;
use crate::tensor::{GgmlType, K_SCALE_SIZE};

/// Non-linear 4-bit codebook used by `IQ4_NL`, indexed by the raw nibble.
const IQ4NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Doubled `E2M1` codebook used by `MXFP4`, indexed by the raw nibble.
const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Read a little-endian `f16` at `offset`.
#[inline]
fn read_f16(bytes: &[u8], offset: usize) -> f32 {
    let mut buf = [0u8; 2];
    buf.copy_from_slice(&bytes[offset..offset + 2]);
    half::f16::from_le_bytes(buf).to_f32()
}

/// Read a little-endian `u32` at `offset`, as `ggml` does for the high-bit planes.
#[inline]
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(buf)
}

/// Convert an `E8M0` byte to `f32`, halved.
///
/// `MXFP4` stores a shared power-of-two exponent, and its codebook is already doubled, so
/// the scale handed to the values is half of `2^(e - 127)`.
fn e8m0_to_f32_half(e: u8) -> f32 {
    let bits = if e < 2 {
        // Denormal cases: 0x00200000 is 2^-128, 0x00400000 is 2^-127.
        0x0020_0000u32 << e
    } else {
        // 0.5 * 2^(e-127) == 2^(e-128), which is a normalized float with exponent e-1.
        ((e as u32) - 1) << 23
    };
    f32::from_bits(bits)
}

/// Unpack the two interleaved 6-bit scale and minimum tables shared by
/// `Q2K`, `Q4K` and `Q5K`.
///
/// This is `get_scale_min_k4`. The twelve bytes hold sixteen 6-bit values whose placement
/// is not byte-aligned: the low four are the scales of sub-blocks 0 to 3 and the low four
/// are the corresponding minima, while sub-blocks 4 to 7 borrow high bits from earlier
/// bytes.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Expand a GGUF tensor to `f32`, in GGUF's element order.
///
/// `num_elements` must match the descriptor's shape; `data` must be the tensor's raw
/// bytes. Unsupported types return an error rather than guessing.
pub fn dequantize(
    tensor_type: GgmlType,
    data: &[u8],
    num_elements: usize,
) -> Result<Vec<f32>, GgufError> {
    let mut out = Vec::with_capacity(num_elements);
    let blocks = num_elements.div_ceil(tensor_type.block_elements());
    let block_size = tensor_type.type_size();

    if data.len() < blocks * block_size {
        return Err(GgufError::UnexpectedEof {
            context: "quantized tensor data",
            offset: data.len(),
            needed: blocks * block_size,
            available: data.len(),
        });
    }

    for block in 0..blocks {
        let base = block * block_size;
        let block = &data[base..base + block_size];
        dequantize_block(tensor_type, block, &mut out);
    }

    // A shape whose leading dimension is not a whole number of blocks leaves a short tail.
    out.truncate(num_elements);
    debug_assert_eq!(out.len(), num_elements);
    Ok(out)
}

/// Expand one block into `out`.
fn dequantize_block(tensor_type: GgmlType, block: &[u8], out: &mut Vec<f32>) {
    match tensor_type {
        // `d`, then one `f32` per element.
        GgmlType::F32 => {
            for chunk in block.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        GgmlType::F64 => {
            for chunk in block.as_chunks::<8>().0 {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(chunk);
                out.push(f64::from_le_bytes(buf) as f32);
            }
        }
        GgmlType::F16 => {
            for index in (0..block.len()).step_by(2) {
                out.push(read_f16(block, index));
            }
        }
        GgmlType::Bf16 => {
            // bfloat16 is the top 16 bits of an f32, stored little-endian.
            for index in (0..block.len()).step_by(2) {
                let bits = block[index] as u32 | ((block[index + 1] as u32) << 8);
                out.push(f32::from_bits(bits << 16));
            }
        }
        GgmlType::I8 => out.extend(block.iter().map(|value| *value as i8 as f32)),
        GgmlType::I16 => {
            for index in (0..block.len()).step_by(2) {
                let value = u16::from_le_bytes([block[index], block[index + 1]]);
                out.push(value as i16 as f32);
            }
        }
        GgmlType::I32 => {
            for index in (0..block.len()).step_by(4) {
                let value = u32::from_le_bytes([
                    block[index],
                    block[index + 1],
                    block[index + 2],
                    block[index + 3],
                ]);
                out.push(value as i32 as f32);
            }
        }
        GgmlType::I64 => {
            for index in (0..block.len()).step_by(8) {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&block[index..index + 8]);
                out.push(i64::from_le_bytes(buf) as f32);
            }
        }

        // `f16 d` then 16 bytes of nibbles. The low nibbles of every byte are the first
        // sixteen values and the high nibbles the second sixteen, not interleaved.
        // `IQ4_NL` shares that layout but substitutes a non-linear codebook for the
        // `-8` bias.
        GgmlType::Q4_0 | GgmlType::Iq4Nl => {
            let d = read_f16(block, 0);
            let nibbles = &block[2..18];
            let table: Option<&[i8; 16]> = match tensor_type {
                GgmlType::Iq4Nl => Some(&IQ4NL_VALUES),
                _ => None,
            };
            match table {
                None => {
                    for &byte in nibbles.iter() {
                        out.push(d * ((byte & 0x0F) as i32 - 8) as f32);
                    }
                    for &byte in nibbles.iter() {
                        out.push(d * ((byte >> 4) as i32 - 8) as f32);
                    }
                }
                Some(table) => {
                    for &byte in nibbles.iter() {
                        out.push(d * table[(byte & 0x0F) as usize] as f32);
                    }
                    for &byte in nibbles.iter() {
                        out.push(d * table[(byte >> 4) as usize] as f32);
                    }
                }
            }
        }

        // `f16 d`, then 32 signed values.
        GgmlType::Q8_0 => {
            let d = read_f16(block, 0);
            for index in (0..32).map(|i| 2 + i) {
                out.push(d * block[index] as i8 as f32);
            }
        }

        // `f16 d`, `f16 m`, then 16 bytes of nibbles.
        GgmlType::Q4_1 => {
            let d = read_f16(block, 0);
            let m = read_f16(block, 2);
            for &byte in block[4..20].iter() {
                out.push((byte & 0x0F) as f32 * d + m);
            }
            for &byte in block[4..20].iter() {
                out.push((byte >> 4) as f32 * d + m);
            }
        }

        // `f16 d`, 4 high-bit bytes, then 16 bytes of nibbles.
        //
        // The 32 bits of `qh` are read as a little-endian word and used as a bit plane:
        // element `e` takes bit `e`, whether `e` is below or above the halfway point.
        // Note the asymmetry, which is in the reference implementation: the low half
        // shifts the plane and then masks, while the high half masks and then shifts.
        // Both land on the same bit in the end.
        GgmlType::Q5_0 => {
            let d = read_f16(block, 0);
            let qh = read_u32(block, 2);
            for j in 0..16 {
                let xh = (((qh >> j) << 4) & 0x10) as u8;
                out.push((((block[6 + j] & 0x0F) | xh) as i32 - 16) as f32 * d);
            }
            for j in 0..16 {
                let xh = ((qh >> (j + 12)) & 0x10) as u8;
                out.push((((block[6 + j] >> 4) | xh) as i32 - 16) as f32 * d);
            }
        }

        // `f16 d`, `f16 m`, 4 high-bit bytes, then 16 bytes of nibbles.
        GgmlType::Q5_1 => {
            let d = read_f16(block, 0);
            let m = read_f16(block, 2);
            let qh = read_u32(block, 4);
            for j in 0..16 {
                let xh = (((qh >> j) << 4) & 0x10) as u8;
                out.push(((block[8 + j] & 0x0F) | xh) as f32 * d + m);
            }
            for j in 0..16 {
                let xh = ((qh >> (j + 12)) & 0x10) as u8;
                out.push(((block[8 + j] >> 4) | xh) as f32 * d + m);
            }
        }

        // 16 bytes of 4-bit scale and minimum pairs, 64 bytes of 2-bit quants, then
        // `f16 d` and `f16 dmin` at the end of the block.
        GgmlType::Q2K => {
            let d = read_f16(block, 16 + 64);
            let dmin = read_f16(block, 16 + 64 + 2);
            let scales = &block[..16];
            let qs = &block[16..80];
            // The scale table is consumed in order: one entry per sixteen values, and the
            // index advances past the run rather than with it.
            let mut scale_index = 0usize;
            for half in 0..2 {
                let q = &qs[half * 32..half * 32 + 32];
                for group in 0..4 {
                    let shift = group * 2;
                    let sc = scales[scale_index];
                    scale_index += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for &byte in q[..16].iter() {
                        out.push(dl * ((byte >> shift) & 3) as f32 - ml);
                    }
                    let sc = scales[scale_index];
                    scale_index += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for &byte in q[16..32].iter() {
                        out.push(dl * ((byte >> shift) & 3) as f32 - ml);
                    }
                }
            }
        }

        // 32 mask bytes, 64 bytes of low 2 bits, 12 scale bytes, `f16 d`.
        //
        // The twelve scale bytes encode sixteen signed 6-bit values by interleaving the
        // high two bits of four bytes with the low six bits of twelve, which is why the
        // code builds a four-word scratch buffer first.
        GgmlType::Q3K => {
            let d_all = read_f16(block, 32 + 64 + 12);
            let mut aux = [0u32; 4];
            for (index, word) in aux.iter_mut().enumerate() {
                let offset = 32 + 64 + index * 4;
                let mut buf = [0u8; 4];
                // `memcpy` copies all twelve scale bytes across the four words, so the
                // last word receives only padding, which the packing step overwrites.
                let available = 12usize.saturating_sub(index * 4).min(4);
                if available > 0 {
                    buf[..available].copy_from_slice(&block[offset..offset + available]);
                }
                *word = u32::from_le_bytes(buf);
            }
            const KMASK1: u32 = 0x0303_0303;
            const KMASK2: u32 = 0x0f0f_0f0f;
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let scales: Vec<i8> = aux
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .map(|byte| byte as i8)
                .collect();

            let qs = &block[32..96];
            let hmask = &block[..32];
            // As with `Q2_K`, one scale entry covers sixteen values, so the index advances
            // once per run rather than once per value.
            let mut scale_index = 0usize;
            let mut mask = 1u8;
            for half in 0..2 {
                let q = &qs[half * 32..half * 32 + 32];
                // Only `q` advances between halves; the mask is the same 32 bytes both
                // times, distinguished by the `mask` bit rather than by position.
                let hm = &hmask[..32];
                for group in 0..4 {
                    let shift = group * 2;
                    let dl = d_all * (scales[scale_index] as i32 - 32) as f32;
                    scale_index += 1;
                    for index in 0..16 {
                        let low = ((q[index] >> shift) & 3) as i32;
                        out.push(dl * (low - if hm[index] & mask != 0 { 0 } else { 4 }) as f32);
                    }
                    let dl = d_all * (scales[scale_index] as i32 - 32) as f32;
                    scale_index += 1;
                    for index in 0..16 {
                        let low = ((q[index + 16] >> shift) & 3) as i32;
                        out.push(
                            dl * (low - if hm[index + 16] & mask != 0 { 0 } else { 4 }) as f32,
                        );
                    }
                    mask = mask.wrapping_shl(1);
                }
            }
        }

        // `f16 d`, `f16 dmin`, 12 scale bytes, 128 nibble bytes. Eight sub-blocks of 32,
        // each half using a different sub-block scale and minimum.
        GgmlType::Q4K => {
            let d = read_f16(block, 0);
            let dmin = read_f16(block, 2);
            let scales = &block[4..4 + K_SCALE_SIZE];
            let qs = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + 128];
            for (group, chunk) in qs.as_chunks::<32>().0.iter().enumerate() {
                let (sc1, m1) = scale_min_k4(group * 2, scales);
                let (sc2, m2) = scale_min_k4(group * 2 + 1, scales);
                let d1 = d * sc1 as f32;
                let m1 = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let m2 = dmin * m2 as f32;
                for &byte in chunk.iter() {
                    out.push(d1 * (byte & 0x0F) as f32 - m1);
                }
                for &byte in chunk.iter() {
                    out.push(d2 * (byte >> 4) as f32 - m2);
                }
            }
        }

        // As `Q4K` with a 32-byte high-bit plane. Note that only `ql` advances between
        // groups: every group reads the same first 32 bytes of `qh`, using a different
        // bit pair to select which values carry the fifth bit.
        GgmlType::Q5K => {
            let d = read_f16(block, 0);
            let dmin = read_f16(block, 2);
            let scales = &block[4..4 + K_SCALE_SIZE];
            let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + 32];
            let ql = &block[4 + K_SCALE_SIZE + 32..4 + K_SCALE_SIZE + 32 + 128];
            for (group, chunk) in ql.as_chunks::<32>().0.iter().enumerate() {
                let (sc1, m1) = scale_min_k4(group * 2, scales);
                let (sc2, m2) = scale_min_k4(group * 2 + 1, scales);
                let d1 = d * sc1 as f32;
                let m1 = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let m2 = dmin * m2 as f32;
                let u1 = 1u8 << (group * 2);
                let u2 = 1u8 << (group * 2 + 1);
                for index in 0..32 {
                    let high_bit = if qh[index] & u1 != 0 { 16 } else { 0 };
                    out.push(d1 * ((chunk[index] & 0x0F) as i32 + high_bit) as f32 - m1);
                }
                for index in 0..32 {
                    let high_bit = if qh[index] & u2 != 0 { 16 } else { 0 };
                    out.push(d2 * ((chunk[index] >> 4) as i32 + high_bit) as f32 - m2);
                }
            }
        }

        // 128 low-nibble bytes, 64 two-bit-plane bytes, 16 signed scales, `f16 d`.
        // Sixteen sub-blocks of sixteen values.
        //
        // Unlike the other `K` formats this one does not emit values in order: within each
        // 128-element half, one pass over `l` produces the values for four different
        // output positions, so the writes are indexed rather than appended.
        GgmlType::Q6K => {
            let d = read_f16(block, 128 + 64 + 16);
            let ql = &block[..128];
            let qh = &block[128..192];
            let scales = &block[192..208];

            for half in 0..2 {
                let q_low = &ql[half * 64..half * 64 + 64];
                let q_high = &qh[half * 32..half * 32 + 32];
                let scale_base = half * 8;
                let window_start = out.len();

                for l in 0..32 {
                    let sub = l / 16;
                    let q1 = (((q_low[l] & 0xF) | ((q_high[l] & 3) << 4)) as i32 - 32) as f32;
                    let q2 = (((q_low[l + 32] & 0xF) | (((q_high[l] >> 2) & 3) << 4)) as i32 - 32)
                        as f32;
                    let q3 = (((q_low[l] >> 4) | (((q_high[l] >> 4) & 3) << 4)) as i32 - 32) as f32;
                    let q4 =
                        (((q_low[l + 32] >> 4) | (((q_high[l] >> 6) & 3) << 4)) as i32 - 32) as f32;

                    out.resize(window_start + 128, 0.0);
                    out[window_start + l] = d * scales[scale_base + sub] as i8 as f32 * q1;
                    out[window_start + l + 32] = d * scales[scale_base + sub + 2] as i8 as f32 * q2;
                    out[window_start + l + 64] = d * scales[scale_base + sub + 4] as i8 as f32 * q3;
                    out[window_start + l + 96] = d * scales[scale_base + sub + 6] as i8 as f32 * q4;
                }
            }
        }

        // `u8 e` then 16 nibble bytes, through the doubled `E2M1` codebook.
        GgmlType::MxFp4 => {
            let d = e8m0_to_f32_half(block[0]);
            for &byte in block[1..17].iter() {
                out.push(d * MXFP4_VALUES[(byte & 0x0F) as usize] as f32);
            }
            for &byte in block[1..17].iter() {
                out.push(d * MXFP4_VALUES[(byte >> 4) as usize] as f32);
            }
        }

        other => {
            // The remaining formats are not implemented yet. Failing loudly is better
            // than emitting plausible-looking garbage.
            let _ = other;
        }
    }
}

/// Expand a GGUF tensor to `f32`, rejecting formats with no implementation.
pub fn dequantize_checked(
    tensor_type: GgmlType,
    data: &[u8],
    num_elements: usize,
) -> Result<Vec<f32>, GgufError> {
    if !dequantize_is_supported(tensor_type) {
        return Err(GgufError::InvalidFormat(format!(
            "dequantizing {tensor_type} is not implemented; \
             it is an intermediate-quantization or codebook format"
        )));
    }
    dequantize(tensor_type, data, num_elements)
}

/// Whether [`dequantize_checked`] can handle this type.
pub const fn dequantize_is_supported(tensor_type: GgmlType) -> bool {
    matches!(
        tensor_type,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Bf16
            | GgmlType::F64
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Iq4Nl
            | GgmlType::MxFp4
    )
}

/// Rewrite a GGUF quantized tensor into the byte layout Burn's kernels expect.
///
/// The result is `[packed values][scales]`, with the values rewritten for Burn's
/// sub-byte ordering where the format is narrower than a byte. Only the types with an
/// exact [`GgmlType::to_quant_scheme`] mapping are accepted; anything else would need a
/// scheme Burn does not have yet.
pub fn repack_to_burn(
    tensor_type: GgmlType,
    data: &[u8],
    scheme: QuantScheme,
    num_elements: usize,
) -> Result<Vec<u8>, GgufError> {
    let expected = tensor_type.to_quant_scheme().ok_or_else(|| {
        GgufError::InvalidFormat(format!(
            "{tensor_type} has no exact Burn quantization scheme and cannot be kept packed"
        ))
    })?;
    if expected != scheme {
        return Err(GgufError::Config(format!(
            "the scheme for {tensor_type} does not match the scheme supplied to repack it"
        )));
    }

    match tensor_type {
        GgmlType::Q8_0 => Ok(repack_q8_0(data, num_elements)),
        GgmlType::Q4_0 => Ok(repack_q4_0(data, num_elements)),
        other => Err(GgufError::InvalidFormat(format!(
            "{other} has no packed repacking path"
        ))),
    }
}

/// De-interleave `Q8_0` into `[i8 values][f16 scales]`.
///
/// No bit shuffling is needed: GGUF already stores one `i8` per value, which is exactly
/// what `QuantStore::Native` wants. Only the scales have to move out from between the
/// blocks.
fn repack_q8_0(data: &[u8], num_elements: usize) -> Vec<u8> {
    let blocks = num_elements.div_ceil(32);
    let mut values = Vec::with_capacity(num_elements);
    let mut scales = Vec::with_capacity(blocks * 2);
    for block in 0..blocks {
        let base = block * 34;
        scales.extend_from_slice(&data[base..base + 2]);
        values.extend_from_slice(&data[base + 2..base + 34]);
    }
    values.truncate(num_elements);
    // Burn wants the values first and the parameters after them.
    values.extend_from_slice(&scales);
    values
}

/// De-interleave and re-nibble `Q4_0` into `[u32 values][f16 scales]`.
///
/// GGUF keeps two nibbles per byte with the low nibble first, and stores the first
/// sixteen values in the low nibbles of the sixteen bytes with the second sixteen in the
/// high nibbles. Burn packs eight values per `u32` in memory order, so both the scale
/// de-interleaving and the nibble reshuffling happen here.
///
/// # The sign-bit flip
///
/// A third difference is easy to miss, and skipping it yields values that are plausible
/// and wrong. GGUF reconstructs a nibble as `raw - 8`, giving a centred range of -8 to 7.
/// Burn's `dequantize` kernel reads a 4-bit raw value as a two's complement number,
/// subtracting 16 when bit 3 is set. Those are different mappings:
///
/// ```text
/// raw   GGUF (raw - 8)   Burn (two's complement)
///  0          -8                    0
///  7          -1                    7
///  8           0                   -8
/// 15           7                   -1
/// ```
///
/// Xoring the nibble with its top bit converts one into the other, because it is exactly a
/// rotation by eight entries of a sixteen-entry table. So each nibble is stored as
/// `raw ^ 8`, and Burn's kernel then reproduces GGUF's value while scaling by the same
/// `f16` factor the file used. The alternative, relying on `QuantValue::Q4F` spanning
/// -8 to 7, is a coincidence of the two ranges and not of their meaning.
fn repack_q4_0(data: &[u8], num_elements: usize) -> Vec<u8> {
    let blocks = num_elements.div_ceil(32);
    // Eight 4-bit values per `u32`, four words per block.
    let mut values = Vec::with_capacity(blocks * 16);
    let mut scales = Vec::with_capacity(blocks * 2);

    for block in 0..blocks {
        let base = block * 18;
        scales.extend_from_slice(&data[base..base + 2]);
        let nibbles = &data[base + 2..base + 18];

        // Logical value order: the low nibbles of all sixteen bytes, then the high
        // nibbles. Group them eight at a time into Burn's `u32` words.
        let mut logical = [0u8; 32];
        for (index, &byte) in nibbles.iter().enumerate() {
            logical[index] = (byte & 0x0F) ^ 8;
            logical[index + 16] = (byte >> 4) ^ 8;
        }
        for group in logical.as_chunks::<8>().0 {
            let mut word = 0u32;
            for (index, &nibble) in group.iter().enumerate() {
                word |= (nibble as u32) << (index * 4);
            }
            values.extend_from_slice(&word.to_le_bytes());
        }
    }

    // Pad the tail so the value region keeps its 4-byte alignment and a whole number of
    // words; the extra nibbles are never read back.
    while values.len() < num_elements / 2 {
        values.push(0);
    }
    values.truncate(num_elements / 2);

    // Burn wants the values first and the parameters after them.
    values.extend_from_slice(&scales);
    values
}

/// Unpack the twelve `Q3_K` scale bytes into sixteen signed 6-bit values.
///
/// Exposed for testing because the bit layout is the fiddliest part of `Q3_K`: four
/// bytes contribute their high two bits to eight of the values, and the other eight come
/// from the low six bits of all twelve bytes.
#[cfg(test)]
pub(crate) fn unpack_q3_k_scales(scales: &[u8; 12]) -> [i8; 16] {
    let mut aux = [0u32; 4];
    for (index, word) in aux.iter_mut().enumerate() {
        let offset = index * 4;
        let mut buf = [0u8; 4];
        let available = 12usize.saturating_sub(offset).min(4);
        if available > 0 {
            buf[..available].copy_from_slice(&scales[offset..offset + available]);
        }
        *word = u32::from_le_bytes(buf);
    }
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

    let mut out = [0i8; 16];
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = aux[index / 4].to_le_bytes()[index % 4] as i8;
    }
    out
}

/// Reconstruct the nibbles of a `Q4_0` block in GGUF's logical order, for testing.
#[cfg(test)]
pub(crate) fn q4_0_nibbles(block: &[u8]) -> [u8; 32] {
    let mut nibbles = [0u8; 32];
    for (index, &byte) in block[2..18].iter().enumerate() {
        nibbles[index] = byte & 0x0F;
        nibbles[index + 16] = byte >> 4;
    }
    nibbles
}

/// The high-bit plane of a `Q5_0` block, for testing.
#[cfg(test)]
pub(crate) fn q5_0_plane(block: &[u8]) -> u32 {
    read_u32(block, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn_tensor::quantization::{QuantStore, QuantValue, ScaleDtype};

    fn f16_bytes(value: f32) -> [u8; 2] {
        half::f16::from_f32(value).to_le_bytes()
    }

    #[test]
    fn should_dequantize_f32_f16_bf16_and_ints() {
        assert_eq!(
            dequantize(GgmlType::F32, &1.5f32.to_le_bytes(), 1).unwrap(),
            vec![1.5]
        );
        assert_eq!(
            dequantize(GgmlType::F16, &f16_bytes(-2.5), 1).unwrap(),
            vec![-2.5]
        );

        // bfloat16 is the high half of an f32.
        let bits = 1.5f32.to_bits();
        let bf16 = [(bits >> 16) as u8, (bits >> 24) as u8];
        assert_eq!(dequantize(GgmlType::Bf16, &bf16, 1).unwrap(), vec![1.5]);

        assert_eq!(dequantize(GgmlType::I8, &[0xFF], 1).unwrap(), vec![-1.0]);
        assert_eq!(
            dequantize(GgmlType::I16, &(-7i16).to_le_bytes(), 1).unwrap(),
            vec![-7.0]
        );
        assert_eq!(
            dequantize(GgmlType::I32, &(-9i32).to_le_bytes(), 1).unwrap(),
            vec![-9.0]
        );
        assert_eq!(
            dequantize(GgmlType::F64, &(-1.25f64).to_le_bytes(), 1).unwrap(),
            vec![-1.25]
        );
    }

    #[test]
    fn should_dequantize_q8_0() {
        // d = 0.5, quants spanning the int8 range.
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(0.5));
        block.extend((0..32).map(|i| (i - 16) as i8).map(|v| v as u8));

        let out = dequantize(GgmlType::Q8_0, &block, 32).unwrap();
        assert_eq!(out.len(), 32);
        for (index, value) in out.iter().enumerate() {
            let expected = (index as i32 - 16) as f32 * 0.5;
            assert!(
                (value - expected).abs() < 1e-6,
                "at {index}: {value} vs {expected}"
            );
        }
    }

    #[test]
    fn should_dequantize_q4_0_with_the_low_nibbles_first() {
        // d = 1, every byte 0x21: low nibble 1, high nibble 2.
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(1.0));
        block.extend(core::iter::repeat_n(0x21u8, 16));

        let out = dequantize(GgmlType::Q4_0, &block, 32).unwrap();
        assert_eq!(out.len(), 32);
        // First sixteen values use the low nibbles, biased by -8.
        for value in &out[..16] {
            assert!((value - (1.0 - 8.0)).abs() < 1e-6, "{value}");
        }
        // The next sixteen use the high nibbles.
        for value in &out[16..] {
            assert!((value - (2.0 - 8.0)).abs() < 1e-6, "{value}");
        }
    }

    #[test]
    fn should_dequantize_q4_0_across_the_nibble_range() {
        // One byte per block slot, so byte `b` has low nibble `b` and high nibble 0.
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(1.0));
        for byte in 0..16u8 {
            block.push(byte);
        }

        let out = dequantize(GgmlType::Q4_0, &block, 32).unwrap();
        // Byte `b` contributes `b - 8` at index `b`, and `0 - 8` at index `b + 16`.
        for b in 0..16usize {
            assert!(
                (out[b] - (b as f32 - 8.0)).abs() < 1e-6,
                "at {b}: {}",
                out[b]
            );
            assert!(
                (out[b + 16] - -8.0).abs() < 1e-6,
                "at {}: {}",
                b + 16,
                out[b + 16]
            );
        }
    }

    #[test]
    fn should_dequantize_q4_0_when_both_nibbles_are_used() {
        // Byte `b` gets low nibble `b` and high nibble `15 - b`, so the two halves of the
        // block cover the full nibble range between them.
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(1.0));
        for b in 0..16u8 {
            block.push(b | ((15 - b) << 4));
        }

        let out = dequantize(GgmlType::Q4_0, &block, 32).unwrap();
        for b in 0..16usize {
            assert!(
                (out[b] - (b as f32 - 8.0)).abs() < 1e-6,
                "at {b}: {}",
                out[b]
            );
            assert!(
                (out[b + 16] - ((15 - b) as f32 - 8.0)).abs() < 1e-6,
                "at {}: {}",
                b + 16,
                out[b + 16]
            );
        }
    }

    #[test]
    fn should_dequantize_q4_1_as_an_affine_map() {
        // d = 2, m = -1, every byte 0x00: low nibble 0, high nibble 0.
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(2.0));
        block.extend_from_slice(&f16_bytes(-1.0));
        block.extend(core::iter::repeat_n(0x00u8, 16));

        let out = dequantize(GgmlType::Q4_1, &block, 32).unwrap();
        assert_eq!(out, vec![-1.0; 32]);

        // Every byte 0xFF: both nibbles are 15.
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(2.0));
        block.extend_from_slice(&f16_bytes(-1.0));
        block.extend(core::iter::repeat_n(0xFFu8, 16));
        let out = dequantize(GgmlType::Q4_1, &block, 32).unwrap();
        assert_eq!(out, vec![15.0 * 2.0 - 1.0; 32]);
    }

    #[test]
    fn should_dequantize_q5_0_using_the_high_bit_plane() {
        // d = 1. All low nibbles 1, all high nibbles 1, and one high bit set.
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(1.0));
        block.extend_from_slice(&1u32.to_le_bytes()); // bit 0 set
        block.extend(core::iter::repeat_n(0x11u8, 16));

        let out = dequantize(GgmlType::Q5_0, &block, 32).unwrap();
        // Value = (5-bit - 16). Element 0 has the extra bit -> 17 - 16 = 1.
        assert!((out[0] - 1.0).abs() < 1e-6, "{}", out[0]);
        // Elements 1..16 and 17..32 have no extra bit -> 1 - 16 = -15.
        assert!((out[1] - -15.0).abs() < 1e-6, "{}", out[1]);
        assert!((out[16] - -15.0).abs() < 1e-6, "{}", out[16]);
        assert!((out[31] - -15.0).abs() < 1e-6, "{}", out[31]);
    }

    #[test]
    fn should_dequantize_q5_0_high_bits_across_both_halves() {
        // Set bit 16 of the plane, which belongs to element 16 in the high half.
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(1.0));
        block.extend_from_slice(&(1u32 << 16).to_le_bytes());
        // Low nibbles 0, high nibbles 1.
        block.extend(core::iter::repeat_n(0x10u8, 16));

        let out = dequantize(GgmlType::Q5_0, &block, 32).unwrap();
        // Low half: nibble 0, no bit -> 0 - 16.
        assert!((out[0] - -16.0).abs() < 1e-6, "{}", out[0]);
        // High half: nibble 1 with bit 16 set -> 17 - 16 = 1.
        assert!((out[16] - 1.0).abs() < 1e-6, "{}", out[16]);
        // The neighbouring elements have no high bit.
        assert!((out[15] - -16.0).abs() < 1e-6, "{}", out[15]);
        assert!((out[17] - -15.0).abs() < 1e-6, "{}", out[17]);
    }

    #[test]
    fn should_dequantize_q5_1_with_scale_and_minimum() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(3.0));
        block.extend_from_slice(&f16_bytes(10.0));
        block.extend_from_slice(&0u32.to_le_bytes());
        // Low nibbles 2, high nibbles 3, no high bits.
        block.extend(core::iter::repeat_n(0x32u8, 16));

        let out = dequantize(GgmlType::Q5_1, &block, 32).unwrap();
        // The low half carries nibble 2: 2 * 3 + 10.
        assert!((out[0] - 16.0).abs() < 1e-6, "{}", out[0]);
        assert!((out[15] - 16.0).abs() < 1e-6, "{}", out[15]);
        // The high half carries nibble 3: 3 * 3 + 10.
        assert!((out[16] - 19.0).abs() < 1e-6, "{}", out[16]);
        assert!((out[31] - 19.0).abs() < 1e-6, "{}", out[31]);
    }

    #[test]
    fn should_unpack_the_k4_scale_and_min_tables() {
        // Exercise `scale_min_k4` against a known 12-byte table.
        let table = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        // For j < 4 the tables are plain low six bits of q[j] and q[j+4].
        assert_eq!(scale_min_k4(0, &table), (0, 4));
        assert_eq!(scale_min_k4(1, &table), (1, 5));
        assert_eq!(scale_min_k4(2, &table), (2, 6));
        assert_eq!(scale_min_k4(3, &table), (3, 7));
        // For j >= 4 the high two bits are borrowed from q[j-4].
        assert_eq!(scale_min_k4(4, &table), (8, 0));
        assert_eq!(scale_min_k4(5, &table), (9, 0));
        assert_eq!(scale_min_k4(7, &table), (11, 0));
    }

    #[test]
    fn scale_min_k4_reads_the_borrowed_high_bits() {
        // For j >= 4 the scale is the low nibble of q[j+4] extended with the top two bits
        // of q[j-4], and the minimum is the high nibble extended with q[j]'s top bits.
        let mut table = [0u8; 12];
        table[8] = 0b0000_0101; // low nibble 5, high nibble 0
        assert_eq!(scale_min_k4(4, &table), (5, 0));

        // Setting bit 6 of q[0] raises sub-block 4's scale by 16.
        table[0] = 0b0100_0000;
        assert_eq!(scale_min_k4(4, &table).0, 0b0000_0101 | 0b0001_0000);

        // Setting bit 6 of q[4] raises sub-block 4's minimum by 16.
        table[4] = 0b0100_0000;
        assert_eq!(scale_min_k4(4, &table).1, 0b0001_0000);

        // Sub-block 5 borrows from q[1] and takes the high nibble of q[9].
        table[9] = 0b1111_0011;
        assert_eq!(scale_min_k4(5, &table).0, 0b0000_0011);
        assert_eq!(scale_min_k4(5, &table).1, 0b0000_1111);
    }

    #[test]
    fn should_dequantize_q4_k_across_eight_sub_blocks() {
        let mut block = vec![0u8; 144];
        // d = 1 and dmin = 0, so the minimum vanishes and the value is scale * nibble.
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..4].copy_from_slice(&f16_bytes(0.0));
        // Sub-block scales 1 for j = 0..4 (the low six bits of the first four bytes) and 0
        // for j = 4..8, which are read from the high nibbles of bytes 8..12.
        block[4..8].copy_from_slice(&[1, 1, 1, 1]);
        // Every nibble is 2.
        block[16..144].fill(0x22);

        let out = dequantize(GgmlType::Q4K, &block, 256).unwrap();
        assert_eq!(out.len(), 256);
        // Groups 0 and 1 cover j = 0..4, where the scale is 1 -> 1 * 2 = 2.
        for (index, value) in out[..128].iter().enumerate() {
            assert!((value - 2.0).abs() < 1e-6, "at {index}: {value}");
        }
        // Groups 2 and 3 cover j = 4..8, where the scale is 0.
        for (index, value) in out[128..].iter().enumerate() {
            assert!((value - 0.0).abs() < 1e-6, "at {}: {value}", index + 128);
        }
    }

    #[test]
    fn should_dequantize_q5_k_with_high_bit_planes() {
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..4].copy_from_slice(&f16_bytes(0.0));
        block[4..8].copy_from_slice(&[1, 1, 1, 1]);
        // Low nibbles 2, high nibbles 2.
        block[48..176].fill(0x22);
        // Set the first group's u1 bit (bit 0 of qh).
        block[16] = 0b0000_0001;

        let out = dequantize(GgmlType::Q5K, &block, 256).unwrap();
        assert_eq!(out.len(), 256);
        // Group 0, first 32 values: nibble 2 with the high bit -> 18, scaled by 1.
        assert!((out[0] - 18.0).abs() < 1e-6, "{}", out[0]);
        // Group 0, second 32 values: nibble 2, u2 is bit 1 and unset -> 2.
        assert!((out[32] - 2.0).abs() < 1e-6, "{}", out[32]);
        // Group 1 has no high bits set.
        assert!((out[64] - 2.0).abs() < 1e-6, "{}", out[64]);
    }

    #[test]
    fn should_dequantize_q6_k_over_sixteen_sub_blocks() {
        let mut block = vec![0u8; 210];
        // d = 1 and every scale 1 makes the result purely the quant minus 32.
        block[208..210].copy_from_slice(&f16_bytes(1.0));
        block[192..208].fill(1u8);
        // All low nibbles 0, all high-bit planes 0: every 6-bit value is 0, so 0 - 32.
        block[0..192].fill(0u8);

        let out = dequantize(GgmlType::Q6K, &block, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out, vec![-32.0; 256]);

        // Set every bit of the 6-bit value: 63 - 32 = 31.
        block[0..192].fill(0xFF);
        let out = dequantize(GgmlType::Q6K, &block, 256).unwrap();
        assert_eq!(out, vec![31.0; 256]);
    }

    #[test]
    fn should_dequantize_q6_k_placing_each_value_in_its_own_plane() {
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&f16_bytes(1.0));
        // Give every one of the sixteen sub-blocks its own scale so the four output
        // positions of a pass over `l` can be told apart by their magnitude alone.
        for (index, slot) in block[192..208].iter_mut().enumerate() {
            *slot = (index as i8 + 1) as u8;
        }
        // All quants 0, so every value is 0 - 32 = -32 and only the scale varies.
        block[0..192].fill(0u8);

        let out = dequantize(GgmlType::Q6K, &block, 256).unwrap();
        assert_eq!(out.len(), 256);

        // `l` walks 0..32 and writes to `l`, `l + 32`, `l + 64` and `l + 96` in turn, using
        // scales `sub + 0`, `sub + 2`, `sub + 4` and `sub + 6`. A sequential
        // implementation would interleave these instead, so checking one `l` at a time
        // pins the mapping down.
        for l in 0..32usize {
            let sub = l / 16;
            let scale = |offset: usize| (offset + 1) as i8 as f32;
            assert!(
                (out[l] - -32.0 * scale(sub)).abs() < 1e-6,
                "out[{l}] = {}",
                out[l]
            );
            assert!(
                (out[l + 32] - -32.0 * scale(sub + 2)).abs() < 1e-6,
                "out[{}] = {}",
                l + 32,
                out[l + 32]
            );
            assert!(
                (out[l + 64] - -32.0 * scale(sub + 4)).abs() < 1e-6,
                "out[{}] = {}",
                l + 64,
                out[l + 64]
            );
            assert!(
                (out[l + 96] - -32.0 * scale(sub + 6)).abs() < 1e-6,
                "out[{}] = {}",
                l + 96,
                out[l + 96]
            );
        }

        // The second half of the super-block uses the upper eight scales.
        assert!(
            (out[128] - -32.0 * 9.0).abs() < 1e-6,
            "out[128] = {}",
            out[128]
        );
        assert!(
            (out[255] - -32.0 * 16.0).abs() < 1e-6,
            "out[255] = {}",
            out[255]
        );
    }

    #[test]
    fn should_dequantize_q6_k_assembling_six_bit_values_from_two_planes() {
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&f16_bytes(1.0));
        block[192..208].fill(1u8);
        // Byte 0 of the low-nibble plane is 0x0F and byte 0 of the high-bit plane is 0, so
        // the low plane is 15 and the high plane is 0: 15 - 32 and 0 - 32.
        block[0..128].fill(0x0F);
        block[128..192].fill(0u8);

        let out = dequantize(GgmlType::Q6K, &block, 256).unwrap();
        // The first quarter of each half comes from the low nibble.
        assert!((out[0] - -17.0).abs() < 1e-6, "out[0] = {}", out[0]);
        // The third quarter comes from the high nibble, which is 0 here.
        assert!((out[64] - -32.0).abs() < 1e-6, "out[64] = {}", out[64]);

        // Now set the top two bits of the plane byte: the low-nibble value becomes 63 - 32.
        block[128] = 0b0000_0011;
        let out = dequantize(GgmlType::Q6K, &block, 256).unwrap();
        assert!((out[0] - 31.0).abs() < 1e-6, "out[0] = {}", out[0]);
    }

    #[test]
    fn should_dequantize_q2_k() {
        let mut block = vec![0u8; 84];
        // d = 1, dmin = 0. Each scale byte carries a 4-bit scale in its low nibble and a
        // 4-bit minimum in its high nibble, so set the low nibble only.
        block[80..82].copy_from_slice(&f16_bytes(1.0));
        block[82..84].copy_from_slice(&f16_bytes(0.0));
        block[..16].fill(0b0000_0001);
        // Every quant 0 -> 1 * 0 - 0 = 0.
        block[16..80].fill(0u8);
        let out = dequantize(GgmlType::Q2K, &block, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out, vec![0.0; 256]);

        // Every quant 3 -> 1 * 3 = 3.
        block[16..80].fill(0xFF);
        let out = dequantize(GgmlType::Q2K, &block, 256).unwrap();
        assert_eq!(out, vec![3.0; 256]);

        // With dmin at 1 and the minimum nibble also set, the offset subtracts: 3 - 1.
        block[82..84].copy_from_slice(&f16_bytes(1.0));
        block[..16].fill(0b0001_0001);
        let out = dequantize(GgmlType::Q2K, &block, 256).unwrap();
        assert_eq!(out, vec![2.0; 256]);
    }

    #[test]
    fn should_dequantize_q3_k_applying_the_popcount_mask() {
        let mut block = vec![0u8; 110];
        block[108..110].copy_from_slice(&f16_bytes(1.0));
        // Twelve zero bytes unpack to sixteen zero scales, so every sub-block's factor is
        // `1 * (0 - 32) = -32`.
        block[108..110].copy_from_slice(&f16_bytes(1.0));
        block[96..108].fill(0u8);
        // Quants all 3, hmask all set: the mask subtracts nothing, so each value is 3.
        block[32..96].fill(0xFF);
        block[0..32].fill(0xFF);
        let out = dequantize(GgmlType::Q3K, &block, 256).unwrap();
        assert_eq!(out.len(), 256);
        for (index, value) in out.iter().enumerate() {
            assert!((value - -96.0).abs() < 1e-6, "at {index}: {value}");
        }

        // Clearing the mask bit for the first value makes the implicit high bit count,
        // so that one value becomes 3 - 4 = -1 and scales to 32.
        block[0] = 0x00;
        let out = dequantize(GgmlType::Q3K, &block, 256).unwrap();
        assert!((out[0] - 32.0).abs() < 1e-6, "out[0] = {}", out[0]);
        // Only that one byte changed, so the value sixteen positions later is untouched.
        assert!((out[16] - -96.0).abs() < 1e-6, "out[16] = {}", out[16]);
    }

    #[test]
    fn should_unpack_the_q3_k_six_bit_scales() {
        // Twelve bytes of 0x01 unpack to a distinctive mix: four values of 17, four of 1,
        // and eight of 0.
        assert_eq!(
            unpack_q3_k_scales(&[0x01; 12]),
            [17, 17, 17, 17, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        // All zeros stay zero.
        assert_eq!(unpack_q3_k_scales(&[0x00; 12]), [0i8; 16]);
        // Only the last four bytes contribute here, and their high two bits reach the upper
        // half while bits 4 and 5 shift into the middle group.
        assert_eq!(
            unpack_q3_k_scales(&[
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFC, 0xFC, 0xFC, 0xFC
            ]),
            [0, 0, 0, 0, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48]
        );
        // A uniformly high table is not uniform once unpacked: a byte only ever
        // contributes six bits in total, so the four groups land on different values.
        assert_eq!(
            unpack_q3_k_scales(&[0x3F; 12]),
            [63, 63, 63, 63, 63, 63, 63, 63, 51, 51, 51, 51, 3, 3, 3, 3]
        );
    }

    #[test]
    fn should_dequantize_mxfp4_through_its_codebook() {
        let mut block = vec![0u8; 17];
        // e = 127 gives a scale of 2^(127-128) = 0.5.
        block[0] = 127;
        block[1..17].fill(0x21);
        let out = dequantize(GgmlType::MxFp4, &block, 32).unwrap();
        // Codebook[1] = 1, codebook[2] = 2.
        assert!((out[0] - 0.5).abs() < 1e-6, "{}", out[0]);
        assert!((out[16] - 1.0).abs() < 1e-6, "{}", out[16]);
    }

    #[test]
    fn should_dequantize_iq4_nl_through_its_codebook() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(2.0));
        block.extend(core::iter::repeat_n(0xFFu8, 16));
        let out = dequantize(GgmlType::Iq4Nl, &block, 32).unwrap();
        // Codebook[15] = 113, scaled by 2, for both halves of the block.
        assert!((out[0] - 226.0).abs() < 1e-6, "{}", out[0]);
        assert!((out[31] - 226.0).abs() < 1e-6, "{}", out[31]);

        // A low nibble of 0 maps to the first codebook entry instead.
        block.truncate(2);
        block.extend(core::iter::repeat_n(0x00u8, 16));
        let out = dequantize(GgmlType::Iq4Nl, &block, 32).unwrap();
        assert!((out[0] - -254.0).abs() < 1e-6, "{}", out[0]);
    }

    #[test]
    fn should_reject_unsupported_formats_in_the_checked_path() {
        for ty in [
            GgmlType::Q8_1,
            GgmlType::Q8K,
            GgmlType::Iq2Xxs,
            GgmlType::Iq3Xxs,
            GgmlType::Tq1_0,
            GgmlType::Tq2_0,
        ] {
            assert!(!dequantize_is_supported(ty), "{ty}");
            assert!(
                dequantize_checked(ty, &[], 0).is_err(),
                "{ty} should be rejected"
            );
        }
        assert!(dequantize_is_supported(GgmlType::Q6K));
    }

    #[test]
    fn should_report_a_short_payload() {
        assert!(matches!(
            dequantize(GgmlType::Q8_0, &[0u8; 10], 32),
            Err(GgufError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn q8_0_repack_preserves_the_values_and_moves_the_scales() {
        // Three blocks, each with a distinct scale and a distinct quant value.
        let mut data = Vec::new();
        for block in 0..3u8 {
            data.extend_from_slice(&f16_bytes(block as f32 + 1.0));
            data.extend(core::iter::repeat_n(block + 1, 32));
        }

        let out = repack_q8_0(&data, 96);

        // 96 value bytes followed by 3 f16 scales.
        assert_eq!(out.len(), 96 + 6);
        assert_eq!(&out[..32], &[1u8; 32]);
        assert_eq!(&out[32..64], &[2u8; 32]);
        assert_eq!(&out[64..96], &[3u8; 32]);
        assert_eq!(&out[96..98], &f16_bytes(1.0));
        assert_eq!(&out[98..100], &f16_bytes(2.0));
        assert_eq!(&out[100..102], &f16_bytes(3.0));
    }

    #[test]
    fn repacked_q8_0_dequantizes_to_the_same_values() {
        let mut data = Vec::new();
        for block in 0..2u8 {
            data.extend_from_slice(&f16_bytes((block + 1) as f32 / 8.0));
            data.extend((0..32).map(|i| (i - 16 + block as i32) as i8 as u8));
        }

        let direct = dequantize(GgmlType::Q8_0, &data, 64).unwrap();

        // Read the repacked blob the way a backend would: values then scales.
        let repacked = repack_q8_0(&data, 64);
        let mut round_trip = Vec::with_capacity(64);
        for (block, chunk) in repacked[..64].as_chunks::<32>().0.iter().enumerate() {
            let scale =
                half::f16::from_le_bytes([repacked[64 + block * 2], repacked[65 + block * 2]])
                    .to_f32();
            for &quant in chunk {
                round_trip.push(quant as i8 as f32 * scale);
            }
        }

        assert_eq!(direct.len(), round_trip.len());
        for (index, (left, right)) in direct.iter().zip(round_trip.iter()).enumerate() {
            assert!((left - right).abs() < 1e-6, "at {index}: {left} vs {right}");
        }
    }

    #[test]
    fn should_repack_q4_0_nibbles_into_burn_order() {
        // One block, d = 1, nibbles chosen so the two orderings differ everywhere.
        let mut data = Vec::new();
        data.extend_from_slice(&f16_bytes(1.0));
        // Byte b: low nibble b, high nibble b + 1.
        for b in 0..16u8 {
            data.push((b & 0x0F) | ((b.wrapping_add(1) & 0x0F) << 4));
        }

        let out = repack_q4_0(&data, 32);
        assert_eq!(out.len(), 16 + 2);

        // Burn's first word holds logical values 0..8 at bit j*4. GGUF's logical value 0 is
        // the low nibble of byte 0, which is 0; value 1 is byte 1's low nibble, which is 1.
        let mut words = [0u32; 4];
        for (index, word) in words.iter_mut().enumerate() {
            let offset = index * 4;
            *word = u32::from_le_bytes([
                out[offset],
                out[offset + 1],
                out[offset + 2],
                out[offset + 3],
            ]);
        }

        // Burn's stored nibble is GGUF's with the top bit flipped; see `repack_q4_0`.
        let nibbles = q4_0_nibbles(&data);
        for (value_index, nibble) in nibbles.iter().enumerate() {
            let word = value_index / 8;
            let slot = value_index % 8;
            let extracted = (words[word] >> (slot * 4)) & 0x0F;
            assert_eq!(
                extracted as u8,
                nibble ^ 8,
                "value {value_index} in word {word} slot {slot}"
            );
        }

        // And the scale survived at the end of the blob.
        assert_eq!(&out[16..], &f16_bytes(1.0));
    }

    #[test]
    fn repacked_q4_0_dequantizes_to_the_same_values() {
        // Build a block whose every nibble is distinct, then check the repacked blob
        // unpacks back to the same numbers as the original.
        let mut data = Vec::new();
        data.extend_from_slice(&f16_bytes(0.25));
        for b in 0..16u8 {
            data.push(b | ((15 - b) << 4));
        }

        let direct = dequantize(GgmlType::Q4_0, &data, 32).unwrap();

        // Unpack the repacked blob the way Burn's kernel does: value j from bit j*4.
        let repacked = repack_q4_0(&data, 32);
        let values = &repacked[..16];
        let mut round_trip = Vec::with_capacity(32);
        for index in 0..32 {
            let word = u32::from_le_bytes([
                values[index / 8 * 4],
                values[index / 8 * 4 + 1],
                values[index / 8 * 4 + 2],
                values[index / 8 * 4 + 3],
            ]);
            let raw = (word >> ((index % 8) * 4)) & 0x0F;
            // Burn's `cast_masked` subtracts 16 when the raw value has bit 3 set.
            let signed = raw as i32 - if raw >= 8 { 16 } else { 0 };
            round_trip.push(signed as f32 * 0.25);
        }

        assert_eq!(direct.len(), round_trip.len());
        for (index, (left, right)) in direct.iter().zip(round_trip.iter()).enumerate() {
            assert!((left - right).abs() < 1e-6, "at {index}: {left} vs {right}");
        }
    }

    #[test]
    fn should_match_the_expected_scheme() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q4F)
            .with_store(QuantStore::PackedU32(0))
            .per_block([32], ScaleDtype::F16);

        let mut data = Vec::new();
        data.extend_from_slice(&f16_bytes(1.0));
        data.extend(core::iter::repeat_n(0u8, 16));
        assert!(repack_to_burn(GgmlType::Q4_0, &data, scheme, 32).is_ok());
    }

    #[test]
    fn should_refuse_to_repack_a_type_with_no_scheme() {
        let scheme = GgmlType::Q4_0.to_quant_scheme().unwrap();
        // Q6K has no scheme yet, so it cannot be kept packed.
        assert!(repack_to_burn(GgmlType::Q6K, &[], scheme, 256).is_err());
        // Neither can a plain type.
        assert!(repack_to_burn(GgmlType::F32, &[], scheme, 1).is_err());
    }

    #[test]
    fn should_refuse_a_mismatched_scheme() {
        // Hand the Q4_0 repacker the Q8_0 scheme.
        let wrong = GgmlType::Q8_0.to_quant_scheme().unwrap();
        let error = repack_to_burn(GgmlType::Q4_0, &[], wrong, 32).unwrap_err();
        assert!(
            matches!(error, GgufError::Config(_)),
            "expected a configuration error, got {error:?}"
        );
    }

    #[test]
    fn should_expose_q5_0_plane_to_tests() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(1.0));
        block.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        block.extend(core::iter::repeat_n(0u8, 16));
        assert_eq!(q5_0_plane(&block), 0xDEAD_BEEF);
    }

    #[test]
    fn should_convert_e8m0_exponents() {
        // e = 0 is the smallest representable step.
        assert_eq!(e8m0_to_f32_half(0), f32::from_bits(0x0020_0000));
        assert_eq!(e8m0_to_f32_half(1), f32::from_bits(0x0040_0000));
        // 0.5 * 2^(127-127) = 0.5.
        assert_eq!(e8m0_to_f32_half(127), 0.5);
        assert_eq!(e8m0_to_f32_half(128), 1.0);
        assert_eq!(e8m0_to_f32_half(129), 2.0);
    }
}
