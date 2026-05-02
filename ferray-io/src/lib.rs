// ferray-io: NumPy-compatible file I/O (.npy, .npz, memory-mapped, text)
//
//! This crate provides functions for reading and writing N-dimensional arrays
//! in NumPy-compatible formats:
//!
//! - **`.npy`** — single array binary format ([`npy::save`], [`npy::load`], [`npy::load_dynamic`])
//! - **`.npz`** — zip archive of `.npy` files ([`npz::savez`], [`npz::savez_compressed`])
//! - **Text I/O** — delimited text files ([`text::savetxt`], [`text::loadtxt`], [`text::genfromtxt`])
//! - **Memory mapping** — zero-copy file-backed arrays ([`memmap::memmap_readonly`], [`memmap::memmap_mut`])

// I/O code parses byte-level header fields, encodes/decodes shape and
// stride arrays as `usize <-> u64`, and converts text-format columns
// across width-narrowing types as part of the dtype contract.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms
)]

pub mod datasource;
pub mod format;
pub mod memmap;
pub mod npy;
pub mod npz;
pub mod text;

// Re-export the most commonly used items at crate root for convenience.
pub use datasource::{DataSource, DataSourceReader};
pub use format::{
    HeaderData, MemmapMode, descr_to_dtype, header_data_from_array_1_0, read_array, write_array,
};
pub use npy::{NpyElement, load, load_dynamic, save};
pub use npz::{NpzFile, savez, savez_compressed};
pub use text::{
    SaveTxtOptions, fromregex, fromregex_from_file, genfromtxt, loadtxt, loadtxt_1d, savetxt,
    savetxt_1d,
};
