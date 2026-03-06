// ferrum-io: NumPy-compatible file I/O (.npy, .npz, memory-mapped, text)
//
//! This crate provides functions for reading and writing N-dimensional arrays
//! in NumPy-compatible formats:
//!
//! - **`.npy`** — single array binary format ([`npy::save`], [`npy::load`], [`npy::load_dynamic`])
//! - **`.npz`** — zip archive of `.npy` files ([`npz::savez`], [`npz::savez_compressed`])
//! - **Text I/O** — delimited text files ([`text::savetxt`], [`text::loadtxt`], [`text::genfromtxt`])
//! - **Memory mapping** — zero-copy file-backed arrays ([`memmap::memmap_readonly`], [`memmap::memmap_mut`])

pub mod format;
pub mod memmap;
pub mod npy;
pub mod npz;
pub mod text;

// Re-export the most commonly used items at crate root for convenience.
pub use format::MemmapMode;
pub use npy::{load, load_dynamic, save, NpyElement};
pub use npz::{savez, savez_compressed, NpzFile};
pub use text::{genfromtxt, loadtxt, savetxt, SaveTxtOptions};
