//! Quickstart: the ten-line "hello world" for `ferray`.
//!
//! Run with:
//!
//! ```text
//! cargo run --example quickstart
//! ```
//!
//! This example is deliberately tiny and only depends on the default
//! feature set, so `docs.rs` can render it and downstream users have
//! a concrete starting point (issue #331).

use ferray::prelude::*;

fn main() -> FerrayResult<()> {
    // Create a 3-element 1-D array from a Vec.
    let a: Array<f64, Ix1> = Array::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0])?;

    // Apply an elementwise ufunc.
    let squared = square(&a)?;

    // Reduce along the sole axis.
    let total = sum(&squared, None)?;
    let total_scalar = total.iter().next().copied().unwrap_or(0.0);

    println!("a       = {a}");
    println!("a^2     = {squared}");
    println!("sum(a^2) = {total_scalar}");
    assert!((total_scalar - 14.0).abs() < 1e-12);

    Ok(())
}
