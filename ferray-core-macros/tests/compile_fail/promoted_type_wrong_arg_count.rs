//! `promoted_type!` takes exactly two type arguments; one argument
//! must fail with a clear "expects exactly two type arguments"
//! diagnostic.

use ferray_core_macros::promoted_type;

fn main() {
    let _x: promoted_type!(f64) = 0.0;
}
