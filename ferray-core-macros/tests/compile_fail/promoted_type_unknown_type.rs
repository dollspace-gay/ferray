//! `promoted_type!` only recognises ferray's known scalar hierarchy
//! (bool/integers/floats/complex). Unknown types should fail with a
//! "cannot promote types" diagnostic rather than silently picking
//! one of the arguments.

use ferray_core_macros::promoted_type;

struct MyCustomType;

fn main() {
    let _x: promoted_type!(MyCustomType, f64);
}
