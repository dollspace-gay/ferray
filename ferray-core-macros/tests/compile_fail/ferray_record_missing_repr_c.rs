//! FerrayRecord derive requires `#[repr(C)]` to guarantee a stable
//! memory layout for FFI and raw-buffer interop. This trybuild test
//! is a compile-fail fixture — it must never compile successfully.
//! The body of `main` doesn't actually run; trybuild just checks that
//! the compiler emits the expected diagnostic for the derive above.

use ferray_core_macros::FerrayRecord;

#[derive(FerrayRecord)]
struct NoReprC {
    x: f64,
    y: i32,
}

fn main() {
    // Sanity: reference the bad type so the derive is actually invoked.
    let _ = std::marker::PhantomData::<NoReprC>;
}
