//! FerrayRecord needs named fields to emit per-field descriptors.
//! Tuple structs have anonymous positional fields so the derive
//! should refuse them with a clear error.

use ferray_core_macros::FerrayRecord;

#[derive(FerrayRecord)]
#[repr(C)]
struct TupleStruct(f64, i32);

fn main() {
    let _ = std::marker::PhantomData::<TupleStruct>;
}
