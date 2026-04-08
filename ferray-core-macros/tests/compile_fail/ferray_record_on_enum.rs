//! FerrayRecord only makes sense for structs — enums have no stable
//! field layout to describe. This compile-fail fixture verifies that
//! deriving on an enum produces a clear error.

use ferray_core_macros::FerrayRecord;

#[derive(FerrayRecord)]
#[repr(C)]
enum NotAStruct {
    A,
    B,
}

fn main() {
    let _ = std::marker::PhantomData::<NotAStruct>;
}
