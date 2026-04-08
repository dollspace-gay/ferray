//! FerrayRecord requires named fields to emit descriptors. A unit
//! struct has no fields at all, so the derive must refuse with the
//! same diagnostic used for tuple structs ("only supports structs
//! with named fields").

use ferray_core_macros::FerrayRecord;

#[derive(FerrayRecord)]
#[repr(C)]
struct UnitStruct;

fn main() {
    let _ = std::marker::PhantomData::<UnitStruct>;
}
