// ferray-core: FerrayRecord support types (REQ-8 prep)
//
// This module defines the traits and types that `#[derive(FerrayRecord)]`
// (implemented by Agent 1d in ferray-core-macros) will generate impls for.
// The proc macro itself is NOT implemented here.

use crate::dtype::DType;

/// Describes a single field within a structured (record) dtype.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldDescriptor {
    /// Name of the field.
    pub name: &'static str,
    /// The scalar dtype of this field.
    pub dtype: DType,
    /// Byte offset of this field within the record.
    pub offset: usize,
    /// Size in bytes of this field.
    pub size: usize,
}

/// Trait implemented by types that can be used as structured array elements.
///
/// `#[derive(FerrayRecord)]` generates this implementation automatically.
/// It provides the field descriptors needed for zero-copy strided views
/// of individual fields within an array of structs.
///
/// # Safety
/// Implementors must ensure that:
/// - The struct is `#[repr(C)]` (no field reordering by the compiler).
/// - All fields implement [`Element`](crate::dtype::Element).
/// - `field_descriptors()` accurately reflects the struct layout.
pub unsafe trait FerrayRecord: Clone + Send + Sync + 'static {
    /// Return descriptors for all fields, in declaration order.
    fn field_descriptors() -> &'static [FieldDescriptor];

    /// Total size of one record in bytes (same as `core::mem::size_of::<Self>()`).
    fn record_size() -> usize;

    /// Return the field descriptor for a named field, if it exists.
    #[must_use]
    fn field_by_name(name: &str) -> Option<&'static FieldDescriptor> {
        Self::field_descriptors().iter().find(|fd| fd.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Manual implementation to verify the trait works before the proc macro exists.
    #[repr(C)]
    #[derive(Clone, Debug)]
    struct TestRecord {
        x: f64,
        y: f64,
        label: i32,
    }

    // In real usage, #[derive(FerrayRecord)] generates this.
    unsafe impl FerrayRecord for TestRecord {
        fn field_descriptors() -> &'static [FieldDescriptor] {
            static FIELDS: [FieldDescriptor; 3] = [
                FieldDescriptor {
                    name: "x",
                    dtype: DType::F64,
                    offset: 0,
                    size: 8,
                },
                FieldDescriptor {
                    name: "y",
                    dtype: DType::F64,
                    offset: 8,
                    size: 8,
                },
                FieldDescriptor {
                    name: "label",
                    dtype: DType::I32,
                    offset: 16,
                    size: 4,
                },
            ];
            &FIELDS
        }

        fn record_size() -> usize {
            core::mem::size_of::<Self>()
        }
    }

    #[test]
    fn record_field_descriptors() {
        let fields = TestRecord::field_descriptors();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "x");
        assert_eq!(fields[0].dtype, DType::F64);
        assert_eq!(fields[1].name, "y");
        assert_eq!(fields[2].name, "label");
        assert_eq!(fields[2].dtype, DType::I32);
    }

    #[test]
    fn record_field_by_name() {
        let fd = TestRecord::field_by_name("y").unwrap();
        assert_eq!(fd.dtype, DType::F64);
        assert_eq!(fd.offset, 8);

        assert!(TestRecord::field_by_name("nonexistent").is_none());
    }

    #[test]
    fn record_size() {
        assert!(TestRecord::record_size() >= 20); // at least 8+8+4, may have padding
    }
}
