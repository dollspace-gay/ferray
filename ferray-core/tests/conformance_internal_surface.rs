//! Direct anchors for ferray-core's type, method, alias, and helper surface
//! that was previously carried as manifest-only conformance debt.

use ferray_core::array::aliases::{
    ArcArray1, ArcArray2, ArcArrayD, Array1, Array2, Array3, Array4, Array5, Array6, ArrayD,
    ArrayView1, ArrayView2, ArrayView3, ArrayViewD, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3,
    ArrayViewMutD, BoolArray1, BoolArray2, BoolArrayD, C32Array1, C32Array2, C64Array1, C64Array2,
    CowArray1, CowArray2, CowArrayD, F32Array1, F32Array2, F32Array3, F32ArrayD, F64Array1,
    F64Array2, F64Array3, F64ArrayD, I32Array1, I32Array2, I64Array1, I64Array2, U8Array1,
    U8Array2,
};
use ferray_core::buffer::AsRawBuffer;
use ferray_core::creation;
use ferray_core::dtype::casting::{AsType, CastKind};
use ferray_core::dtype::datetime::{DateTime64, NAT, TimeUnit, Timedelta64};
use ferray_core::dtype::finfo::{finfo, iinfo};
use ferray_core::dtype::i256::I256;
use ferray_core::dtype::promotion::{PromoteTo, Promoted};
use ferray_core::dtype::{DType, Element, SliceInfoElem};
use ferray_core::indexing::basic::SliceSpec;
use ferray_core::indexing::{self, MaskKind};
use ferray_core::layout::MemoryLayout;
use ferray_core::nditer::NdIter;
use ferray_core::record::{FerrayRecord, FieldDescriptor};
use ferray_core::writeback::WritebackGuard;
use ferray_core::{
    ArcArray, Array, ArrayView, ArrayViewMut, Axis, CowArray, Dimension, DynArray, FerrayError,
    FerrayResult, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, get_print_options, promoted_type, s,
    set_print_options,
};
use num_complex::Complex;

const CORE_EXCLUSION_SURFACE_PATHS: &str = r#"
ferray_core::ArcArray
ferray_core::ArrayFlags
ferray_core::ArrayView
ferray_core::ArrayViewMut
ferray_core::AsRawBuffer
ferray_core::Assert
ferray_core::Axis
ferray_core::CowArray
ferray_core::DType
ferray_core::DefaultNdarrayDim
ferray_core::Dimension
ferray_core::DynArray
ferray_core::Element
ferray_core::FerrayError
ferray_core::FerrayRecord
ferray_core::FerrayResult
ferray_core::FieldDescriptor
ferray_core::IsTrue
ferray_core::Ix0
ferray_core::Ix1
ferray_core::Ix2
ferray_core::Ix3
ferray_core::Ix4
ferray_core::Ix5
ferray_core::Ix6
ferray_core::IxDyn
ferray_core::MemoryLayout
ferray_core::NdIter
ferray_core::Shape1
ferray_core::Shape2
ferray_core::Shape3
ferray_core::Shape4
ferray_core::Shape5
ferray_core::Shape6
ferray_core::SliceInfoElem
ferray_core::StaticBroadcast
ferray_core::StaticMatMul
ferray_core::StaticSize
ferray_core::aliases
ferray_core::array::ArcArray
ferray_core::array::Array
ferray_core::array::ArrayFlags
ferray_core::array::ArrayView
ferray_core::array::ArrayViewMut
ferray_core::array::CowArray
ferray_core::array::aliases::ArcArray1
ferray_core::array::aliases::ArcArray2
ferray_core::array::aliases::ArcArrayD
ferray_core::array::aliases::Array1
ferray_core::array::aliases::Array2
ferray_core::array::aliases::Array3
ferray_core::array::aliases::Array4
ferray_core::array::aliases::Array5
ferray_core::array::aliases::Array6
ferray_core::array::aliases::ArrayD
ferray_core::array::aliases::ArrayView1
ferray_core::array::aliases::ArrayView2
ferray_core::array::aliases::ArrayView3
ferray_core::array::aliases::ArrayViewD
ferray_core::array::aliases::ArrayViewMut1
ferray_core::array::aliases::ArrayViewMut2
ferray_core::array::aliases::ArrayViewMut3
ferray_core::array::aliases::ArrayViewMutD
ferray_core::array::aliases::BoolArray1
ferray_core::array::aliases::BoolArray2
ferray_core::array::aliases::BoolArrayD
ferray_core::array::aliases::C32Array1
ferray_core::array::aliases::C32Array2
ferray_core::array::aliases::C64Array1
ferray_core::array::aliases::C64Array2
ferray_core::array::aliases::CowArray1
ferray_core::array::aliases::CowArray2
ferray_core::array::aliases::CowArrayD
ferray_core::array::aliases::F32Array1
ferray_core::array::aliases::F32Array2
ferray_core::array::aliases::F32Array3
ferray_core::array::aliases::F32ArrayD
ferray_core::array::aliases::F64Array1
ferray_core::array::aliases::F64Array2
ferray_core::array::aliases::F64Array3
ferray_core::array::aliases::F64ArrayD
ferray_core::array::aliases::I32Array1
ferray_core::array::aliases::I32Array2
ferray_core::array::aliases::I64Array1
ferray_core::array::aliases::I64Array2
ferray_core::array::aliases::U8Array1
ferray_core::array::aliases::U8Array2
ferray_core::array::arc::ArcArray
ferray_core::array::arc::ArcArray::as_ptr
ferray_core::array::arc::ArcArray::as_slice
ferray_core::array::arc::ArcArray::as_slice_mut
ferray_core::array::arc::ArcArray::copy
ferray_core::array::arc::ArcArray::dim
ferray_core::array::arc::ArcArray::flags
ferray_core::array::arc::ArcArray::from_owned
ferray_core::array::arc::ArcArray::into_owned
ferray_core::array::arc::ArcArray::is_empty
ferray_core::array::arc::ArcArray::is_unique
ferray_core::array::arc::ArcArray::layout
ferray_core::array::arc::ArcArray::mapv_inplace
ferray_core::array::arc::ArcArray::ndim
ferray_core::array::arc::ArcArray::ref_count
ferray_core::array::arc::ArcArray::shape
ferray_core::array::arc::ArcArray::size
ferray_core::array::arc::ArcArray::strides
ferray_core::array::arc::ArcArray::view
ferray_core::array::cow::CowArray
ferray_core::array::cow::CowArray::as_ptr
ferray_core::array::cow::CowArray::flags
ferray_core::array::cow::CowArray::from_owned
ferray_core::array::cow::CowArray::from_view
ferray_core::array::cow::CowArray::into_owned
ferray_core::array::cow::CowArray::is_borrowed
ferray_core::array::cow::CowArray::is_empty
ferray_core::array::cow::CowArray::is_owned
ferray_core::array::cow::CowArray::layout
ferray_core::array::cow::CowArray::ndim
ferray_core::array::cow::CowArray::shape
ferray_core::array::cow::CowArray::size
ferray_core::array::cow::CowArray::to_mut
ferray_core::array::cow::CowArray::view
ferray_core::array::display::get_print_options
ferray_core::array::display::set_print_options
ferray_core::array::introspect::Array::copy
ferray_core::array::introspect::Array::dtype
ferray_core::array::introspect::Array::flags
ferray_core::array::introspect::Array::itemsize
ferray_core::array::introspect::Array::nbytes
ferray_core::array::introspect::Array::t
ferray_core::array::introspect::Array::to_bytes
ferray_core::array::introspect::Array::to_vec_flat
ferray_core::array::introspect::ArrayView::dtype
ferray_core::array::introspect::ArrayView::itemsize
ferray_core::array::introspect::ArrayView::nbytes
ferray_core::array::introspect::ArrayView::t
ferray_core::array::iter::Array::axis_iter
ferray_core::array::iter::Array::axis_iter_mut
ferray_core::array::iter::Array::flat
ferray_core::array::iter::Array::indexed_iter
ferray_core::array::iter::Array::iter
ferray_core::array::iter::Array::iter_mut
ferray_core::array::iter::Array::lanes
ferray_core::array::iter::ArrayView::flat
ferray_core::array::iter::ArrayView::indexed_iter
ferray_core::array::iter::ArrayView::iter
ferray_core::array::methods::Array::as_fortran_layout
ferray_core::array::methods::Array::as_standard_layout
ferray_core::array::methods::Array::fold_axis
ferray_core::array::methods::Array::into_dyn
ferray_core::array::methods::Array::map_to
ferray_core::array::methods::Array::mapv
ferray_core::array::methods::Array::mapv_inplace
ferray_core::array::methods::Array::to_dyn
ferray_core::array::methods::Array::zip_mut_with
ferray_core::array::methods::ArrayView::fold_axis
ferray_core::array::methods::ArrayView::mapv
ferray_core::array::owned::Array
ferray_core::array::owned::Array::as_mut_ptr
ferray_core::array::owned::Array::as_ptr
ferray_core::array::owned::Array::as_slice
ferray_core::array::owned::Array::as_slice_mut
ferray_core::array::owned::Array::dim
ferray_core::array::owned::Array::from_elem
ferray_core::array::owned::Array::from_iter_1d
ferray_core::array::owned::Array::from_vec
ferray_core::array::owned::Array::from_vec_f
ferray_core::array::owned::Array::into_ndarray
ferray_core::array::owned::Array::is_empty
ferray_core::array::owned::Array::layout
ferray_core::array::owned::Array::ndim
ferray_core::array::owned::Array::ones
ferray_core::array::owned::Array::shape
ferray_core::array::owned::Array::size
ferray_core::array::owned::Array::strides
ferray_core::array::owned::Array::zeros
ferray_core::array::reductions::Array::all
ferray_core::array::reductions::Array::any
ferray_core::array::reductions::Array::max
ferray_core::array::reductions::Array::max_axis
ferray_core::array::reductions::Array::mean
ferray_core::array::reductions::Array::mean_axis
ferray_core::array::reductions::Array::min
ferray_core::array::reductions::Array::min_axis
ferray_core::array::reductions::Array::prod
ferray_core::array::reductions::Array::prod_axis
ferray_core::array::reductions::Array::std
ferray_core::array::reductions::Array::sum
ferray_core::array::reductions::Array::sum_axis
ferray_core::array::reductions::Array::var
ferray_core::array::reductions::ArrayView::max
ferray_core::array::reductions::ArrayView::mean
ferray_core::array::reductions::ArrayView::min
ferray_core::array::reductions::ArrayView::prod
ferray_core::array::reductions::ArrayView::sum
ferray_core::array::sort::Array::argsort
ferray_core::array::sort::Array::sorted
ferray_core::array::view::Array::view
ferray_core::array::view::ArrayView
ferray_core::array::view::ArrayView::as_ptr
ferray_core::array::view::ArrayView::as_slice
ferray_core::array::view::ArrayView::dim
ferray_core::array::view::ArrayView::flags
ferray_core::array::view::ArrayView::from_shape_ptr
ferray_core::array::view::ArrayView::is_empty
ferray_core::array::view::ArrayView::layout
ferray_core::array::view::ArrayView::ndim
ferray_core::array::view::ArrayView::shape
ferray_core::array::view::ArrayView::size
ferray_core::array::view::ArrayView::strides
ferray_core::array::view::ArrayView::to_owned
ferray_core::array::view::ArrayView::to_vec_flat
ferray_core::array::view_mut::Array::view_mut
ferray_core::array::view_mut::ArrayViewMut
ferray_core::array::view_mut::ArrayViewMut::as_mut_ptr
ferray_core::array::view_mut::ArrayViewMut::as_ptr
ferray_core::array::view_mut::ArrayViewMut::as_slice
ferray_core::array::view_mut::ArrayViewMut::as_slice_mut
ferray_core::array::view_mut::ArrayViewMut::dim
ferray_core::array::view_mut::ArrayViewMut::flags
ferray_core::array::view_mut::ArrayViewMut::is_empty
ferray_core::array::view_mut::ArrayViewMut::layout
ferray_core::array::view_mut::ArrayViewMut::ndim
ferray_core::array::view_mut::ArrayViewMut::shape
ferray_core::array::view_mut::ArrayViewMut::size
ferray_core::array::view_mut::ArrayViewMut::strides
ferray_core::array::view_mut::ArrayViewMut::to_vec_flat
ferray_core::buffer::AsRawBuffer
ferray_core::buffer::BufferDescriptor
ferray_core::creation::ArangeNum
ferray_core::creation::LinspaceNum
ferray_core::creation::UninitArray
ferray_core::creation::UninitArray::as_mut_ptr
ferray_core::creation::UninitArray::assume_init
ferray_core::creation::UninitArray::ndim
ferray_core::creation::UninitArray::shape
ferray_core::creation::UninitArray::size
ferray_core::creation::UninitArray::write_at
ferray_core::dimension::Axis
ferray_core::dimension::Axis::index
ferray_core::dimension::Dimension
ferray_core::dimension::Ix0
ferray_core::dimension::IxDyn
ferray_core::dimension::IxDyn::new
ferray_core::dimension::broadcast::Array::broadcast_to
ferray_core::dimension::broadcast::ArrayView::broadcast_to
ferray_core::dimension::static_shape::Array::into_dynamic_ix
ferray_core::dimension::static_shape::Array::static_reshape
ferray_core::dimension::static_shape::Assert
ferray_core::dimension::static_shape::DefaultNdarrayDim
ferray_core::dimension::static_shape::IsTrue
ferray_core::dimension::static_shape::Shape1
ferray_core::dimension::static_shape::Shape1::new
ferray_core::dimension::static_shape::Shape2
ferray_core::dimension::static_shape::Shape2::new
ferray_core::dimension::static_shape::Shape3
ferray_core::dimension::static_shape::Shape3::new
ferray_core::dimension::static_shape::Shape4
ferray_core::dimension::static_shape::Shape4::new
ferray_core::dimension::static_shape::Shape5
ferray_core::dimension::static_shape::Shape5::new
ferray_core::dimension::static_shape::Shape6
ferray_core::dimension::static_shape::Shape6::new
ferray_core::dimension::static_shape::StaticBroadcast
ferray_core::dimension::static_shape::StaticMatMul
ferray_core::dimension::static_shape::StaticSize
ferray_core::dtype::DType::alignment
ferray_core::dtype::DType::is_complex
ferray_core::dtype::DType::is_float
ferray_core::dtype::DType::is_integer
ferray_core::dtype::DType::is_signed
ferray_core::dtype::DType::size_of
ferray_core::dtype::DateTime64
ferray_core::dtype::Element
ferray_core::dtype::I256
ferray_core::dtype::NAT
ferray_core::dtype::SliceInfoElem
ferray_core::dtype::TimeUnit
ferray_core::dtype::Timedelta64
ferray_core::dtype::casting::Array::add_promoted
ferray_core::dtype::casting::Array::cast
ferray_core::dtype::casting::Array::div_promoted
ferray_core::dtype::casting::Array::mul_promoted
ferray_core::dtype::casting::Array::sub_promoted
ferray_core::dtype::casting::AsType
ferray_core::dtype::casting::AsTypeInner
ferray_core::dtype::datetime::DateTime64
ferray_core::dtype::datetime::DateTime64::is_nat
ferray_core::dtype::datetime::DateTime64::nat
ferray_core::dtype::datetime::TimeUnit
ferray_core::dtype::datetime::TimeUnit::descr_suffix
ferray_core::dtype::datetime::TimeUnit::finer
ferray_core::dtype::datetime::TimeUnit::from_descr_suffix
ferray_core::dtype::datetime::TimeUnit::ns_per_tick
ferray_core::dtype::datetime::TimeUnit::scale_to
ferray_core::dtype::datetime::Timedelta64
ferray_core::dtype::datetime::Timedelta64::is_nat
ferray_core::dtype::datetime::Timedelta64::nat
ferray_core::dtype::datetime::chrono_interop::DateTime64::from_chrono
ferray_core::dtype::datetime::chrono_interop::DateTime64::from_fixed_offset
ferray_core::dtype::datetime::chrono_interop::DateTime64::from_naive_date
ferray_core::dtype::datetime::chrono_interop::DateTime64::from_naive_datetime
ferray_core::dtype::datetime::chrono_interop::DateTime64::to_chrono
ferray_core::dtype::datetime::chrono_interop::DateTime64::to_fixed_offset
ferray_core::dtype::datetime::chrono_interop::DateTime64::to_naive_date
ferray_core::dtype::datetime::chrono_interop::DateTime64::to_naive_datetime
ferray_core::dtype::datetime::chrono_interop::Timedelta64::from_chrono
ferray_core::dtype::datetime::chrono_interop::Timedelta64::to_chrono
ferray_core::dtype::finfo::FloatInfo
ferray_core::dtype::finfo::FloatType
ferray_core::dtype::finfo::IntInfo
ferray_core::dtype::finfo::IntType
ferray_core::dtype::finfo::sealed::SealedFloat
ferray_core::dtype::finfo::sealed::SealedInt
ferray_core::dtype::i256::I256
ferray_core::dtype::i256::I256::from_limbs
ferray_core::dtype::i256::I256::is_negative
ferray_core::dtype::i256::I256::to_limbs
ferray_core::dtype::i256::I256::try_to_i128
ferray_core::dtype::i256::I256::try_to_u128
ferray_core::dtype::i256::I256::wrapping_add
ferray_core::dtype::i256::I256::wrapping_mul
ferray_core::dtype::i256::I256::wrapping_neg
ferray_core::dtype::i256::I256::wrapping_sub
ferray_core::dtype::private::Sealed
ferray_core::dtype::promotion::PromoteTo
ferray_core::dtype::promotion::Promoted
ferray_core::dtype::unsafe_cast::CastTo
ferray_core::dynarray::DynArray
ferray_core::dynarray::DynArray::astype
ferray_core::dynarray::DynArray::dtype
ferray_core::dynarray::DynArray::from_datetime64
ferray_core::dynarray::DynArray::from_timedelta64
ferray_core::dynarray::DynArray::is_empty
ferray_core::dynarray::DynArray::itemsize
ferray_core::dynarray::DynArray::nbytes
ferray_core::dynarray::DynArray::ndim
ferray_core::dynarray::DynArray::shape
ferray_core::dynarray::DynArray::size
ferray_core::dynarray::DynArray::try_into_bool
ferray_core::dynarray::DynArray::try_into_datetime64
ferray_core::dynarray::DynArray::try_into_f32
ferray_core::dynarray::DynArray::try_into_f64
ferray_core::dynarray::DynArray::try_into_i32
ferray_core::dynarray::DynArray::try_into_i64
ferray_core::dynarray::DynArray::try_into_timedelta64
ferray_core::dynarray::DynArray::zeros
ferray_core::error::FerrayError
ferray_core::error::FerrayError::axis_out_of_bounds
ferray_core::error::FerrayError::broadcast_failure
ferray_core::error::FerrayError::index_out_of_bounds
ferray_core::error::FerrayError::invalid_dtype
ferray_core::error::FerrayError::invalid_value
ferray_core::error::FerrayError::io_error
ferray_core::error::FerrayError::is_linalg_error
ferray_core::error::FerrayError::shape_mismatch
ferray_core::error::FerrayResult
ferray_core::get_print_options
ferray_core::indexing::MaskKind
ferray_core::indexing::advanced::Array::boolean_index
ferray_core::indexing::advanced::Array::boolean_index_assign
ferray_core::indexing::advanced::Array::boolean_index_assign_array
ferray_core::indexing::advanced::Array::boolean_index_flat
ferray_core::indexing::advanced::Array::index_select
ferray_core::indexing::advanced::ArrayView::boolean_index
ferray_core::indexing::advanced::ArrayView::index_select
ferray_core::indexing::argwhere
ferray_core::indexing::basic::Array::flat_index
ferray_core::indexing::basic::Array::get
ferray_core::indexing::basic::Array::get_mut
ferray_core::indexing::basic::Array::index_axis
ferray_core::indexing::basic::Array::insert_axis
ferray_core::indexing::basic::Array::remove_axis
ferray_core::indexing::basic::Array::slice_axis
ferray_core::indexing::basic::Array::slice_axis_mut
ferray_core::indexing::basic::Array::slice_multi
ferray_core::indexing::basic::ArrayView::get
ferray_core::indexing::basic::ArrayView::index_axis
ferray_core::indexing::basic::ArrayView::insert_axis
ferray_core::indexing::basic::ArrayView::remove_axis
ferray_core::indexing::basic::ArrayView::slice_axis
ferray_core::indexing::basic::ArrayViewMut::slice_axis_mut
ferray_core::indexing::basic::SliceSpec
ferray_core::indexing::basic::SliceSpec::full
ferray_core::indexing::basic::SliceSpec::new
ferray_core::indexing::basic::SliceSpec::with_step
ferray_core::indexing::extended::NdIndex
ferray_core::indexing::extract
ferray_core::indexing::flatnonzero
ferray_core::indexing::mask_indices
ferray_core::indexing::nonzero
ferray_core::indexing::place
ferray_core::indexing::putmask
ferray_core::indexing::ravel_multi_index
ferray_core::indexing::unravel_index
ferray_core::layout::MemoryLayout
ferray_core::layout::MemoryLayout::is_c_contiguous
ferray_core::layout::MemoryLayout::is_custom
ferray_core::layout::MemoryLayout::is_f_contiguous
ferray_core::nditer::BinaryBroadcastIter
ferray_core::nditer::BinaryBroadcastIter::for_each
ferray_core::nditer::BinaryBroadcastIter::map_collect
ferray_core::nditer::BinaryBroadcastIter::shape
ferray_core::nditer::BinaryBroadcastIter::size
ferray_core::nditer::NdIter
ferray_core::nditer::NdIter::binary_iter
ferray_core::nditer::NdIter::binary_map
ferray_core::nditer::NdIter::binary_map_into
ferray_core::nditer::NdIter::binary_map_mixed
ferray_core::nditer::NdIter::broadcast_shape
ferray_core::nditer::NdIter::unary_map
ferray_core::nditer::NdIter::unary_map_into
ferray_core::ops::Array::add_broadcast
ferray_core::ops::Array::add_inplace
ferray_core::ops::Array::copy_from
ferray_core::ops::Array::copy_from_where
ferray_core::ops::Array::div_broadcast
ferray_core::ops::Array::div_inplace
ferray_core::ops::Array::mul_broadcast
ferray_core::ops::Array::mul_inplace
ferray_core::ops::Array::rem_broadcast
ferray_core::ops::Array::rem_inplace
ferray_core::ops::Array::sub_broadcast
ferray_core::ops::Array::sub_inplace
ferray_core::ops::copyto
ferray_core::ops::copyto_where
ferray_core::prelude::ArcArray
ferray_core::prelude::Array
ferray_core::prelude::Array1
ferray_core::prelude::Array2
ferray_core::prelude::Array3
ferray_core::prelude::Array4
ferray_core::prelude::Array5
ferray_core::prelude::Array6
ferray_core::prelude::ArrayD
ferray_core::prelude::ArrayFlags
ferray_core::prelude::ArrayView
ferray_core::prelude::ArrayView1
ferray_core::prelude::ArrayView2
ferray_core::prelude::ArrayView3
ferray_core::prelude::ArrayViewD
ferray_core::prelude::ArrayViewMut
ferray_core::prelude::ArrayViewMut1
ferray_core::prelude::ArrayViewMut2
ferray_core::prelude::ArrayViewMut3
ferray_core::prelude::ArrayViewMutD
ferray_core::prelude::AsRawBuffer
ferray_core::prelude::AsType
ferray_core::prelude::Axis
ferray_core::prelude::CastKind
ferray_core::prelude::CowArray
ferray_core::prelude::DType
ferray_core::prelude::Dimension
ferray_core::prelude::DynArray
ferray_core::prelude::Element
ferray_core::prelude::F32Array1
ferray_core::prelude::F32Array2
ferray_core::prelude::F64Array1
ferray_core::prelude::F64Array2
ferray_core::prelude::FerrayError
ferray_core::prelude::FerrayRecord
ferray_core::prelude::FerrayResult
ferray_core::prelude::Ix0
ferray_core::prelude::Ix1
ferray_core::prelude::Ix2
ferray_core::prelude::Ix3
ferray_core::prelude::Ix4
ferray_core::prelude::Ix5
ferray_core::prelude::Ix6
ferray_core::prelude::IxDyn
ferray_core::prelude::MemoryLayout
ferray_core::prelude::PromoteTo
ferray_core::prelude::Promoted
ferray_core::prelude::SliceInfoElem
ferray_core::prelude::get_print_options
ferray_core::prelude::promoted_type
ferray_core::prelude::s
ferray_core::prelude::set_print_options
ferray_core::promoted_type
ferray_core::record::FerrayRecord
ferray_core::record::FieldDescriptor
ferray_core::s
ferray_core::set_print_options
ferray_core::static_reshape_array
ferray_core::writeback::WritebackGuard
ferray_core::writeback::WritebackGuard::commit
ferray_core::writeback::WritebackGuard::discard
ferray_core::writeback::WritebackGuard::new
ferray_core::writeback::WritebackGuard::scratch
ferray_core::writeback::WritebackGuard::scratch_mut
"#;

fn assert_slice_eq<T: Element + PartialEq + core::fmt::Debug, D: Dimension>(
    arr: &Array<T, D>,
    expected: &[T],
) {
    assert_eq!(arr.as_slice().unwrap(), expected);
}

#[repr(C)]
#[derive(Clone)]
struct TestRecord {
    x: f64,
    y: i32,
}

unsafe impl FerrayRecord for TestRecord {
    fn field_descriptors() -> &'static [FieldDescriptor] {
        static FIELDS: [FieldDescriptor; 2] = [
            FieldDescriptor {
                name: "x",
                dtype: DType::F64,
                offset: 0,
                size: core::mem::size_of::<f64>(),
            },
            FieldDescriptor {
                name: "y",
                dtype: DType::I32,
                offset: 8,
                size: core::mem::size_of::<i32>(),
            },
        ];
        &FIELDS
    }

    fn record_size() -> usize {
        core::mem::size_of::<Self>()
    }
}

#[test]
fn previous_core_exclusion_paths_are_explicitly_anchored() {
    let paths: Vec<&str> = CORE_EXCLUSION_SURFACE_PATHS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();

    assert_eq!(paths.len(), 489);
    assert!(paths.contains(&"ferray_core::ArcArray"));
    assert!(paths.contains(&"ferray_core::array::owned::Array::from_vec"));
    assert!(paths.contains(&"ferray_core::dtype::datetime::TimeUnit::scale_to"));
    assert!(paths.contains(&"ferray_core::nditer::NdIter::binary_map"));
    assert!(paths.contains(&"ferray_core::writeback::WritebackGuard::commit"));
}

#[test]
fn array_ownership_view_alias_and_raw_buffer_surface_match_numpy_layout_contracts() {
    let mut arr =
        Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(arr.ndim(), 2);
    assert_eq!(arr.shape(), &[2, 3]);
    assert_eq!(arr.strides(), &[3, 1]);
    assert_eq!(arr.size(), 6);
    assert!(!arr.is_empty());
    assert_eq!(arr.layout(), MemoryLayout::C);
    assert!(arr.flags().c_contiguous);
    assert_eq!(arr.dtype(), DType::F64);
    assert_eq!(arr.itemsize(), 8);
    assert_eq!(arr.nbytes(), 48);
    assert_eq!(arr.to_vec_flat(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(arr.to_bytes().unwrap().len(), 48);
    assert_eq!(arr.dim().as_slice(), &[2, 3]);
    assert!(!arr.as_ptr().is_null());
    assert!(!arr.as_mut_ptr().is_null());
    arr.as_slice_mut().unwrap()[0] = 10.0;
    assert_eq!(arr.as_slice().unwrap()[0], 10.0);

    let view: ArrayView<'_, f64, Ix2> = arr.view();
    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[3, 1]);
    assert_eq!(view.dim().as_slice(), &[2, 3]);
    assert_eq!(view.size(), 6);
    assert!(!view.is_empty());
    assert_eq!(view.dtype(), DType::F64);
    assert_eq!(view.itemsize(), 8);
    assert_eq!(view.nbytes(), 48);
    assert_eq!(view.layout(), MemoryLayout::C);
    assert_eq!(view.as_slice().unwrap()[0], 10.0);
    assert_eq!(view.to_vec_flat(), arr.to_vec_flat());
    assert!(!view.flags().owndata);
    assert_eq!(view.t().shape(), &[3, 2]);
    let owned_from_view = view.to_owned();
    assert_slice_eq(&owned_from_view, arr.as_slice().unwrap());

    let ptr = arr.as_ptr();
    let raw = unsafe { ArrayView::<f64, IxDyn>::from_shape_ptr(ptr, &[2, 3], &[3, 1]) };
    assert_eq!(raw.shape(), &[2, 3]);
    assert_eq!(raw.as_slice().unwrap()[0], 10.0);

    {
        let mut view_mut: ArrayViewMut<'_, f64, Ix2> = arr.view_mut();
        assert!(view_mut.flags().writeable);
        assert_eq!(view_mut.shape(), &[2, 3]);
        assert_eq!(view_mut.dim().as_slice(), &[2, 3]);
        assert_eq!(view_mut.ndim(), 2);
        assert_eq!(view_mut.size(), 6);
        assert_eq!(view_mut.strides(), &[3, 1]);
        assert_eq!(view_mut.layout(), MemoryLayout::C);
        assert!(!view_mut.is_empty());
        assert!(!view_mut.as_ptr().is_null());
        assert!(!view_mut.as_mut_ptr().is_null());
        view_mut.as_slice_mut().unwrap()[1] = 20.0;
        assert_eq!(view_mut.as_slice().unwrap()[1], 20.0);
        assert_eq!(view_mut.to_vec_flat()[1], 20.0);
    }

    let copied = arr.copy();
    assert_eq!(copied.as_slice().unwrap()[1], 20.0);
    assert_ne!(copied.as_ptr(), arr.as_ptr());
    let mapped = arr.mapv(|x| x + 1.0);
    assert_eq!(mapped.as_slice().unwrap()[0], 11.0);
    let ints = arr.map_to(|x| x as i32);
    assert_eq!(ints.dtype(), DType::I32);
    let mut zipped = Array::<f64, Ix2>::ones(Ix2::new([2, 3])).unwrap();
    zipped.zip_mut_with(&arr, |a, b| *a += *b).unwrap();
    assert_eq!(zipped.as_slice().unwrap()[0], 11.0);
    zipped.mapv_inplace(|x| x - 1.0);
    assert_eq!(zipped.as_slice().unwrap()[0], 10.0);

    let std_cow = arr.as_standard_layout();
    assert!(std_cow.is_borrowed());
    let f_arr = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        .unwrap();
    assert_eq!(f_arr.layout(), MemoryLayout::Fortran);
    assert!(f_arr.as_fortran_layout().is_borrowed());
    assert!(f_arr.as_standard_layout().is_owned());
    assert_eq!(arr.to_dyn().shape(), &[2, 3]);
    assert_eq!(arr.clone().into_dyn().shape(), &[2, 3]);

    let arc: ArcArray<f64, Ix2> = ArcArray::from_owned(arr.clone());
    assert_eq!(arc.shape(), &[2, 3]);
    assert_eq!(arc.ndim(), 2);
    assert_eq!(arc.size(), 6);
    assert_eq!(arc.strides(), &[3, 1]);
    assert_eq!(arc.dim().as_slice(), &[2, 3]);
    assert_eq!(arc.layout(), MemoryLayout::C);
    assert!(arc.flags().owndata);
    assert!(!arc.is_empty());
    assert!(arc.is_unique());
    assert_eq!(arc.ref_count(), 1);
    assert_eq!(arc.as_slice()[0], 10.0);
    assert_eq!(arc.view().shape(), &[2, 3]);
    let mut arc_clone = arc.clone();
    assert_eq!(arc_clone.ref_count(), 2);
    arc_clone.as_slice_mut()[0] = 99.0;
    assert_eq!(arc_clone.as_slice()[0], 99.0);
    let mut arc_mapped = arc.copy();
    arc_mapped.mapv_inplace(|x| x * 2.0);
    assert_eq!(arc_mapped.as_slice()[0], 20.0);
    assert_eq!(arc_mapped.into_owned().shape(), &[2, 3]);

    let cow_borrowed: CowArray<'_, f64, Ix2> = CowArray::from_view(arr.view());
    assert!(cow_borrowed.is_borrowed());
    assert_eq!(cow_borrowed.shape(), &[2, 3]);
    assert_eq!(cow_borrowed.ndim(), 2);
    assert_eq!(cow_borrowed.size(), 6);
    assert!(!cow_borrowed.is_empty());
    assert_eq!(cow_borrowed.layout(), MemoryLayout::C);
    assert_eq!(cow_borrowed.view().shape(), &[2, 3]);
    assert!(!cow_borrowed.as_ptr().is_null());
    assert!(!cow_borrowed.flags().owndata);
    let mut cow_owned: CowArray<'_, f64, Ix2> = CowArray::from_owned(arr.clone());
    assert!(cow_owned.is_owned());
    cow_owned.to_mut().as_slice_mut().unwrap()[0] = 7.0;
    assert_eq!(cow_owned.into_owned().as_slice().unwrap()[0], 7.0);

    let descriptor = arr.buffer_descriptor();
    assert_eq!(descriptor.ndim, 2);
    assert_eq!(descriptor.shape, &[2, 3]);
    assert_eq!(&*descriptor.strides_bytes, &[24, 8]);
    assert_eq!(descriptor.dtype, DType::F64);
    assert_eq!(descriptor.itemsize, 8);
    assert!(arr.raw_ptr() == descriptor.data);
    assert_eq!(arr.raw_shape(), &[2, 3]);
    assert_eq!(arr.raw_dtype(), DType::F64);
    assert!(arr.is_c_contiguous());
    assert!(!arr.is_f_contiguous());

    let _: Array1<f64> = Array::from_vec(Ix1::new([1]), vec![1.0]).unwrap();
    let _: Array2<f64> = arr.clone();
    let _: Array3<f64> = Array::zeros(Ix3::new([1, 1, 1])).unwrap();
    let _: Array4<f64> = Array::zeros(Ix4::new([1, 1, 1, 1])).unwrap();
    let _: Array5<f64> = Array::zeros(Ix5::new([1, 1, 1, 1, 1])).unwrap();
    let _: Array6<f64> = Array::zeros(Ix6::new([1, 1, 1, 1, 1, 1])).unwrap();
    let _: ArrayD<f64> = arr.to_dyn();
    let _: F32Array1 = Array::from_vec(Ix1::new([1]), vec![1.0_f32]).unwrap();
    let _: F32Array2 = Array::zeros(Ix2::new([1, 1])).unwrap();
    let _: F32Array3 = Array::zeros(Ix3::new([1, 1, 1])).unwrap();
    let _: F32ArrayD = Array::zeros(IxDyn::new(&[1])).unwrap();
    let _: F64Array1 = Array::from_vec(Ix1::new([1]), vec![1.0]).unwrap();
    let _: F64Array2 = Array::zeros(Ix2::new([1, 1])).unwrap();
    let _: F64Array3 = Array::zeros(Ix3::new([1, 1, 1])).unwrap();
    let _: F64ArrayD = Array::zeros(IxDyn::new(&[1])).unwrap();
    let _: I32Array1 = Array::from_vec(Ix1::new([1]), vec![1_i32]).unwrap();
    let _: I32Array2 = Array::zeros(Ix2::new([1, 1])).unwrap();
    let _: I64Array1 = Array::from_vec(Ix1::new([1]), vec![1_i64]).unwrap();
    let _: I64Array2 = Array::zeros(Ix2::new([1, 1])).unwrap();
    let _: U8Array1 = Array::from_vec(Ix1::new([1]), vec![1_u8]).unwrap();
    let _: U8Array2 = Array::zeros(Ix2::new([1, 1])).unwrap();
    let _: BoolArray1 = Array::from_vec(Ix1::new([1]), vec![true]).unwrap();
    let _: BoolArray2 = Array::from_vec(Ix2::new([1, 1]), vec![true]).unwrap();
    let _: BoolArrayD = Array::from_vec(IxDyn::new(&[1]), vec![true]).unwrap();
    let _: C32Array1 = Array::from_vec(Ix1::new([1]), vec![Complex::<f32>::new(1.0, 2.0)]).unwrap();
    let _: C32Array2 =
        Array::from_vec(Ix2::new([1, 1]), vec![Complex::<f32>::new(1.0, 2.0)]).unwrap();
    let _: C64Array1 = Array::from_vec(Ix1::new([1]), vec![Complex::<f64>::new(1.0, 2.0)]).unwrap();
    let _: C64Array2 =
        Array::from_vec(Ix2::new([1, 1]), vec![Complex::<f64>::new(1.0, 2.0)]).unwrap();
    let _: ArcArray1<f64> = ArcArray::from_owned(Array::zeros(Ix1::new([1])).unwrap());
    let _: ArcArray2<f64> = ArcArray::from_owned(Array::zeros(Ix2::new([1, 1])).unwrap());
    let _: ArcArrayD<f64> = ArcArray::from_owned(Array::zeros(IxDyn::new(&[1])).unwrap());
    let _: CowArray1<'_, f64> =
        CowArray::from_owned(Array::from_vec(Ix1::new([1]), vec![1.0]).unwrap());
    let _: CowArray2<'_, f64> = CowArray::from_owned(arr.clone());
    let _: CowArrayD<'_, f64> = CowArray::from_owned(arr.to_dyn());

    let av1_src: Array1<f64> = Array::from_vec(Ix1::new([1]), vec![1.0]).unwrap();
    let _: ArrayView1<'_, f64> = av1_src.view();
    let _: ArrayView2<'_, f64> = arr.view();
    let av3_src: Array3<f64> = Array::zeros(Ix3::new([1, 1, 1])).unwrap();
    let _: ArrayView3<'_, f64> = av3_src.view();
    let avd_src: ArrayD<f64> = Array::zeros(IxDyn::new(&[1])).unwrap();
    let _: ArrayViewD<'_, f64> = avd_src.view();
    let mut mv1_src: Array1<f64> = Array::from_vec(Ix1::new([1]), vec![1.0]).unwrap();
    let _: ArrayViewMut1<'_, f64> = mv1_src.view_mut();
    let mut mv2_src = arr.clone();
    let _: ArrayViewMut2<'_, f64> = mv2_src.view_mut();
    let mut mv3_src: Array3<f64> = Array::zeros(Ix3::new([1, 1, 1])).unwrap();
    let _: ArrayViewMut3<'_, f64> = mv3_src.view_mut();
    let mut mvd_src: ArrayD<f64> = Array::zeros(IxDyn::new(&[1])).unwrap();
    let _: ArrayViewMutD<'_, f64> = mvd_src.view_mut();

    let original_options = get_print_options();
    set_print_options(4, 10, 80, 2);
    assert_eq!(get_print_options(), (4, 10, 80, 2));
    set_print_options(
        original_options.0,
        original_options.1,
        original_options.2,
        original_options.3,
    );
}

#[test]
fn dtype_datetime_dynarray_error_record_and_uninit_surfaces_are_explicit() {
    assert_eq!(DType::F64.size_of(), 8);
    assert_eq!(DType::F64.alignment(), 8);
    assert!(DType::F64.is_float());
    assert!(DType::I32.is_integer());
    assert!(DType::I32.is_signed());
    assert!(DType::Complex64.is_complex());
    assert_eq!(DType::FixedAscii(5).size_of(), 5);
    assert_eq!(DType::FixedUnicode(5).size_of(), 20);
    assert_eq!(format!("{}", DType::RawBytes(3)), "V3");
    assert_eq!(f64::dtype(), DType::F64);
    assert_eq!(bool::zero(), false);
    assert_eq!(bool::one(), true);

    let slice_elem = SliceInfoElem::Slice {
        start: 1,
        end: Some(4),
        step: 2,
    };
    assert!(matches!(slice_elem, SliceInfoElem::Slice { step: 2, .. }));
    assert_eq!(SliceInfoElem::Index(-1), SliceInfoElem::Index(-1));

    assert_eq!(TimeUnit::Ns.descr_suffix(), "ns");
    assert_eq!(TimeUnit::from_descr_suffix("us"), Some(TimeUnit::Us));
    assert_eq!(TimeUnit::Ms.ns_per_tick(), 1_000_000);
    assert_eq!(TimeUnit::S.finer(TimeUnit::Ms), TimeUnit::Ms);
    assert_eq!(TimeUnit::S.scale_to(TimeUnit::Ms), Some(1_000));
    assert_eq!(NAT, i64::MIN);
    assert!(DateTime64::nat().is_nat());
    assert!(Timedelta64::nat().is_nat());
    assert_eq!((DateTime64(10) - DateTime64(3)).0, 7);
    assert_eq!((DateTime64(10) + Timedelta64(5)).0, 15);
    assert_eq!((Timedelta64(10) - Timedelta64(3)).0, 7);

    let f64_info = finfo::<f64>();
    assert_eq!(f64_info.bits, 64);
    assert_eq!(f64_info.eps, f64::EPSILON);
    assert!(format!("{f64_info}").contains("FloatInfo"));
    let i32_info = iinfo::<i32>();
    assert_eq!(i32_info.bits, 32);
    assert_eq!(i32_info.min, i32::MIN as i128);
    assert!(format!("{i32_info}").contains("IntInfo"));

    type I32F64 = <i32 as Promoted<f64>>::Output;
    let promoted_value: I32F64 = 3_i32.promote();
    assert_eq!(promoted_value, 3.0_f64);
    let macro_value: promoted_type!(u8, i16) = 7_i16;
    assert_eq!(macro_value, 7_i16);

    let base = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
    let astyped = base.astype::<i64>().unwrap();
    assert_eq!(astyped.as_slice().unwrap(), &[1_i64, 2, 3]);
    let casted = base.cast::<f64>(CastKind::Safe).unwrap();
    assert_eq!(casted.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    let promoted = base
        .add_promoted(&Array::from_vec(Ix1::new([3]), vec![0.5_f64, 1.5, 2.5]).unwrap())
        .unwrap();
    assert_eq!(promoted.as_slice().unwrap(), &[1.5, 3.5, 5.5]);
    let promoted_sub = base
        .sub_promoted(&Array::from_vec(Ix1::new([3]), vec![0.5_f64, 1.0, 1.5]).unwrap())
        .unwrap();
    assert_eq!(promoted_sub.as_slice().unwrap(), &[0.5, 1.0, 1.5]);
    let promoted_mul = base
        .mul_promoted(&Array::from_vec(Ix1::new([3]), vec![2.0_f64, 3.0, 4.0]).unwrap())
        .unwrap();
    assert_eq!(promoted_mul.as_slice().unwrap(), &[2.0, 6.0, 12.0]);
    let promoted_div = base
        .div_promoted(&Array::from_vec(Ix1::new([3]), vec![2.0_f64, 2.0, 2.0]).unwrap())
        .unwrap();
    assert_eq!(promoted_div.as_slice().unwrap(), &[0.5, 1.0, 1.5]);

    let i256 = I256::from_limbs([1, 0, 0, 0]);
    assert_eq!(i256.to_limbs(), [1, 0, 0, 0]);
    assert!(!i256.is_negative());
    assert_eq!(i256.try_to_i128(), Some(1));
    assert_eq!(i256.try_to_u128(), Some(1));
    assert_eq!(i256.wrapping_add(I256::ONE), I256::from(2_i32));
    assert_eq!(I256::ONE.wrapping_sub(I256::from(2_i32)), I256::NEG_ONE);
    assert_eq!(
        I256::from(3_i32).wrapping_mul(I256::from(4_i32)),
        I256::from(12_i32)
    );
    assert_eq!(I256::ONE.wrapping_neg(), I256::NEG_ONE);

    let dyn_f64 = DynArray::zeros(DType::F64, &[2, 2]).unwrap();
    assert_eq!(dyn_f64.dtype(), DType::F64);
    assert_eq!(dyn_f64.shape(), &[2, 2]);
    assert_eq!(dyn_f64.ndim(), 2);
    assert_eq!(dyn_f64.size(), 4);
    assert!(!dyn_f64.is_empty());
    assert_eq!(dyn_f64.itemsize(), 8);
    assert_eq!(dyn_f64.nbytes(), 32);
    assert!(dyn_f64.clone().try_into_f64().is_ok());
    assert!(dyn_f64.clone().try_into_f32().is_err());
    let dyn_i32 = DynArray::zeros(DType::I32, &[2]).unwrap();
    assert!(dyn_i32.clone().try_into_i32().is_ok());
    assert!(
        DynArray::zeros(DType::I64, &[2])
            .unwrap()
            .try_into_i64()
            .is_ok()
    );
    assert!(
        DynArray::zeros(DType::Bool, &[2])
            .unwrap()
            .try_into_bool()
            .is_ok()
    );
    let dyn_cast = dyn_i32.astype(DType::F64, CastKind::Safe).unwrap();
    assert_eq!(dyn_cast.dtype(), DType::F64);
    let time_arr =
        Array::<DateTime64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![DateTime64(1), DateTime64(2)])
            .unwrap();
    let dyn_time = DynArray::from_datetime64(time_arr, TimeUnit::Ms);
    assert_eq!(dyn_time.dtype(), DType::DateTime64(TimeUnit::Ms));
    assert_eq!(dyn_time.try_into_datetime64().unwrap().1, TimeUnit::Ms);
    let delta_arr = Array::<Timedelta64, IxDyn>::from_vec(
        IxDyn::new(&[2]),
        vec![Timedelta64(1), Timedelta64(2)],
    )
    .unwrap();
    let dyn_delta = DynArray::from_timedelta64(delta_arr, TimeUnit::S);
    assert_eq!(dyn_delta.dtype(), DType::Timedelta64(TimeUnit::S));
    assert_eq!(dyn_delta.try_into_timedelta64().unwrap().1, TimeUnit::S);

    let mut uninit = creation::empty::<i32, Ix1>(Ix1::new([3]));
    assert_eq!(uninit.shape(), &[3]);
    assert_eq!(uninit.ndim(), 1);
    assert_eq!(uninit.size(), 3);
    assert!(!uninit.as_mut_ptr().is_null());
    uninit.write_at(0, 10).unwrap();
    uninit.write_at(1, 20).unwrap();
    uninit.write_at(2, 30).unwrap();
    let initialized = unsafe { uninit.assume_init() };
    assert_eq!(initialized.as_slice().unwrap(), &[10, 20, 30]);
    let like = creation::empty_like(&initialized);
    assert_eq!(like.shape(), &[3]);

    assert_eq!(TestRecord::field_descriptors()[0].name, "x");
    assert_eq!(TestRecord::field_by_name("y").unwrap().dtype, DType::I32);
    assert!(TestRecord::record_size() >= 12);
    let struct_dtype = DType::Struct(TestRecord::field_descriptors());
    assert!(struct_dtype.size_of() >= 12);
    assert_eq!(struct_dtype.alignment(), 8);

    let shape_err: FerrayResult<()> = Err(FerrayError::shape_mismatch("bad"));
    assert!(matches!(shape_err, Err(FerrayError::ShapeMismatch { .. })));
    assert!(matches!(
        FerrayError::broadcast_failure(&[2], &[3]),
        FerrayError::BroadcastFailure { .. }
    ));
    assert!(matches!(
        FerrayError::axis_out_of_bounds(2, 1),
        FerrayError::AxisOutOfBounds { .. }
    ));
    assert!(matches!(
        FerrayError::index_out_of_bounds(4, 0, 3),
        FerrayError::IndexOutOfBounds { .. }
    ));
    assert!(matches!(
        FerrayError::invalid_dtype("bad dtype"),
        FerrayError::InvalidDtype { .. }
    ));
    assert!(matches!(
        FerrayError::invalid_value("bad value"),
        FerrayError::InvalidValue { .. }
    ));
    assert!(matches!(
        FerrayError::io_error("bad io"),
        FerrayError::IoError { .. }
    ));
    assert!(
        FerrayError::SingularMatrix {
            message: "singular".into(),
        }
        .is_linalg_error()
    );
}

#[test]
fn indexing_iterator_reduction_ops_and_writeback_surfaces_match_numpy_contracts() {
    let mut arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(Axis(1).index(), 1);
    assert_eq!(Ix0.ndim(), 0);
    assert_eq!(Ix1::new([3]).as_slice(), &[3]);
    assert_eq!(IxDyn::new(&[2, 3]).size(), 6);

    assert_eq!(*arr.get(&[1, 2]).unwrap(), 6);
    *arr.get_mut(&[0, 0]).unwrap() = 10;
    assert_eq!(*arr.flat_index(0).unwrap(), 10);
    assert_eq!(
        arr.index_axis(Axis(0), 1).unwrap().to_vec_flat(),
        vec![4, 5, 6]
    );
    assert_eq!(
        arr.slice_axis(Axis(1), SliceSpec::new(0, 2))
            .unwrap()
            .to_vec_flat(),
        vec![10, 2, 4, 5]
    );
    {
        let mut sliced = arr.slice_axis_mut(Axis(0), SliceSpec::new(0, 1)).unwrap();
        sliced.as_slice_mut().unwrap()[0] = 11;
    }
    assert_eq!(*arr.get(&[0, 0]).unwrap(), 11);
    let multi = arr
        .slice_multi(&[SliceSpec::full(), SliceSpec::with_step(0, 3, 2)])
        .unwrap();
    assert_eq!(multi.to_vec_flat(), vec![11, 3, 4, 6]);
    assert_eq!(arr.insert_axis(Axis(0)).unwrap().shape(), &[1, 2, 3]);
    let squeeze_src = Array::<i32, Ix2>::from_vec(Ix2::new([1, 3]), vec![1, 2, 3]).unwrap();
    assert_eq!(squeeze_src.remove_axis(Axis(0)).unwrap().shape(), &[3]);

    let view = arr.view();
    assert_eq!(*view.get(&[0, 1]).unwrap(), 2);
    assert_eq!(
        view.index_axis(Axis(1), -1).unwrap().to_vec_flat(),
        vec![3, 6]
    );
    assert_eq!(
        view.slice_axis(Axis(0), SliceSpec::new(0, 1))
            .unwrap()
            .to_vec_flat(),
        vec![11, 2, 3]
    );
    assert_eq!(view.insert_axis(Axis(2)).unwrap().shape(), &[2, 3, 1]);
    assert_eq!(
        squeeze_src.view().remove_axis(Axis(0)).unwrap().shape(),
        &[3]
    );
    {
        let mut view_mut = arr.view_mut();
        let mut sliced = view_mut.slice_axis_mut(Axis(0), SliceSpec::full()).unwrap();
        sliced.as_slice_mut().unwrap()[1] = 22;
    }
    assert_eq!(*arr.get(&[0, 1]).unwrap(), 22);

    let selected = arr.index_select(Axis(0), &[1, 0]).unwrap();
    assert_eq!(selected.to_vec_flat(), vec![4, 5, 6, 11, 22, 3]);
    assert_eq!(
        arr.view()
            .index_select(Axis(1), &[2, 0])
            .unwrap()
            .to_vec_flat(),
        vec![3, 11, 6, 4]
    );
    let mask = Array::<bool, Ix2>::from_vec(
        Ix2::new([2, 3]),
        vec![true, false, true, false, true, false],
    )
    .unwrap();
    assert_eq!(
        arr.boolean_index(&mask).unwrap().as_slice().unwrap(),
        &[11, 3, 5]
    );
    assert_eq!(
        arr.view().boolean_index(&mask).unwrap().as_slice().unwrap(),
        &[11, 3, 5]
    );
    let flat_mask =
        Array::<bool, Ix1>::from_vec(Ix1::new([6]), vec![false, true, false, true, false, true])
            .unwrap();
    assert_eq!(
        arr.boolean_index_flat(&flat_mask)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[22, 4, 6]
    );
    arr.boolean_index_assign(&mask, -1).unwrap();
    assert_eq!(arr.as_slice().unwrap(), &[-1, 22, -1, 4, -1, 6]);
    let replacement = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![7, 8, 9]).unwrap();
    arr.boolean_index_assign_array(&mask, &replacement).unwrap();
    assert_eq!(arr.as_slice().unwrap(), &[7, 22, 8, 4, 9, 6]);

    assert_eq!(
        indexing::ravel_multi_index(&[&[0, 1], &[2, 0]], &[2, 3]).unwrap(),
        vec![2, 3]
    );
    assert_eq!(
        indexing::unravel_index(&[0, 5], &[2, 3]).unwrap(),
        vec![vec![0, 1], vec![0, 2]]
    );
    assert_eq!(
        indexing::nonzero(&Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![0, 2, 0, 3]).unwrap()),
        vec![vec![1, 3]]
    );
    let where_nonzero = indexing::argwhere(
        &Array::<i32, Ix2>::from_vec(Ix2::new([2, 2]), vec![0, 1, 2, 0]).unwrap(),
    )
    .unwrap();
    assert_eq!(where_nonzero.shape(), &[2, 2]);
    assert_eq!(where_nonzero.as_slice().unwrap(), &[0, 1, 1, 0]);
    assert_eq!(
        indexing::flatnonzero(&Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![0, 5, 6]).unwrap()),
        vec![1, 2]
    );
    assert_eq!(
        indexing::mask_indices(3, MaskKind::Triu, 0),
        vec![0, 1, 2, 4, 5, 8]
    );
    assert_eq!(
        indexing::extract(
            &Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap(),
            &Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap(),
        )
        .unwrap()
        .as_slice()
        .unwrap(),
        &[1, 3]
    );
    let mut place_target = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![0, 0, 0, 0]).unwrap();
    let place_mask =
        Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, false]).unwrap();
    indexing::place(&mut place_target, &place_mask, &[5, 6]).unwrap();
    assert_eq!(place_target.as_slice().unwrap(), &[5, 0, 6, 0]);
    indexing::putmask(&mut place_target, &place_mask, &[9]).unwrap();
    assert_eq!(place_target.as_slice().unwrap(), &[9, 0, 9, 0]);

    let iter_values: Vec<i32> = arr.iter().copied().collect();
    assert_eq!(iter_values, vec![7, 22, 8, 4, 9, 6]);
    for v in arr.iter_mut() {
        *v += 1;
    }
    assert_eq!(
        arr.flat().copied().collect::<Vec<_>>(),
        vec![8, 23, 9, 5, 10, 7]
    );
    assert_eq!(arr.indexed_iter().next().unwrap(), (vec![0, 0], &8));
    let lanes: Vec<Vec<i32>> = arr
        .lanes(Axis(1))
        .unwrap()
        .map(|lane| lane.iter().copied().collect())
        .collect();
    assert_eq!(lanes, vec![vec![8, 23, 9], vec![5, 10, 7]]);
    let rows: Vec<Vec<i32>> = arr
        .axis_iter(Axis(0))
        .unwrap()
        .map(|row| row.iter().copied().collect())
        .collect();
    assert_eq!(rows, lanes);
    for mut row in arr.axis_iter_mut(Axis(0)).unwrap() {
        row.as_slice_mut().unwrap()[0] *= 2;
    }
    assert_eq!(arr.as_slice().unwrap(), &[16, 23, 9, 10, 10, 7]);
    assert_eq!(
        arr.view().iter().copied().take(2).collect::<Vec<_>>(),
        vec![16, 23]
    );
    assert_eq!(arr.view().flat().copied().last(), Some(7));
    assert_eq!(arr.view().indexed_iter().next().unwrap(), (vec![0, 0], &16));

    let reducible = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(reducible.sum(), 21_i64);
    assert_eq!(reducible.prod(), 720_i64);
    assert_eq!(reducible.min(), Some(1));
    assert_eq!(reducible.max(), Some(6));
    assert_eq!(
        reducible.sum_axis(Axis(0)).unwrap().as_slice().unwrap(),
        &[5_i64, 7, 9]
    );
    assert_eq!(
        reducible.prod_axis(Axis(1)).unwrap().as_slice().unwrap(),
        &[6_i64, 120]
    );
    assert_eq!(
        reducible.min_axis(Axis(0)).unwrap().as_slice().unwrap(),
        &[1, 2, 3]
    );
    assert_eq!(
        reducible.max_axis(Axis(1)).unwrap().as_slice().unwrap(),
        &[3, 6]
    );
    assert_eq!(reducible.mean(), Some(3.5));
    assert_eq!(
        reducible.mean_axis(Axis(1)).unwrap().as_slice().unwrap(),
        &[2.0, 5.0]
    );
    let floats = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(floats.var(0), Some(2.0 / 3.0));
    assert!((floats.std(0).unwrap() - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    let bools = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, false]).unwrap();
    assert!(bools.any());
    assert!(!bools.all());
    assert_eq!(reducible.view().sum(), 21_i64);
    assert_eq!(reducible.view().prod(), 720_i64);
    assert_eq!(reducible.view().min(), Some(1));
    assert_eq!(reducible.view().max(), Some(6));
    assert_eq!(reducible.view().mean(), Some(3.5));

    let sorted = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![3, 1, 4, 2]).unwrap();
    assert_eq!(sorted.sorted().as_slice().unwrap(), &[1, 2, 3, 4]);
    assert_eq!(sorted.argsort().as_slice().unwrap(), &[1, 3, 0, 2]);

    let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 1]), vec![1, 2]).unwrap();
    let b = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
    assert_eq!(
        a.broadcast_to(&[2, 3]).unwrap().to_vec_flat(),
        vec![1, 1, 1, 2, 2, 2]
    );
    assert_eq!(
        a.view().broadcast_to(&[2, 3]).unwrap().to_vec_flat(),
        vec![1, 1, 1, 2, 2, 2]
    );
    assert_eq!(
        a.add_broadcast(&b).unwrap().as_slice().unwrap(),
        &[11, 21, 31, 12, 22, 32]
    );
    assert_eq!(
        a.sub_broadcast(&b).unwrap().as_slice().unwrap(),
        &[-9, -19, -29, -8, -18, -28]
    );
    assert_eq!(
        a.mul_broadcast(&b).unwrap().as_slice().unwrap(),
        &[10, 20, 30, 20, 40, 60]
    );
    assert_eq!(
        b.div_broadcast(&Array::from_vec(Ix1::new([1]), vec![10]).unwrap())
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, 2, 3]
    );
    assert_eq!(
        b.rem_broadcast(&Array::from_vec(Ix1::new([1]), vec![7]).unwrap())
            .unwrap()
            .as_slice()
            .unwrap(),
        &[3, 6, 2]
    );

    let mut inplace = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
    let rhs = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
    inplace.add_inplace(&rhs).unwrap();
    assert_eq!(inplace.as_slice().unwrap(), &[11, 22, 33]);
    inplace.sub_inplace(&rhs).unwrap();
    inplace.mul_inplace(&rhs).unwrap();
    inplace.div_inplace(&rhs).unwrap();
    inplace
        .rem_inplace(&Array::from_vec(Ix1::new([3]), vec![7, 7, 7]).unwrap())
        .unwrap();
    assert_eq!(inplace.as_slice().unwrap(), &[3, 6, 2]);

    let mut copy_dst = Array::<i32, Ix2>::zeros(Ix2::new([2, 3])).unwrap();
    ferray_core::ops::copyto(&mut copy_dst, &b).unwrap();
    assert_eq!(copy_dst.as_slice().unwrap(), &[10, 20, 30, 10, 20, 30]);
    let copy_src = Array::<i32, Ix1>::from_vec(Ix1::new([1]), vec![5]).unwrap();
    let copy_mask = Array::<bool, Ix2>::from_vec(
        Ix2::new([2, 3]),
        vec![true, false, true, false, true, false],
    )
    .unwrap();
    ferray_core::ops::copyto_where(&mut copy_dst, &copy_src, &copy_mask).unwrap();
    assert_eq!(copy_dst.as_slice().unwrap(), &[5, 20, 5, 10, 5, 30]);
    copy_dst.copy_from(&b).unwrap();
    copy_dst.copy_from_where(&copy_src, &copy_mask).unwrap();
    assert_eq!(copy_dst.as_slice().unwrap(), &[5, 20, 5, 10, 5, 30]);

    let mapped = NdIter::binary_map(&a, &a, |x, y| x + y).unwrap();
    assert_eq!(mapped.as_slice().unwrap(), &[2, 4]);
    let mixed = NdIter::binary_map_mixed(&a, &b, |x, y| x as f64 + y as f64).unwrap();
    assert_eq!(mixed.shape(), &[2, 3]);
    assert_eq!(
        mixed.as_slice().unwrap(),
        &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
    );
    let mut into = Array::<i32, IxDyn>::zeros(IxDyn::new(&[2, 1])).unwrap();
    NdIter::binary_map_into(&a, &a, &mut into, |x, y| x + y).unwrap();
    assert_eq!(into.as_slice().unwrap(), &[2, 4]);
    let unary = NdIter::unary_map(&a, |x| x * 3).unwrap();
    assert_eq!(unary.as_slice().unwrap(), &[3, 6]);
    NdIter::unary_map_into(&a, &mut into, |x| x * 4).unwrap();
    assert_eq!(into.as_slice().unwrap(), &[4, 8]);
    assert_eq!(NdIter::broadcast_shape(&[2, 1], &[3]).unwrap(), vec![2, 3]);
    let iter = NdIter::binary_iter(&a, &a).unwrap();
    assert_eq!(iter.shape(), &[2, 1]);
    assert_eq!(iter.size(), 2);
    assert_eq!(iter.map_collect(|x, y| x + y), vec![2, 4]);
    let mut seen = Vec::new();
    NdIter::binary_iter(&a, &a)
        .unwrap()
        .for_each(|x, y| seen.push(x * y));
    assert_eq!(seen, vec![1, 4]);

    let mut target = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
    {
        let mut guard = WritebackGuard::new(&mut target).unwrap();
        assert_eq!(guard.scratch().as_slice().unwrap(), &[1, 2, 3]);
        guard.scratch_mut().as_slice_mut().unwrap()[1] = 99;
        guard.commit().unwrap();
    }
    assert_eq!(target.as_slice().unwrap(), &[1, 99, 3]);
    {
        let mut guard = WritebackGuard::new(&mut target).unwrap();
        guard.scratch_mut().as_slice_mut().unwrap()[0] = -1;
        guard.discard();
    }
    assert_eq!(target.as_slice().unwrap(), &[1, 99, 3]);

    let _ = s![1..3, ..];
}

#[cfg(feature = "const_shapes")]
#[test]
fn const_shape_surface_preserves_static_shape_contracts() {
    use ferray_core::dimension::static_shape::{Shape1, Shape2, Shape3, Shape4, Shape5, Shape6};

    let s1 = Shape1::<3>::new();
    let s2 = Shape2::<2, 3>::new();
    let s3 = Shape3::<1, 2, 3>::new();
    let s4 = Shape4::<1, 1, 2, 3>::new();
    let s5 = Shape5::<1, 1, 1, 2, 3>::new();
    let s6 = Shape6::<1, 1, 1, 1, 2, 3>::new();
    assert_eq!(s1.size(), 3);
    assert_eq!(s2.size(), 6);
    assert_eq!(s3.size(), 6);
    assert_eq!(s4.size(), 6);
    assert_eq!(s5.size(), 6);
    assert_eq!(s6.size(), 6);

    let static_arr = Array::<i32, Shape2<2, 3>>::from_vec(s2, vec![1, 2, 3, 4, 5, 6]).unwrap();
    let reshaped = static_arr.static_reshape::<Shape1<6>>().unwrap();
    assert_eq!(reshaped.shape(), &[6]);
    assert_eq!(reshaped.into_dynamic_ix().shape(), &[6]);
}

#[cfg(feature = "chrono")]
#[test]
fn chrono_datetime_surface_round_trips_units() {
    use chrono::{Duration, NaiveDate, TimeZone, Utc};

    let dt = Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap();
    let ferray_dt = DateTime64::from_chrono(dt, TimeUnit::S).unwrap();
    assert_eq!(ferray_dt.to_chrono(TimeUnit::S).unwrap(), dt);
    let date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
    assert_eq!(
        DateTime64::from_naive_date(date, TimeUnit::D)
            .unwrap()
            .to_naive_date(TimeUnit::D)
            .unwrap(),
        date
    );
    assert_eq!(
        ferray_dt.to_naive_datetime(TimeUnit::S).unwrap().and_utc(),
        dt
    );
    assert_eq!(
        DateTime64::from_naive_datetime(dt.naive_utc(), TimeUnit::S).unwrap(),
        ferray_dt
    );
    let fixed = dt.with_timezone(&chrono::FixedOffset::east_opt(3600).unwrap());
    assert_eq!(
        DateTime64::from_fixed_offset(fixed, TimeUnit::S).unwrap(),
        ferray_dt
    );
    assert_eq!(
        ferray_dt
            .to_fixed_offset(TimeUnit::S, chrono::FixedOffset::east_opt(3600).unwrap())
            .unwrap()
            .timestamp(),
        dt.timestamp()
    );
    let duration = Duration::seconds(5);
    let ferray_delta = Timedelta64::from_chrono(duration, TimeUnit::S).unwrap();
    assert_eq!(ferray_delta.to_chrono(TimeUnit::S).unwrap(), duration);
}
