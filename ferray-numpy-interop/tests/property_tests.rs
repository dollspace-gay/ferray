//! Property tests for ferray-numpy-interop conversions (#312).
//!
//! Round-trip properties on Arrow and Polars paths: ferray → external
//! → ferray must preserve every value in shape and order. proptest was
//! a dev-dependency but unused; this file exercises it.

#![cfg(any(feature = "arrow", feature = "polars"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use proptest::prelude::*;

#[cfg(feature = "arrow")]
mod arrow_props {
    use super::*;
    use ferray_numpy_interop::{FromArrow, FromArrowBool, ToArrow, ToArrowBool};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        // Arrow integer round-trips.

        #[test]
        fn prop_arrow_i32_roundtrip(data in proptest::collection::vec(any::<i32>(), 0..=100)) {
            let n = data.len();
            let arr = Array::<i32, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let arrow_arr = arr.to_arrow().unwrap();
            let back: Array<i32, Ix1> = arrow_arr.to_ferray().unwrap();
            prop_assert_eq!(back.as_slice().unwrap(), &data[..]);
        }

        #[test]
        fn prop_arrow_i64_roundtrip(data in proptest::collection::vec(any::<i64>(), 0..=100)) {
            let n = data.len();
            let arr = Array::<i64, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let arrow_arr = arr.to_arrow().unwrap();
            let back: Array<i64, Ix1> = arrow_arr.to_ferray().unwrap();
            prop_assert_eq!(back.as_slice().unwrap(), &data[..]);
        }

        #[test]
        fn prop_arrow_u8_roundtrip(data in proptest::collection::vec(any::<u8>(), 0..=100)) {
            let n = data.len();
            let arr = Array::<u8, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let arrow_arr = arr.to_arrow().unwrap();
            let back: Array<u8, Ix1> = arrow_arr.to_ferray().unwrap();
            prop_assert_eq!(back.as_slice().unwrap(), &data[..]);
        }

        #[test]
        fn prop_arrow_u32_roundtrip(data in proptest::collection::vec(any::<u32>(), 0..=100)) {
            let n = data.len();
            let arr = Array::<u32, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let arrow_arr = arr.to_arrow().unwrap();
            let back: Array<u32, Ix1> = arrow_arr.to_ferray().unwrap();
            prop_assert_eq!(back.as_slice().unwrap(), &data[..]);
        }

        // Float round-trips: bit-exact via to_bits comparison so NaN
        // and ±Inf round-trip too.

        #[test]
        fn prop_arrow_f64_roundtrip(data in proptest::collection::vec(any::<f64>(), 0..=100)) {
            let n = data.len();
            let arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let arrow_arr = arr.to_arrow().unwrap();
            let back: Array<f64, Ix1> = arrow_arr.to_ferray().unwrap();
            for (got, want) in back.as_slice().unwrap().iter().zip(data.iter()) {
                prop_assert_eq!(got.to_bits(), want.to_bits());
            }
        }

        #[test]
        fn prop_arrow_f32_roundtrip(data in proptest::collection::vec(any::<f32>(), 0..=100)) {
            let n = data.len();
            let arr = Array::<f32, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let arrow_arr = arr.to_arrow().unwrap();
            let back: Array<f32, Ix1> = arrow_arr.to_ferray().unwrap();
            for (got, want) in back.as_slice().unwrap().iter().zip(data.iter()) {
                prop_assert_eq!(got.to_bits(), want.to_bits());
            }
        }

        #[test]
        fn prop_arrow_bool_roundtrip(
            data in proptest::collection::vec(any::<bool>(), 0..=100)
        ) {
            let n = data.len();
            let arr = Array::<bool, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let arrow_arr = arr.to_arrow().unwrap();
            let back: Array<bool, Ix1> = arrow_arr.to_ferray_bool().unwrap();
            prop_assert_eq!(back.as_slice().unwrap(), &data[..]);
        }

        // Length is preserved across round-trip even when zero.

        #[test]
        fn prop_arrow_length_preserved(data in proptest::collection::vec(any::<i32>(), 0..=100)) {
            let n = data.len();
            let arr = Array::<i32, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
            let arrow_arr = arr.to_arrow().unwrap();
            prop_assert_eq!(arrow_arr.len(), n);
        }
    }
}

#[cfg(feature = "polars")]
mod polars_props {
    use super::*;
    use ferray_numpy_interop::{FromPolars, FromPolarsBool, ToPolars, ToPolarsBool};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn prop_polars_i32_roundtrip(data in proptest::collection::vec(any::<i32>(), 1..=100)) {
            let n = data.len();
            let arr = Array::<i32, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let series = arr.to_polars_series("col").unwrap();
            let back: Array<i32, Ix1> = series.into_ferray().unwrap();
            prop_assert_eq!(back.as_slice().unwrap(), &data[..]);
        }

        #[test]
        fn prop_polars_i64_roundtrip(data in proptest::collection::vec(any::<i64>(), 1..=100)) {
            let n = data.len();
            let arr = Array::<i64, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let series = arr.to_polars_series("col").unwrap();
            let back: Array<i64, Ix1> = series.into_ferray().unwrap();
            prop_assert_eq!(back.as_slice().unwrap(), &data[..]);
        }

        #[test]
        fn prop_polars_f64_roundtrip(data in proptest::collection::vec(any::<f64>(), 1..=100)) {
            let n = data.len();
            // Polars uses NaN-canonical comparisons; filter out NaN
            // inputs to keep the round-trip assertion strict.
            let data: Vec<f64> = data.into_iter().map(|v| if v.is_nan() { 0.0 } else { v }).collect();
            let arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let series = arr.to_polars_series("col").unwrap();
            let back: Array<f64, Ix1> = series.into_ferray().unwrap();
            for (got, want) in back.as_slice().unwrap().iter().zip(data.iter()) {
                prop_assert_eq!(got.to_bits(), want.to_bits());
            }
        }

        #[test]
        fn prop_polars_bool_roundtrip(
            data in proptest::collection::vec(any::<bool>(), 1..=100)
        ) {
            let n = data.len();
            let arr = Array::<bool, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            let series = arr.to_polars_series("col").unwrap();
            let back = series.into_ferray_bool().unwrap();
            prop_assert_eq!(back.as_slice().unwrap(), &data[..]);
        }
    }
}
