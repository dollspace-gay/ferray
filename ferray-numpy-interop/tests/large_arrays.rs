//! Large-array conversion tests (#313).
//!
//! The existing test corpus uses 3-7 element arrays. Add 10k+ tests
//! covering Arrow and Polars round-trips, F-contiguous handling for
//! large 2-D matrices, and bit-exact float preservation across a
//! large range. These exercise edge cases that small inputs miss:
//! buffer-allocation paths, copy hot loops, and any internal chunking
//! that activates only above a threshold.

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
use ferray_core::dimension::{Ix1, Ix2};

const LARGE: usize = 10_000;
const VERY_LARGE: usize = 100_000;

#[cfg(feature = "arrow")]
mod arrow_large {
    use super::*;
    use ferray_numpy_interop::{FromArrow, FromArrowBool, ToArrow, ToArrowBool};

    #[test]
    fn arrow_i32_10k_roundtrip() {
        let data: Vec<i32> = (0..LARGE as i32).collect();
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([LARGE]), data.clone()).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        assert_eq!(arrow_arr.len(), LARGE);
        let back: Array<i32, Ix1> = arrow_arr.to_ferray().unwrap();
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn arrow_f64_100k_roundtrip_bit_exact() {
        // Sweep across many float magnitudes; preserve every bit.
        let data: Vec<f64> = (0..VERY_LARGE)
            .map(|i| (i as f64).sin() * 1e6 + (i as f64) * 1e-9)
            .collect();
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([VERY_LARGE]), data.clone()).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        let back: Array<f64, Ix1> = arrow_arr.to_ferray().unwrap();
        for (got, want) in back.as_slice().unwrap().iter().zip(data.iter()) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }

    #[test]
    fn arrow_bool_50k_roundtrip() {
        // Bit-packed Arrow BooleanArray has internal chunking; 50k
        // exercises the multi-byte path with mixed bit patterns.
        let n = 50_000;
        let data: Vec<bool> = (0..n).map(|i| i % 7 < 3).collect();
        let arr = Array::<bool, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        let back: Array<bool, Ix1> = arrow_arr.to_ferray_bool().unwrap();
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn arrow_2d_100x100_flattens_correctly() {
        let n = 100;
        let data: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([n, n]), data.clone()).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        assert_eq!(arrow_arr.len(), n * n);
        let values: Vec<f64> = arrow_arr.values().iter().copied().collect();
        // Row-major flatten preserves the linear sequence.
        assert_eq!(values, data);
    }
}

#[cfg(feature = "polars")]
mod polars_large {
    use super::*;
    use ferray_numpy_interop::{FromPolars, ToPolars};

    #[test]
    fn polars_i64_10k_roundtrip() {
        let data: Vec<i64> = (0..LARGE as i64).collect();
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([LARGE]), data.clone()).unwrap();
        let series = arr.to_polars_series("col").unwrap();
        assert_eq!(series.len(), LARGE);
        let back: Array<i64, Ix1> = series.into_ferray().unwrap();
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn polars_f64_100k_roundtrip_bit_exact() {
        let data: Vec<f64> = (0..VERY_LARGE)
            .map(|i| (i as f64).cos() * 1e3 + (i as f64) * 1e-12)
            .collect();
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([VERY_LARGE]), data.clone()).unwrap();
        let series = arr.to_polars_series("col").unwrap();
        let back: Array<f64, Ix1> = series.into_ferray().unwrap();
        for (got, want) in back.as_slice().unwrap().iter().zip(data.iter()) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }
}
