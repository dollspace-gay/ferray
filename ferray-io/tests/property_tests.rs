// Property-based tests for ferray-io
//
// Tests roundtrip invariants of npy binary I/O and text I/O using proptest.

// Property tests sample integer sizes and lift them into f64 / array
// shape values; roundtrip invariants assert exact float equality.
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

use ferray_io::npy::{load_from_reader, save_to_writer};
use ferray_io::text::{SaveTxtOptions, genfromtxt_from_str, loadtxt_from_str, savetxt_to_writer};

use proptest::prelude::*;
use std::io::Cursor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn arr1_f64(data: Vec<f64>) -> Array<f64, Ix1> {
    let n = data.len();
    Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_f32(data: Vec<f32>) -> Array<f32, Ix1> {
    let n = data.len();
    Array::<f32, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_i64(data: Vec<i64>) -> Array<i64, Ix1> {
    let n = data.len();
    Array::<i64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_i32(data: Vec<i32>) -> Array<i32, Ix1> {
    let n = data.len();
    Array::<i32, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_u8(data: Vec<u8>) -> Array<u8, Ix1> {
    let n = data.len();
    Array::<u8, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_bool(data: Vec<bool>) -> Array<bool, Ix1> {
    let n = data.len();
    Array::<bool, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_u16(data: Vec<u16>) -> Array<u16, Ix1> {
    let n = data.len();
    Array::<u16, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_u32(data: Vec<u32>) -> Array<u32, Ix1> {
    let n = data.len();
    Array::<u32, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_u64(data: Vec<u64>) -> Array<u64, Ix1> {
    let n = data.len();
    Array::<u64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_u128(data: Vec<u128>) -> Array<u128, Ix1> {
    let n = data.len();
    Array::<u128, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_i16(data: Vec<i16>) -> Array<i16, Ix1> {
    let n = data.len();
    Array::<i16, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr1_i128(data: Vec<i128>) -> Array<i128, Ix1> {
    let n = data.len();
    Array::<i128, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. npy f64 roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn prop_npy_f64_roundtrip(data in proptest::collection::vec(-1e6f64..1e6, 0..=100)) {
        let arr = arr1_f64(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<f64, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    // -----------------------------------------------------------------------
    // 2. npy f32 roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn prop_npy_f32_roundtrip(data in proptest::collection::vec(-1e6f32..1e6, 0..=100)) {
        let arr = arr1_f32(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<f32, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    // -----------------------------------------------------------------------
    // 3. npy i64 roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn prop_npy_i64_roundtrip(data in proptest::collection::vec(any::<i64>(), 0..=100)) {
        let arr = arr1_i64(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<i64, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    // -----------------------------------------------------------------------
    // 4. npy i32 roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn prop_npy_i32_roundtrip(data in proptest::collection::vec(any::<i32>(), 0..=100)) {
        let arr = arr1_i32(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<i32, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    // -----------------------------------------------------------------------
    // 5. npy u8 roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn prop_npy_u8_roundtrip(data in proptest::collection::vec(any::<u8>(), 0..=100)) {
        let arr = arr1_u8(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<u8, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    // -----------------------------------------------------------------------
    // 6. npy bool roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn prop_npy_bool_roundtrip(data in proptest::collection::vec(any::<bool>(), 0..=100)) {
        let arr = arr1_bool(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<bool, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    // -----------------------------------------------------------------------
    // 7. npy 2D f64 roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn prop_npy_2d_roundtrip(
        rows in 1usize..=10,
        cols in 1usize..=10,
        seed in proptest::collection::vec(-1e6f64..1e6, 100..=100),
    ) {
        let n = rows * cols;
        let data: Vec<f64> = seed.into_iter().cycle().take(n).collect();
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data.clone()).unwrap();
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<f64, Ix2> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.shape(), &[rows, cols]);
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    // -----------------------------------------------------------------------
    // 8. npy preserves shape
    // -----------------------------------------------------------------------
    #[test]
    fn prop_npy_preserves_shape(
        rows in 1usize..=10,
        cols in 1usize..=10,
    ) {
        let n = rows * cols;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap();
        let shape_before = arr.shape().to_vec();
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<f64, Ix2> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.shape(), &shape_before[..]);
    }

    // -----------------------------------------------------------------------
    // 9. text f64 roundtrip (integer-valued to avoid formatting issues)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_text_f64_roundtrip(
        rows in 1usize..=8,
        cols in 1usize..=8,
        vals in proptest::collection::vec(-1000i32..1000, 64..=64),
    ) {
        let n = rows * cols;
        let data: Vec<f64> = vals.into_iter().cycle().take(n).map(f64::from).collect();
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data.clone()).unwrap();
        let opts = SaveTxtOptions {
            delimiter: ',',
            ..Default::default()
        };
        let mut buf = Vec::new();
        savetxt_to_writer(&mut buf, &arr, &opts).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let loaded: Array<f64, Ix2> = loadtxt_from_str(&text, ',', 0).unwrap();
        prop_assert_eq!(loaded.shape(), &[rows, cols]);
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    // -----------------------------------------------------------------------
    // 10. text delimiter invariance (comma vs tab produce same data)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_text_delimiter_invariance(
        rows in 1usize..=6,
        cols in 1usize..=6,
        vals in proptest::collection::vec(-500i32..500, 36..=36),
    ) {
        let n = rows * cols;
        let data: Vec<f64> = vals.into_iter().cycle().take(n).map(f64::from).collect();
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap();

        // Save with comma, load with comma
        let comma_opts = SaveTxtOptions {
            delimiter: ',',
            ..Default::default()
        };
        let mut comma_buf = Vec::new();
        savetxt_to_writer(&mut comma_buf, &arr, &comma_opts).unwrap();
        let comma_text = String::from_utf8(comma_buf).unwrap();
        let comma_loaded: Array<f64, Ix2> = loadtxt_from_str(&comma_text, ',', 0).unwrap();

        // Save with tab, load with tab
        let tab_opts = SaveTxtOptions {
            delimiter: '\t',
            ..Default::default()
        };
        let mut tab_buf = Vec::new();
        savetxt_to_writer(&mut tab_buf, &arr, &tab_opts).unwrap();
        let tab_text = String::from_utf8(tab_buf).unwrap();
        let tab_loaded: Array<f64, Ix2> = loadtxt_from_str(&tab_text, '\t', 0).unwrap();

        prop_assert_eq!(
            comma_loaded.as_slice().unwrap(),
            tab_loaded.as_slice().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // 11. genfromtxt fills missing values with the fill value
    // -----------------------------------------------------------------------
    #[test]
    fn prop_genfromtxt_fills_missing(
        good_vals in proptest::collection::vec(1i32..100, 2..=4),
        fill_val in -9999.0f64..-9990.0,
    ) {
        // Build a 2-row CSV where the second row has some "NA" entries
        let ncols = good_vals.len();
        let row1: Vec<String> = good_vals.iter().map(std::string::ToString::to_string).collect();
        let mut row2: Vec<String> = good_vals.iter().map(|v| (v + 1).to_string()).collect();
        // Replace the first column with NA
        row2[0] = "NA".to_string();

        let content = format!("{}\n{}\n", row1.join(","), row2.join(","));
        let arr = genfromtxt_from_str(&content, ',', fill_val, 0, &["NA"]).unwrap();
        let slice = arr.as_slice().unwrap();

        // First row: all good values
        for (i, &v) in good_vals.iter().enumerate() {
            prop_assert!(
                (slice[i] - f64::from(v)).abs() < 1e-10,
                "row 0 col {}: expected {}, got {}",
                i, v, slice[i]
            );
        }
        // Second row, first col: should be fill_val
        prop_assert!(
            (slice[ncols] - fill_val).abs() < 1e-10,
            "missing value not filled: expected {}, got {}",
            fill_val, slice[ncols]
        );
    }

    // -----------------------------------------------------------------------
    // 12. npy empty array roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn prop_npy_empty_array(_dummy in 0u8..1) {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<f64, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.shape(), &[0usize]);
        prop_assert!(loaded.as_slice().unwrap().is_empty());
    }

    // -----------------------------------------------------------------------
    // 13-18. npy roundtrips for the remaining integer dtypes (#242).
    // u8/i32/i64 already covered above; add u16, u32, u64, u128, i16, i128.
    // -----------------------------------------------------------------------

    #[test]
    fn prop_npy_u16_roundtrip(data in proptest::collection::vec(any::<u16>(), 0..=100)) {
        let arr = arr1_u16(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<u16, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn prop_npy_u32_roundtrip(data in proptest::collection::vec(any::<u32>(), 0..=100)) {
        let arr = arr1_u32(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<u32, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn prop_npy_u64_roundtrip(data in proptest::collection::vec(any::<u64>(), 0..=100)) {
        let arr = arr1_u64(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<u64, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn prop_npy_u128_roundtrip(data in proptest::collection::vec(any::<u128>(), 0..=100)) {
        let arr = arr1_u128(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<u128, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn prop_npy_i16_roundtrip(data in proptest::collection::vec(any::<i16>(), 0..=100)) {
        let arr = arr1_i16(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<i16, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn prop_npy_i128_roundtrip(data in proptest::collection::vec(any::<i128>(), 0..=100)) {
        let arr = arr1_i128(data.clone());
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let mut cursor = Cursor::new(&buf);
        let loaded: Array<i128, Ix1> = load_from_reader(&mut cursor).unwrap();
        prop_assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }
}
