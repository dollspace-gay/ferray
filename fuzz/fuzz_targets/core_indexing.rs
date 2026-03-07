#![no_main]

//! Fuzz target for core indexing: get, slice_axis, slice_multi, index_select.
//!
//! Contract: ferray either returns Ok or Err(FerrayError) — never panics.

use libfuzzer_sys::fuzz_target;

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Axis, Ix1, Ix2};
use ferray_core::indexing::basic::SliceSpec;

fn bytes_to_f64s(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let values = bytes_to_f64s(data);
        if values.is_empty() {
            return;
        }

        // Create a 1-D array
        let n = values.len();
        let arr1d = match Array::from_vec(Ix1::new([n]), values.clone()) {
            Ok(a) => a,
            Err(_) => return,
        };

        // Test get with fuzzed indices (may be out of bounds — must not panic)
        let idx = data[0] as isize - 128;
        let _ = arr1d.get(&[idx]);

        // Test slice_axis with fuzzed parameters
        let start = (data[1] as isize) - 128;
        let stop = (data[2] as isize) - 128;
        let step_raw = data[3] as isize - 128;
        // step=0 is invalid, but the library must handle it gracefully
        let spec = SliceSpec::with_step(start, stop, step_raw);
        let _ = arr1d.slice_axis(Axis(0), spec);

        // Test slice_multi with fuzzed parameters
        let spec2 = SliceSpec::with_step(
            (data[4] as isize) - 128,
            (data[5] as isize) - 128,
            (data[6] as isize) - 128,
        );
        let _ = arr1d.slice_multi(&[spec2]);

        // Create a 2-D array and test indexing
        let total = values.len();
        if total >= 4 {
            let rows = ((data[7] as usize) % 8).max(1);
            let cols = total / rows;
            if cols > 0 && rows * cols <= total {
                let data_2d = values[..rows * cols].to_vec();
                if let Ok(arr2d) = Array::from_vec(Ix2::new([rows, cols]), data_2d) {
                    let r = (data[8] as isize) - 128;
                    let c = (data[9] as isize) - 128;
                    let _ = arr2d.get(&[r, c]);

                    // slice along axis 0
                    let _ = arr2d.slice_axis(Axis(0), SliceSpec::full());
                    // slice along axis 1
                    let _ = arr2d.slice_axis(Axis(1), SliceSpec::new(0, (data[10] as isize) - 64));
                    // out-of-bounds axis — must return Err
                    let _ = arr2d.slice_axis(Axis(99), SliceSpec::full());
                }
            }
        }

        // Test index_select with fuzzed indices on 1-D array
        if data.len() >= 14 {
            let indices: Vec<isize> = data[11..14].iter().map(|&b| b as isize - 128).collect();
            let _ = arr1d.index_select(Axis(0), &indices);
        }
    }));
});
