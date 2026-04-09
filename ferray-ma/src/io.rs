// ferray-ma: MaskedArray I/O via ferray-io (#509)
//
// This module is gated behind the `io` cargo feature. It wraps
// `ferray-io::npy::{save, load}` to persist a MaskedArray as a pair
// of `.npy` files:
//
//   save_masked("data.npy", "mask.npy", &ma)  -> writes both files
//   load_masked::<T, D>("data.npy", "mask.npy") -> MaskedArray<T, D>
//
// NumPy's `np.ma.dump`/`load` are pickle-based and not portable; the
// more common NumPy workflow is `np.savez(data=ma.data, mask=ma.mask)`
// plus manual reconstruction. This module provides the ferray
// equivalent as a single-call pair. The `fill_value` and `hard_mask`
// flags are NOT persisted (pickle round-trips those, but our binary
// format deliberately doesn't) — callers who need to preserve them
// should call `with_fill_value` / `harden_mask` after loading.

use std::path::Path;

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use ferray_io::npy::{NpyElement, load as npy_load, save as npy_save};

use crate::MaskedArray;

/// Save a masked array to two `.npy` files on disk.
///
/// `data_path` receives the underlying data array and `mask_path`
/// receives the boolean mask. Both files can be loaded independently
/// with `ferray_io::npy::load` if needed, or reconstructed into a
/// `MaskedArray` via [`load_masked`].
///
/// The `fill_value` and `hard_mask` flags are not persisted — only
/// the data + mask pair, which is the NumPy-compatible serialization
/// format. Callers that need to preserve those settings should re-
/// apply them after loading.
///
/// # Errors
/// Returns `FerrayError::IoError` if either file cannot be created or
/// written.
pub fn save_masked<T, D, P1, P2>(
    data_path: P1,
    mask_path: P2,
    ma: &MaskedArray<T, D>,
) -> FerrayResult<()>
where
    T: Element + NpyElement,
    D: Dimension,
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    npy_save(data_path.as_ref(), ma.data())?;
    npy_save(mask_path.as_ref(), ma.mask())?;
    Ok(())
}

/// Load a masked array from a pair of `.npy` files.
///
/// The `data_path` file's element type and shape must match the
/// requested `T` / `D` type parameters; the `mask_path` file must
/// contain a `bool` array of the identical shape. The loaded mask
/// is validated against the data shape via
/// `MaskedArray::new` — mismatches surface as
/// `FerrayError::ShapeMismatch`.
///
/// The returned `MaskedArray` has the default `fill_value`
/// (`T::zero()`) and a soft mask. Apply [`MaskedArray::with_fill_value`]
/// / [`mask_ops::harden_mask`] afterwards if you need different
/// defaults.
///
/// # Errors
/// - `FerrayError::IoError` on I/O failures for either file.
/// - `FerrayError::InvalidDtype` if the data file's dtype doesn't
///   match `T` or the mask file isn't bool.
/// - `FerrayError::ShapeMismatch` if the shapes don't line up.
pub fn load_masked<T, D, P1, P2>(
    data_path: P1,
    mask_path: P2,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + NpyElement,
    D: Dimension,
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let data: Array<T, D> = npy_load(data_path.as_ref())?;
    let mask: Array<bool, D> = npy_load(mask_path.as_ref())?;
    MaskedArray::new(data, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Array;
    use ferray_core::dimension::{Ix1, Ix2};

    fn test_dir() -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ferray_ma_io_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn test_file(name: &str) -> std::path::PathBuf {
        test_dir().join(name)
    }

    #[test]
    fn save_and_load_roundtrips_1d_f64_masked_array() {
        let d = Array::<f64, Ix1>::from_vec(
            Ix1::new([5]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let m = Array::<bool, Ix1>::from_vec(
            Ix1::new([5]),
            vec![false, true, false, false, true],
        )
        .unwrap();
        let ma = MaskedArray::new(d, m).unwrap();

        let data_path = test_file("test_1d.data.npy");
        let mask_path = test_file("test_1d.mask.npy");
        save_masked(&data_path, &mask_path, &ma).unwrap();

        let loaded: MaskedArray<f64, Ix1> = load_masked(&data_path, &mask_path).unwrap();
        assert_eq!(loaded.shape(), &[5]);
        assert_eq!(
            loaded.data().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        );
        assert_eq!(
            loaded.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false, false, true]
        );

        let _ = std::fs::remove_file(&data_path);
        let _ = std::fs::remove_file(&mask_path);
    }

    #[test]
    fn save_and_load_roundtrips_2d_i32_masked_array() {
        // Integer data type — exercises the NpyElement path for non-float.
        let d = Array::<i32, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![10, 20, 30, 40, 50, 60],
        )
        .unwrap();
        let m = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, false, true, true, false, false],
        )
        .unwrap();
        let ma = MaskedArray::new(d, m).unwrap();

        let data_path = test_file("test_2d.data.npy");
        let mask_path = test_file("test_2d.mask.npy");
        save_masked(&data_path, &mask_path, &ma).unwrap();

        let loaded: MaskedArray<i32, Ix2> = load_masked(&data_path, &mask_path).unwrap();
        assert_eq!(loaded.shape(), &[2, 3]);
        assert_eq!(
            loaded.data().iter().copied().collect::<Vec<_>>(),
            vec![10, 20, 30, 40, 50, 60]
        );
        assert_eq!(
            loaded.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, false, true, true, false, false]
        );

        let _ = std::fs::remove_file(&data_path);
        let _ = std::fs::remove_file(&mask_path);
    }

    #[test]
    fn load_masked_defaults_soft_mask_and_zero_fill() {
        // The binary format doesn't persist fill_value or hard_mask;
        // a freshly loaded MaskedArray should have the defaults.
        let d = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let m =
            Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let mut ma = MaskedArray::new(d, m).unwrap();
        // Set non-default values before save — these should NOT survive.
        ma.set_fill_value(-999.0);
        ma.harden_mask().unwrap();

        let data_path = test_file("test_defaults.data.npy");
        let mask_path = test_file("test_defaults.mask.npy");
        save_masked(&data_path, &mask_path, &ma).unwrap();

        let loaded: MaskedArray<f64, Ix1> = load_masked(&data_path, &mask_path).unwrap();
        assert_eq!(loaded.fill_value(), 0.0);
        assert!(!loaded.is_hard_mask());

        let _ = std::fs::remove_file(&data_path);
        let _ = std::fs::remove_file(&mask_path);
    }

    #[test]
    fn load_masked_rejects_mismatched_shapes() {
        // Save two files with different shapes, then try to load as
        // a masked array — the MaskedArray::new shape check should fire.
        let d = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let wrong_mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![false; 4]).unwrap();

        let data_path = test_file("test_bad_shape.data.npy");
        let mask_path = test_file("test_bad_shape.mask.npy");
        ferray_io::npy::save(&data_path, &d).unwrap();
        ferray_io::npy::save(&mask_path, &wrong_mask).unwrap();

        let result: FerrayResult<MaskedArray<f64, Ix1>> = load_masked(&data_path, &mask_path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&data_path);
        let _ = std::fs::remove_file(&mask_path);
    }

    #[test]
    fn load_masked_rejects_wrong_dtype() {
        // Save a data file as f32, try to load as f64 — should fail
        // with an InvalidDtype error from ferray_io.
        let d = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let m =
            Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();

        let data_path = test_file("test_wrong_dtype.data.npy");
        let mask_path = test_file("test_wrong_dtype.mask.npy");
        ferray_io::npy::save(&data_path, &d).unwrap();
        ferray_io::npy::save(&mask_path, &m).unwrap();

        let result: FerrayResult<MaskedArray<f64, Ix1>> = load_masked(&data_path, &mask_path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&data_path);
        let _ = std::fs::remove_file(&mask_path);
    }
}
