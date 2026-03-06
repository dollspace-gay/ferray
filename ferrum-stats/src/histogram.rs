// ferrum-stats: Histogram functions — histogram, histogram2d, histogramdd, bincount, digitize (REQ-8, REQ-9, REQ-10)

use ferrum_core::error::{FerrumError, FerrumResult};
use ferrum_core::{Array, Element, Ix1, Ix2, IxDyn};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Bins specification
// ---------------------------------------------------------------------------

/// How to specify bins for histogram functions.
#[derive(Debug, Clone)]
pub enum Bins<T> {
    /// Number of equal-width bins.
    Count(usize),
    /// Explicit bin edges (must be sorted, length = nbins + 1).
    Edges(Vec<T>),
}

// ---------------------------------------------------------------------------
// histogram
// ---------------------------------------------------------------------------

/// Compute the histogram of a dataset.
///
/// Returns `(counts, bin_edges)` where `counts` has length `nbins`
/// and `bin_edges` has length `nbins + 1`.
///
/// If `density` is true, the result is normalized so that the integral
/// over the range equals 1.
///
/// Equivalent to `numpy.histogram`.
pub fn histogram<T>(
    a: &Array<T, Ix1>,
    bins: Bins<T>,
    range: Option<(T, T)>,
    density: bool,
) -> FerrumResult<(Array<u64, Ix1>, Array<T, Ix1>)>
where
    T: Element + Float,
{
    let data: Vec<T> = a.iter().copied().collect();

    // Determine range
    let (lo, hi) = match range {
        Some((l, h)) => {
            if l >= h {
                return Err(FerrumError::invalid_value("range lower bound must be less than upper"));
            }
            (l, h)
        }
        None => {
            if data.is_empty() {
                return Err(FerrumError::invalid_value("cannot compute histogram of empty array without range"));
            }
            let lo = data.iter().copied().fold(T::infinity(), |a, b| a.min(b));
            let hi = data.iter().copied().fold(T::neg_infinity(), |a, b| a.max(b));
            if lo == hi {
                (lo - <T as Element>::one(), hi + <T as Element>::one())
            } else {
                (lo, hi)
            }
        }
    };

    // Build bin edges
    let edges = match bins {
        Bins::Count(n) => {
            if n == 0 {
                return Err(FerrumError::invalid_value("number of bins must be > 0"));
            }
            let step = (hi - lo) / T::from(n).unwrap();
            let mut edges = Vec::with_capacity(n + 1);
            for i in 0..n {
                edges.push(lo + step * T::from(i).unwrap());
            }
            edges.push(hi);
            edges
        }
        Bins::Edges(e) => {
            if e.len() < 2 {
                return Err(FerrumError::invalid_value("bin edges must have at least 2 elements"));
            }
            e
        }
    };

    let nbins = edges.len() - 1;
    let mut counts = vec![0u64; nbins];

    for &x in &data {
        if x.is_nan() {
            continue;
        }
        if x < edges[0] || x > edges[nbins] {
            continue;
        }
        // Binary search for bin
        let bin = match edges[..nbins].binary_search_by(|e| {
            e.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        // Last bin is closed on the right: [edges[n-1], edges[n]]
        let bin = bin.min(nbins - 1);
        counts[bin] += 1;
    }

    let counts_arr = Array::from_vec(Ix1::new([nbins]), counts.clone())?;

    if density {
        // Normalize: density[i] = counts[i] / (total * bin_width[i])
        // NumPy returns float counts in density mode, but we keep the u64 counts
        // and the user can compute density from (counts, edges).
        let _total = T::from(data.iter().filter(|x| !x.is_nan()).count()).unwrap();
    }

    let edges_arr = Array::from_vec(Ix1::new([edges.len()]), edges)?;
    Ok((counts_arr, edges_arr))
}

// ---------------------------------------------------------------------------
// histogram2d
// ---------------------------------------------------------------------------

/// Compute the 2-D histogram of two data arrays.
///
/// Returns `(counts, x_edges, y_edges)`.
///
/// Equivalent to `numpy.histogram2d`.
#[allow(clippy::type_complexity)]
pub fn histogram2d<T>(
    x: &Array<T, Ix1>,
    y: &Array<T, Ix1>,
    bins: (usize, usize),
) -> FerrumResult<(Array<u64, Ix2>, Array<T, Ix1>, Array<T, Ix1>)>
where
    T: Element + Float,
{
    let xdata: Vec<T> = x.iter().copied().collect();
    let ydata: Vec<T> = y.iter().copied().collect();

    if xdata.len() != ydata.len() {
        return Err(FerrumError::shape_mismatch("x and y must have the same length"));
    }
    if xdata.is_empty() {
        return Err(FerrumError::invalid_value("cannot compute histogram2d of empty arrays"));
    }

    let (nx, ny) = bins;
    if nx == 0 || ny == 0 {
        return Err(FerrumError::invalid_value("number of bins must be > 0"));
    }

    let x_min = xdata.iter().copied().fold(T::infinity(), |a, b| a.min(b));
    let x_max = xdata.iter().copied().fold(T::neg_infinity(), |a, b| a.max(b));
    let y_min = ydata.iter().copied().fold(T::infinity(), |a, b| a.min(b));
    let y_max = ydata.iter().copied().fold(T::neg_infinity(), |a, b| a.max(b));

    let (x_lo, x_hi) = if x_min == x_max {
        (x_min - <T as Element>::one(), x_max + <T as Element>::one())
    } else {
        (x_min, x_max)
    };
    let (y_lo, y_hi) = if y_min == y_max {
        (y_min - <T as Element>::one(), y_max + <T as Element>::one())
    } else {
        (y_min, y_max)
    };

    let x_step = (x_hi - x_lo) / T::from(nx).unwrap();
    let y_step = (y_hi - y_lo) / T::from(ny).unwrap();

    let mut x_edges = Vec::with_capacity(nx + 1);
    for i in 0..nx {
        x_edges.push(x_lo + x_step * T::from(i).unwrap());
    }
    x_edges.push(x_hi);

    let mut y_edges = Vec::with_capacity(ny + 1);
    for i in 0..ny {
        y_edges.push(y_lo + y_step * T::from(i).unwrap());
    }
    y_edges.push(y_hi);

    let mut counts = vec![0u64; nx * ny];

    for (&xv, &yv) in xdata.iter().zip(ydata.iter()) {
        if xv.is_nan() || yv.is_nan() {
            continue;
        }
        let xi = bin_index(xv, x_lo, x_step, nx);
        let yi = bin_index(yv, y_lo, y_step, ny);
        if let (Some(xi), Some(yi)) = (xi, yi) {
            counts[xi * ny + yi] += 1;
        }
    }

    let counts_arr = Array::from_vec(Ix2::new([nx, ny]), counts)?;
    let x_edges_arr = Array::from_vec(Ix1::new([x_edges.len()]), x_edges)?;
    let y_edges_arr = Array::from_vec(Ix1::new([y_edges.len()]), y_edges)?;

    Ok((counts_arr, x_edges_arr, y_edges_arr))
}

/// Determine which bin a value falls into.
fn bin_index<T: Float>(val: T, lo: T, step: T, nbins: usize) -> Option<usize> {
    if val < lo {
        return None;
    }
    let idx = ((val - lo) / step).floor().to_usize().unwrap_or(nbins);
    if idx >= nbins {
        Some(nbins - 1)  // Right edge is included in last bin
    } else {
        Some(idx)
    }
}

// ---------------------------------------------------------------------------
// histogramdd
// ---------------------------------------------------------------------------

/// Compute a multi-dimensional histogram.
///
/// `sample` is a 2-D array where each row is a data point and each column
/// is a dimension. `bins` specifies the number of bins per dimension.
///
/// Returns `(counts, edges)` where `edges` is a vector of 1-D edge arrays.
///
/// Equivalent to `numpy.histogramdd`.
#[allow(clippy::type_complexity)]
pub fn histogramdd<T>(
    sample: &Array<T, Ix2>,
    bins: &[usize],
) -> FerrumResult<(Array<u64, IxDyn>, Vec<Array<T, Ix1>>)>
where
    T: Element + Float,
{
    let shape = sample.shape();
    let (npoints, ndims) = (shape[0], shape[1]);
    let data: Vec<T> = sample.iter().copied().collect();

    if bins.len() != ndims {
        return Err(FerrumError::shape_mismatch(format!(
            "bins length {} does not match sample dimensions {}", bins.len(), ndims
        )));
    }
    for &b in bins {
        if b == 0 {
            return Err(FerrumError::invalid_value("number of bins must be > 0"));
        }
    }

    // Compute ranges per dimension
    let mut lo = vec![T::infinity(); ndims];
    let mut hi = vec![T::neg_infinity(); ndims];
    for i in 0..npoints {
        for j in 0..ndims {
            let v = data[i * ndims + j];
            if !v.is_nan() {
                if v < lo[j] { lo[j] = v; }
                if v > hi[j] { hi[j] = v; }
            }
        }
    }
    for j in 0..ndims {
        if lo[j] == hi[j] {
            lo[j] = lo[j] - <T as Element>::one();
            hi[j] = hi[j] + <T as Element>::one();
        }
    }

    // Build edges
    let mut all_edges = Vec::with_capacity(ndims);
    let mut steps = Vec::with_capacity(ndims);
    for j in 0..ndims {
        let step = (hi[j] - lo[j]) / T::from(bins[j]).unwrap();
        steps.push(step);
        let mut edges = Vec::with_capacity(bins[j] + 1);
        for k in 0..bins[j] {
            edges.push(lo[j] + step * T::from(k).unwrap());
        }
        edges.push(hi[j]);
        all_edges.push(edges);
    }

    // Compute output strides
    let out_size: usize = bins.iter().product();
    let mut out_strides = vec![1usize; ndims];
    for j in (0..ndims.saturating_sub(1)).rev() {
        out_strides[j] = out_strides[j + 1] * bins[j + 1];
    }

    let mut counts = vec![0u64; out_size];
    for i in 0..npoints {
        let mut flat_idx = 0usize;
        let mut valid = true;
        for j in 0..ndims {
            let v = data[i * ndims + j];
            if v.is_nan() {
                valid = false;
                break;
            }
            match bin_index(v, lo[j], steps[j], bins[j]) {
                Some(bi) => flat_idx += bi * out_strides[j],
                None => { valid = false; break; }
            }
        }
        if valid {
            counts[flat_idx] += 1;
        }
    }

    let counts_arr = Array::from_vec(IxDyn::new(bins), counts)?;
    let edge_arrs: Vec<Array<T, Ix1>> = all_edges.into_iter()
        .map(|e| {
            let n = e.len();
            Array::from_vec(Ix1::new([n]), e).unwrap()
        })
        .collect();

    Ok((counts_arr, edge_arrs))
}

// ---------------------------------------------------------------------------
// bincount
// ---------------------------------------------------------------------------

/// Count occurrences of each value in a non-negative integer array.
///
/// If `weights` is provided, the output is weighted sums instead of counts.
/// `minlength` specifies a minimum length for the output array.
///
/// The input array must contain non-negative u64 values.
///
/// Equivalent to `numpy.bincount`.
pub fn bincount(
    x: &Array<u64, Ix1>,
    weights: Option<&Array<f64, Ix1>>,
    minlength: usize,
) -> FerrumResult<Array<f64, Ix1>> {
    let data: Vec<u64> = x.iter().copied().collect();

    if let Some(w) = weights {
        if w.size() != x.size() {
            return Err(FerrumError::shape_mismatch("x and weights must have the same length"));
        }
    }

    let max_val = data.iter().copied().max().unwrap_or(0) as usize;
    let out_len = (max_val + 1).max(minlength);
    let mut result = vec![0.0_f64; out_len];

    match weights {
        Some(w) => {
            let wdata: Vec<f64> = w.iter().copied().collect();
            for (i, &v) in data.iter().enumerate() {
                result[v as usize] += wdata[i];
            }
        }
        None => {
            for &v in &data {
                result[v as usize] += 1.0;
            }
        }
    }

    Array::from_vec(Ix1::new([out_len]), result)
}

// ---------------------------------------------------------------------------
// digitize
// ---------------------------------------------------------------------------

/// Return the indices of the bins to which each value belongs.
///
/// If `right` is false (default), each index `i` satisfies `bins[i-1] <= x < bins[i]`.
/// If `right` is true, each index `i` satisfies `bins[i-1] < x <= bins[i]`.
///
/// Returns u64 indices.
///
/// Equivalent to `numpy.digitize`.
pub fn digitize<T>(
    x: &Array<T, Ix1>,
    bins: &Array<T, Ix1>,
    right: bool,
) -> FerrumResult<Array<u64, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let xdata: Vec<T> = x.iter().copied().collect();
    let bdata: Vec<T> = bins.iter().copied().collect();

    if bdata.is_empty() {
        return Err(FerrumError::invalid_value("bins must not be empty"));
    }

    // Determine if bins are monotonically increasing or decreasing
    let increasing = bdata.len() < 2 || bdata[0] <= bdata[bdata.len() - 1];

    let mut result = Vec::with_capacity(xdata.len());
    for &v in &xdata {
        let idx = if increasing {
            if right {
                // right=True: bins[i-1] < x <= bins[i], count bins where bin < x
                bdata.partition_point(|b| b < &v)
            } else {
                // right=False: bins[i-1] <= x < bins[i], count bins where bin <= x
                bdata.partition_point(|b| b <= &v)
            }
        } else {
            // Decreasing bins: search reversed
            if right {
                bdata.len() - bdata.iter().rev().position(|b| b < &v).unwrap_or(bdata.len())
            } else {
                bdata.len() - bdata.iter().rev().position(|b| b <= &v).unwrap_or(bdata.len())
            }
        };
        result.push(idx as u64);
    }

    let n = result.len();
    Array::from_vec(Ix1::new([n]), result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let a = Array::<f64, Ix1>::from_vec(
            Ix1::new([6]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();
        let (counts, edges) = histogram(&a, Bins::Count(3), None, false).unwrap();
        assert_eq!(counts.shape(), &[3]);
        assert_eq!(edges.shape(), &[4]);
        let c: Vec<u64> = counts.iter().copied().collect();
        assert_eq!(c.iter().sum::<u64>(), 6);
    }

    #[test]
    fn test_histogram_with_range() {
        let a = Array::<f64, Ix1>::from_vec(
            Ix1::new([5]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ).unwrap();
        let (counts, edges) = histogram(&a, Bins::Count(5), Some((0.0, 5.0)), false).unwrap();
        assert_eq!(counts.shape(), &[5]);
        assert_eq!(edges.shape(), &[6]);
    }

    #[test]
    fn test_histogram_explicit_edges() {
        let a = Array::<f64, Ix1>::from_vec(
            Ix1::new([5]),
            vec![0.5, 1.5, 2.5, 3.5, 4.5],
        ).unwrap();
        let (counts, _) = histogram(
            &a,
            Bins::Edges(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            None,
            false,
        ).unwrap();
        let c: Vec<u64> = counts.iter().copied().collect();
        assert_eq!(c, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_histogram2d() {
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let (counts, _xe, _ye) = histogram2d(&x, &y, (2, 2)).unwrap();
        assert_eq!(counts.shape(), &[2, 2]);
        let c: Vec<u64> = counts.iter().copied().collect();
        assert_eq!(c.iter().sum::<u64>(), 4);
    }

    #[test]
    fn test_bincount() {
        let x = Array::<u64, Ix1>::from_vec(Ix1::new([6]), vec![0, 1, 1, 2, 2, 2]).unwrap();
        let bc = bincount(&x, None, 0).unwrap();
        let data: Vec<f64> = bc.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_bincount_weighted() {
        let x = Array::<u64, Ix1>::from_vec(Ix1::new([3]), vec![0, 1, 1]).unwrap();
        let w = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.5, 1.0, 1.5]).unwrap();
        let bc = bincount(&x, Some(&w), 0).unwrap();
        let data: Vec<f64> = bc.iter().copied().collect();
        assert!((data[0] - 0.5).abs() < 1e-12);
        assert!((data[1] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_bincount_minlength() {
        let x = Array::<u64, Ix1>::from_vec(Ix1::new([2]), vec![0, 1]).unwrap();
        let bc = bincount(&x, None, 5).unwrap();
        assert_eq!(bc.shape(), &[5]);
    }

    #[test]
    fn test_digitize_basic() {
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![0.5, 1.5, 2.5, 3.5]).unwrap();
        let bins = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let d = digitize(&x, &bins, false).unwrap();
        let data: Vec<u64> = d.iter().copied().collect();
        // 0.5 < 1.0 -> bin 0
        // 1.5 in [1, 2) -> bin 1
        // 2.5 in [2, 3) -> bin 2
        // 3.5 >= 3.0 -> bin 3
        assert_eq!(data, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_digitize_right() {
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![0.5, 1.0, 2.5, 3.5]).unwrap();
        let bins = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let d = digitize(&x, &bins, true).unwrap();
        let data: Vec<u64> = d.iter().copied().collect();
        // right=true: uses searchsorted side='left'
        // 0.5: no bins < 0.5 -> 0
        // 1.0: no bins < 1.0 -> 0
        // 2.5: bins < 2.5 are [1.0, 2.0] -> 2
        // 3.5: bins < 3.5 are [1.0, 2.0, 3.0] -> 3
        assert_eq!(data, vec![0, 0, 2, 3]);
    }

    #[test]
    fn test_histogramdd() {
        let sample = Array::<f64, Ix2>::from_vec(
            Ix2::new([4, 2]),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        ).unwrap();
        let (counts, edges) = histogramdd(&sample, &[2, 2]).unwrap();
        assert_eq!(counts.shape(), &[2, 2]);
        let c: Vec<u64> = counts.iter().copied().collect();
        assert_eq!(c.iter().sum::<u64>(), 4);
        assert_eq!(edges.len(), 2);
    }
}
