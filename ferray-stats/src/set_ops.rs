// ferray-stats: Set operations — union1d, intersect1d, setdiff1d, setxor1d, in1d, isin (REQ-18)

use ferray_core::error::FerrayResult;
use ferray_core::{Array, Element, Ix1};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sort and deduplicate a vector.
fn sorted_unique<T: PartialOrd + Copy>(data: &[T]) -> Vec<T> {
    let mut v: Vec<T> = data.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.dedup_by(|a, b| (*a).partial_cmp(b) == Some(std::cmp::Ordering::Equal));
    v
}

/// Collect array data into a Vec.
fn to_vec<T: Element + Copy>(a: &Array<T, Ix1>) -> Vec<T> {
    a.iter().copied().collect()
}

/// Make a 1-D result array.
fn make_1d<T: Element>(data: Vec<T>) -> FerrayResult<Array<T, Ix1>> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data)
}

// ---------------------------------------------------------------------------
// union1d
// ---------------------------------------------------------------------------

/// Return the sorted union of two 1-D arrays.
///
/// Equivalent to `numpy.union1d`.
pub fn union1d<T>(
    a: &Array<T, Ix1>,
    b: &Array<T, Ix1>,
    assume_unique: bool,
) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let av = if assume_unique {
        to_vec(a)
    } else {
        sorted_unique(&to_vec(a))
    };
    let bv = if assume_unique {
        to_vec(b)
    } else {
        sorted_unique(&to_vec(b))
    };

    // Merge two sorted arrays
    let mut result = Vec::with_capacity(av.len() + bv.len());
    let (mut i, mut j) = (0, 0);
    while i < av.len() && j < bv.len() {
        match av[i]
            .partial_cmp(&bv[j])
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Less => {
                result.push(av[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(bv[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(av[i]);
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&av[i..]);
    result.extend_from_slice(&bv[j..]);

    make_1d(result)
}

// ---------------------------------------------------------------------------
// intersect1d
// ---------------------------------------------------------------------------

/// Return the sorted intersection of two 1-D arrays.
///
/// Equivalent to `numpy.intersect1d`.
pub fn intersect1d<T>(
    a: &Array<T, Ix1>,
    b: &Array<T, Ix1>,
    assume_unique: bool,
) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let av = if assume_unique {
        to_vec(a)
    } else {
        sorted_unique(&to_vec(a))
    };
    let bv = if assume_unique {
        to_vec(b)
    } else {
        sorted_unique(&to_vec(b))
    };

    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < av.len() && j < bv.len() {
        match av[i]
            .partial_cmp(&bv[j])
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result.push(av[i]);
                i += 1;
                j += 1;
            }
        }
    }

    make_1d(result)
}

// ---------------------------------------------------------------------------
// setdiff1d
// ---------------------------------------------------------------------------

/// Return the sorted set difference of two 1-D arrays (elements in `a` not in `b`).
///
/// Equivalent to `numpy.setdiff1d`.
pub fn setdiff1d<T>(
    a: &Array<T, Ix1>,
    b: &Array<T, Ix1>,
    assume_unique: bool,
) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let av = if assume_unique {
        to_vec(a)
    } else {
        sorted_unique(&to_vec(a))
    };
    let bv = if assume_unique {
        to_vec(b)
    } else {
        sorted_unique(&to_vec(b))
    };

    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < av.len() {
        if j >= bv.len() {
            result.push(av[i]);
            i += 1;
        } else {
            match av[i]
                .partial_cmp(&bv[j])
                .unwrap_or(std::cmp::Ordering::Equal)
            {
                std::cmp::Ordering::Less => {
                    result.push(av[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
    }

    make_1d(result)
}

// ---------------------------------------------------------------------------
// setxor1d
// ---------------------------------------------------------------------------

/// Return the sorted symmetric difference of two 1-D arrays.
///
/// Elements that are in exactly one of the two arrays.
///
/// Equivalent to `numpy.setxor1d`.
pub fn setxor1d<T>(
    a: &Array<T, Ix1>,
    b: &Array<T, Ix1>,
    assume_unique: bool,
) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let av = if assume_unique {
        to_vec(a)
    } else {
        sorted_unique(&to_vec(a))
    };
    let bv = if assume_unique {
        to_vec(b)
    } else {
        sorted_unique(&to_vec(b))
    };

    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < av.len() && j < bv.len() {
        match av[i]
            .partial_cmp(&bv[j])
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Less => {
                result.push(av[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(bv[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&av[i..]);
    result.extend_from_slice(&bv[j..]);

    make_1d(result)
}

// ---------------------------------------------------------------------------
// in1d
// ---------------------------------------------------------------------------

/// Test whether each element of `a` is also present in `b`.
///
/// Returns a boolean array of the same length as `a`.
///
/// Equivalent to `numpy.in1d`.
pub fn in1d<T>(
    a: &Array<T, Ix1>,
    b: &Array<T, Ix1>,
    assume_unique: bool,
) -> FerrayResult<Array<bool, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let av = to_vec(a);
    let bv = if assume_unique {
        to_vec(b)
    } else {
        sorted_unique(&to_vec(b))
    };

    let result: Vec<bool> = av
        .iter()
        .map(|&val| {
            bv.binary_search_by(|probe| {
                probe.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)
            })
            .is_ok()
        })
        .collect();

    let n = result.len();
    Array::from_vec(Ix1::new([n]), result)
}

// ---------------------------------------------------------------------------
// isin
// ---------------------------------------------------------------------------

/// Test whether each element of `element` is in `test_elements`.
///
/// This is the same as `in1d` but named to match `numpy.isin`.
///
/// Equivalent to `numpy.isin`.
pub fn isin<T>(
    element: &Array<T, Ix1>,
    test_elements: &Array<T, Ix1>,
    assume_unique: bool,
) -> FerrayResult<Array<bool, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    in1d(element, test_elements, assume_unique)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: Vec<i32>) -> Array<i32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn test_union1d() {
        let a = arr(vec![1, 2, 3]);
        let b = arr(vec![2, 3, 4]);
        let u = union1d(&a, &b, false).unwrap();
        let data: Vec<i32> = u.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_intersect1d() {
        let a = arr(vec![1, 2, 3, 4]);
        let b = arr(vec![2, 4, 6]);
        let i = intersect1d(&a, &b, false).unwrap();
        let data: Vec<i32> = i.iter().copied().collect();
        assert_eq!(data, vec![2, 4]);
    }

    #[test]
    fn test_setdiff1d() {
        let a = arr(vec![1, 2, 3, 4]);
        let b = arr(vec![2, 4]);
        let d = setdiff1d(&a, &b, false).unwrap();
        let data: Vec<i32> = d.iter().copied().collect();
        assert_eq!(data, vec![1, 3]);
    }

    #[test]
    fn test_setxor1d() {
        let a = arr(vec![1, 2, 3]);
        let b = arr(vec![2, 3, 4]);
        let x = setxor1d(&a, &b, false).unwrap();
        let data: Vec<i32> = x.iter().copied().collect();
        assert_eq!(data, vec![1, 4]);
    }

    #[test]
    fn test_in1d() {
        let a = arr(vec![1, 2, 3, 4, 5]);
        let b = arr(vec![2, 4]);
        let r = in1d(&a, &b, false).unwrap();
        let data: Vec<bool> = r.iter().copied().collect();
        assert_eq!(data, vec![false, true, false, true, false]);
    }

    #[test]
    fn test_isin() {
        let elem = arr(vec![1, 2, 3, 4, 5]);
        let test = arr(vec![3, 5]);
        let r = isin(&elem, &test, false).unwrap();
        let data: Vec<bool> = r.iter().copied().collect();
        assert_eq!(data, vec![false, false, true, false, true]);
    }

    #[test]
    fn test_union1d_with_duplicates() {
        let a = arr(vec![3, 1, 2, 1]);
        let b = arr(vec![4, 2, 3, 2]);
        let u = union1d(&a, &b, false).unwrap();
        let data: Vec<i32> = u.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_union1d_assume_unique() {
        let a = arr(vec![1, 2, 3]);
        let b = arr(vec![2, 3, 4]);
        let u = union1d(&a, &b, true).unwrap();
        let data: Vec<i32> = u.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_setdiff1d_empty_result() {
        let a = arr(vec![1, 2, 3]);
        let b = arr(vec![1, 2, 3, 4]);
        let d = setdiff1d(&a, &b, false).unwrap();
        assert_eq!(d.size(), 0);
    }

    #[test]
    fn test_intersect1d_empty_result() {
        let a = arr(vec![1, 2, 3]);
        let b = arr(vec![4, 5, 6]);
        let i = intersect1d(&a, &b, false).unwrap();
        assert_eq!(i.size(), 0);
    }
}
