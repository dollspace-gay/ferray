//! Compact storage backend for string arrays (#736).
//!
//! Mirrors Apache Arrow's `StringArray` layout: one contiguous
//! `Vec<u8>` of UTF-8 byte content, plus a `Vec<usize>` of offsets
//! where `offsets[i]` is the byte position of the i-th element's
//! start (and `offsets[N]` is the total length). The current
//! [`crate::StringArray`] backend stores one `String` per element,
//! costing 24 bytes (Vec header) + heap allocation per string —
//! cache-hostile for arrays with many short strings.
//!
//! The compact layout's per-element overhead is `sizeof::<usize>()`
//! (8 bytes on 64-bit) rather than 24, plus a single shared
//! allocation for all character data. Per-element string slices
//! are produced lazily via [`CompactStringArray::as_str`] /
//! [`CompactStringArray::iter`].
//!
//! Trade-offs vs the owned-`String` backend:
//!
//! - **Memory**: ~3× smaller for typical short-string workloads.
//! - **Cache locality**: full sequential traversal touches one
//!   contiguous buffer.
//! - **Mutation**: not supported — modifying any element would
//!   require rebuilding both buffers. Use the owned-`String`
//!   [`crate::StringArray`] when individual writes matter.
//! - **Zero-copy interop**: callers with an existing Arrow-style
//!   buffer can construct a `CompactStringArray` without
//!   re-allocating via [`CompactStringArray::from_raw_parts`].
//!
//! Behind the `compact-storage` feature flag while we evaluate
//! whether to make this the default backend.

use ferray_core::error::{FerrayError, FerrayResult};

use crate::string_array::{StringArray, StringArray1};

/// Compact, Arrow-style string array (#736).
///
/// `offsets` has length `len + 1`; `offsets[i]..offsets[i+1]` is
/// the byte slice of the i-th string in `data`. The buffers are
/// always self-consistent (the constructor enforces UTF-8 validity
/// of each substring on the way in).
#[derive(Debug, Clone)]
pub struct CompactStringArray {
    data: Vec<u8>,
    offsets: Vec<usize>,
}

impl CompactStringArray {
    /// Build from any iterator of `&str`. Allocates two buffers and
    /// fills them in one pass.
    #[must_use]
    pub fn from_iter_str<'a, I>(strs: I) -> Self
    where
        I: IntoIterator<Item = &'a str>,
    {
        let iter = strs.into_iter();
        let (lower, _) = iter.size_hint();
        let mut data = Vec::with_capacity(lower * 8);
        let mut offsets = Vec::with_capacity(lower + 1);
        offsets.push(0);
        for s in iter {
            data.extend_from_slice(s.as_bytes());
            offsets.push(data.len());
        }
        Self { data, offsets }
    }

    /// Build from a slice of `&str`. Convenience wrapper over
    /// [`from_iter_str`].
    #[must_use]
    pub fn from_strs(strs: &[&str]) -> Self {
        Self::from_iter_str(strs.iter().copied())
    }

    /// Construct from already-prepared buffers.
    ///
    /// The supplied `data` must be valid UTF-8 across each
    /// `offsets[i]..offsets[i+1]` slice. The function checks this
    /// invariant on the way in — invalid input fails with
    /// [`FerrayError::InvalidValue`].
    ///
    /// # Errors
    /// - `FerrayError::InvalidValue` if `offsets` is empty,
    ///   `offsets[0] != 0`, the offsets are not non-decreasing,
    ///   `offsets.last() > data.len()`, or any element-slice is
    ///   not valid UTF-8.
    pub fn from_raw_parts(data: Vec<u8>, offsets: Vec<usize>) -> FerrayResult<Self> {
        if offsets.is_empty() {
            return Err(FerrayError::invalid_value(
                "CompactStringArray: offsets must have at least one element (the leading 0)",
            ));
        }
        if offsets[0] != 0 {
            return Err(FerrayError::invalid_value(
                "CompactStringArray: offsets[0] must be 0",
            ));
        }
        for w in offsets.windows(2) {
            if w[1] < w[0] {
                return Err(FerrayError::invalid_value(
                    "CompactStringArray: offsets must be non-decreasing",
                ));
            }
        }
        let total = *offsets.last().unwrap();
        if total > data.len() {
            return Err(FerrayError::invalid_value(format!(
                "CompactStringArray: trailing offset {total} exceeds data length {}",
                data.len()
            )));
        }
        for w in offsets.windows(2) {
            if std::str::from_utf8(&data[w[0]..w[1]]).is_err() {
                return Err(FerrayError::invalid_value(format!(
                    "CompactStringArray: bytes [{}, {}] are not valid UTF-8",
                    w[0], w[1]
                )));
            }
        }
        Ok(Self { data, offsets })
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Whether the array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total byte length of all string data combined.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        *self.offsets.last().unwrap_or(&0)
    }

    /// Read-only slice of the i-th element. Returns `None` when
    /// `i >= len()`.
    #[must_use]
    pub fn as_str(&self, i: usize) -> Option<&str> {
        if i >= self.len() {
            return None;
        }
        let lo = self.offsets[i];
        let hi = self.offsets[i + 1];
        // SAFETY: validated UTF-8 at construction time.
        Some(unsafe { std::str::from_utf8_unchecked(&self.data[lo..hi]) })
    }

    /// Iterate over `&str` slices.
    pub fn iter(&self) -> CompactStringIter<'_> {
        CompactStringIter { arr: self, pos: 0 }
    }

    /// Borrow the underlying byte buffer.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Borrow the offsets buffer (length = `len() + 1`).
    #[must_use]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Convert to the owned-`String` backend
    /// ([`crate::StringArray1`]).
    ///
    /// # Errors
    /// Returns errors only on internal array construction failure.
    pub fn to_string_array(&self) -> FerrayResult<StringArray1> {
        let data: Vec<String> = self.iter().map(String::from).collect();
        let n = data.len();
        StringArray::<ferray_core::dimension::Ix1>::from_vec(
            ferray_core::dimension::Ix1::new([n]),
            data,
        )
    }

    /// Build from the owned-`String` backend.
    #[must_use]
    pub fn from_string_array<D: ferray_core::Dimension>(arr: &StringArray<D>) -> Self {
        Self::from_iter_str(arr.iter().map(String::as_str))
    }

    /// Memory-efficiency comparison: estimated heap bytes used.
    /// Useful for benchmarks. Matches `Vec<String>` accounting:
    /// each `String` costs `size_of::<String>` (24 on 64-bit) plus
    /// its heap allocation (capacity bytes); compact uses one
    /// allocation totaling `total_bytes()` plus
    /// `(len + 1) * size_of::<usize>()` for offsets.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        self.data.capacity()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
            + std::mem::size_of::<Self>()
    }
}

/// Iterator over `&str` slices. Returned by
/// [`CompactStringArray::iter`].
pub struct CompactStringIter<'a> {
    arr: &'a CompactStringArray,
    pos: usize,
}

impl<'a> Iterator for CompactStringIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        let s = self.arr.as_str(self.pos)?;
        self.pos += 1;
        Some(s)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.arr.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CompactStringIter<'_> {}

/// Estimate the heap cost of a `Vec<String>` for the given strings.
/// Uses `size_of::<String>` (header) plus capacity bytes per
/// element. Matches the accounting [`CompactStringArray::estimated_bytes`]
/// uses for its own report.
#[must_use]
pub fn estimated_string_array_bytes(strs: &[String]) -> usize {
    let header = std::mem::size_of_val(strs);
    let payload: usize = strs.iter().map(String::capacity).sum();
    header + payload + std::mem::size_of::<Vec<String>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_strs_round_trips() {
        let xs = ["hello", "", "world", "rust", "ferray"];
        let c = CompactStringArray::from_strs(&xs);
        assert_eq!(c.len(), 5);
        for (i, s) in xs.iter().enumerate() {
            assert_eq!(c.as_str(i).unwrap(), *s);
        }
    }

    #[test]
    fn iter_yields_each_string_in_order() {
        let xs = ["alpha", "beta", "gamma"];
        let c = CompactStringArray::from_strs(&xs);
        let collected: Vec<&str> = c.iter().collect();
        assert_eq!(collected, xs);
    }

    #[test]
    fn empty_array_has_total_bytes_zero() {
        let c = CompactStringArray::from_iter_str(std::iter::empty::<&str>());
        assert!(c.is_empty());
        assert_eq!(c.total_bytes(), 0);
        assert_eq!(c.iter().count(), 0);
    }

    #[test]
    fn unicode_strings_preserve_bytes() {
        let xs = ["héllo", "日本語", "🦀"];
        let c = CompactStringArray::from_strs(&xs);
        assert_eq!(c.as_str(0).unwrap(), "héllo");
        assert_eq!(c.as_str(1).unwrap(), "日本語");
        assert_eq!(c.as_str(2).unwrap(), "🦀");
    }

    #[test]
    fn out_of_bounds_index_returns_none() {
        let c = CompactStringArray::from_strs(&["only"]);
        assert!(c.as_str(1).is_none());
        assert!(c.as_str(100).is_none());
    }

    #[test]
    fn from_raw_parts_round_trips_arrow_style_buffers() {
        // Three strings: "a", "bb", "ccc". Offsets [0, 1, 3, 6].
        let data = b"abbccc".to_vec();
        let offsets = vec![0_usize, 1, 3, 6];
        let c = CompactStringArray::from_raw_parts(data, offsets).unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c.as_str(0).unwrap(), "a");
        assert_eq!(c.as_str(1).unwrap(), "bb");
        assert_eq!(c.as_str(2).unwrap(), "ccc");
    }

    #[test]
    fn from_raw_parts_rejects_non_zero_first_offset() {
        let data = b"abc".to_vec();
        let offsets = vec![1_usize, 3];
        assert!(CompactStringArray::from_raw_parts(data, offsets).is_err());
    }

    #[test]
    fn from_raw_parts_rejects_decreasing_offsets() {
        let data = b"abc".to_vec();
        let offsets = vec![0_usize, 2, 1];
        assert!(CompactStringArray::from_raw_parts(data, offsets).is_err());
    }

    #[test]
    fn from_raw_parts_rejects_offset_past_data() {
        let data = b"abc".to_vec();
        let offsets = vec![0_usize, 5];
        assert!(CompactStringArray::from_raw_parts(data, offsets).is_err());
    }

    #[test]
    fn from_raw_parts_rejects_invalid_utf8() {
        // 0xFF is never valid as a UTF-8 leading byte.
        let data = vec![0xFF_u8, 0xFE];
        let offsets = vec![0_usize, 2];
        assert!(CompactStringArray::from_raw_parts(data, offsets).is_err());
    }

    #[test]
    fn from_raw_parts_empty_offsets_errors() {
        let data = vec![];
        let offsets = vec![];
        assert!(CompactStringArray::from_raw_parts(data, offsets).is_err());
    }

    #[test]
    fn round_trip_with_string_array() {
        let xs = ["foo", "bar", "baz"];
        let compact = CompactStringArray::from_strs(&xs);
        let owned = compact.to_string_array().unwrap();
        let back = CompactStringArray::from_string_array(&owned);
        let collected: Vec<&str> = back.iter().collect();
        assert_eq!(collected, xs);
    }

    #[test]
    fn compact_uses_less_memory_than_vec_string_for_short_strings() {
        // Short strings amplify the per-element header overhead in
        // Vec<String>; compact wins. With 100 single-character
        // strings, owned uses ~2.4KB of headers + heap allocs;
        // compact uses ~100 bytes data + 808 bytes offsets.
        let n = 100;
        let owned: Vec<String> = (0..n).map(|i| format!("{}", i % 10)).collect();
        let compact = CompactStringArray::from_iter_str(owned.iter().map(String::as_str));

        let owned_bytes = estimated_string_array_bytes(&owned);
        let compact_bytes = compact.estimated_bytes();
        assert!(
            compact_bytes < owned_bytes,
            "compact ({compact_bytes}) should beat owned ({owned_bytes}) for short strings"
        );
    }

    #[test]
    fn data_and_offsets_views_match_construction() {
        let xs = ["abc", "de", "f"];
        let c = CompactStringArray::from_strs(&xs);
        assert_eq!(c.data(), b"abcdef");
        assert_eq!(c.offsets(), &[0_usize, 3, 5, 6]);
        assert_eq!(c.total_bytes(), 6);
    }

    #[test]
    fn iter_size_hint_is_exact() {
        let xs = ["a", "b", "c", "d"];
        let c = CompactStringArray::from_strs(&xs);
        let mut it = c.iter();
        assert_eq!(it.size_hint(), (4, Some(4)));
        it.next();
        assert_eq!(it.size_hint(), (3, Some(3)));
    }
}
