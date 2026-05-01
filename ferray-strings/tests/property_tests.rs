// Property-based tests for ferray-strings.
//
// Uses proptest with 256 cases per test. All test names are prefixed `prop_`.

use proptest::prelude::*;

use ferray_strings::align::center;
use ferray_strings::array;
use ferray_strings::case::{lower, upper};
use ferray_strings::concat::{add, multiply};
use ferray_strings::search::{count, endswith, find, replace, startswith};
use ferray_strings::split_join::join;
use ferray_strings::strip::strip;

fn config() -> ProptestConfig {
    ProptestConfig::with_cases(256)
}

proptest! {
    #![proptest_config(config())]

    // 1. upper(upper(x)) == upper(x)
    #[test]
    fn prop_upper_idempotent(
        s1 in "[a-z]{1,10}",
        s2 in "[a-z]{1,10}",
        s3 in "[a-z]{1,10}",
    ) {
        let a = array(&[&s1, &s2, &s3]).unwrap();
        let once = upper(&a).unwrap();
        let twice = upper(&once).unwrap();
        prop_assert_eq!(once, twice);
    }

    // 2. lower(lower(x)) == lower(x)
    #[test]
    fn prop_lower_idempotent(
        s1 in "[a-z]{1,10}",
        s2 in "[a-z]{1,10}",
        s3 in "[a-z]{1,10}",
    ) {
        let a = array(&[&s1, &s2, &s3]).unwrap();
        let once = lower(&a).unwrap();
        let twice = lower(&once).unwrap();
        prop_assert_eq!(once, twice);
    }

    // 3. lower(upper(x)) == lower(x) for ASCII strings
    #[test]
    fn prop_upper_lower_roundtrip(
        s1 in "[a-zA-Z]{1,10}",
        s2 in "[a-zA-Z]{1,10}",
        s3 in "[a-zA-Z]{1,10}",
    ) {
        let a = array(&[&s1, &s2, &s3]).unwrap();
        let via_upper = lower(&upper(&a).unwrap()).unwrap();
        let direct = lower(&a).unwrap();
        prop_assert_eq!(via_upper, direct);
    }

    // 4. strip(strip(x)) == strip(x) with None chars (whitespace)
    #[test]
    fn prop_strip_idempotent(
        s1 in "[a-z ]{1,10}",
        s2 in "[a-z ]{1,10}",
        s3 in "[a-z ]{1,10}",
    ) {
        let a = array(&[&s1, &s2, &s3]).unwrap();
        let once = strip(&a, None).unwrap();
        let twice = strip(&once, None).unwrap();
        prop_assert_eq!(once, twice);
    }

    // 5. join(sep, split(x, sep)) == x for strings not containing sep
    #[test]
    fn prop_split_join_roundtrip(
        s1 in "[a-z]{1,5}",
        s2 in "[a-z]{1,5}",
        s3 in "[a-z]{1,5}",
    ) {
        // Use "|" as separator; generated strings contain only [a-z]
        // so the separator never appears in the input. Each split
        // produces a single-element row → rejoin gives back the
        // original string.
        let sep = "|";
        let a = array(&[&s1, &s2, &s3]).unwrap();
        let parts = ferray_strings::split_ragged(&a, sep).unwrap();
        let rejoined = join(sep, &parts).unwrap();
        prop_assert_eq!(a.as_slice(), rejoined.as_slice());
    }

    // 6. multiply(x, n) has each element length == original_len * n
    #[test]
    fn prop_multiply_length(
        s1 in "[a-z]{1,10}",
        s2 in "[a-z]{1,10}",
        s3 in "[a-z]{1,10}",
        n in 0usize..10,
    ) {
        let a = array(&[&s1, &s2, &s3]).unwrap();
        let result = multiply(&a, n).unwrap();
        for (original, repeated) in a.as_slice().iter().zip(result.as_slice().iter()) {
            prop_assert_eq!(repeated.len(), original.len() * n);
        }
    }

    // 7. find(x, substring_not_in_x) returns -1 for all elements
    #[test]
    fn prop_find_not_found(
        s1 in "[a-z]{1,10}",
        s2 in "[a-z]{1,10}",
        s3 in "[a-z]{1,10}",
    ) {
        // Use a substring that cannot appear in [a-z] strings
        let needle = "999";
        let a = array(&[&s1, &s2, &s3]).unwrap();
        let result = find(&a, needle).unwrap();
        let data = result.as_slice().unwrap();
        for &val in data {
            prop_assert_eq!(val, -1_i64);
        }
    }

    // 8. count always >= 0 (result is u64, so this is guaranteed by the type,
    //    but we verify the operation succeeds and produces valid values)
    #[test]
    fn prop_count_nonnegative(
        s1 in "[a-z]{1,10}",
        s2 in "[a-z]{1,10}",
        s3 in "[a-z]{1,10}",
        needle in "[a-z]{1,3}",
    ) {
        let a = array(&[&s1, &s2, &s3]).unwrap();
        let result = count(&a, &needle).unwrap();
        let data = result.as_slice().unwrap();
        for &val in data {
            // u64 is always >= 0 by definition, but we verify the operation
            // completes without error and values are sensible
            prop_assert!(val <= s1.len() as u64 || val <= s2.len() as u64 || val <= s3.len() as u64
                || val == 0);
        }
    }

    // 9. If we prepend prefix, startswith(prefix) is true for all elements
    #[test]
    fn prop_startswith_after_add(
        s1 in "[a-z]{1,10}",
        s2 in "[a-z]{1,10}",
        s3 in "[a-z]{1,10}",
        prefix in "[A-Z]{1,5}",
    ) {
        let base = array(&[&s1, &s2, &s3]).unwrap();
        let pfx = array(&[&*prefix]).unwrap();
        // add broadcasts: pfx (len 1) against base (len 3)
        let prepended = add(&pfx, &base).unwrap();
        let result = startswith(&prepended, &prefix).unwrap();
        let data = result.as_slice().unwrap();
        for &val in data {
            prop_assert!(val, "Expected startswith to be true after prepending prefix");
        }
    }

    // 10. If we append suffix, endswith(suffix) is true for all elements
    #[test]
    fn prop_endswith_after_add(
        s1 in "[a-z]{1,10}",
        s2 in "[a-z]{1,10}",
        s3 in "[a-z]{1,10}",
        suffix in "[A-Z]{1,5}",
    ) {
        let base = array(&[&s1, &s2, &s3]).unwrap();
        let sfx = array(&[&*suffix]).unwrap();
        // add broadcasts: base (len 3) + sfx (len 1)
        let appended = add(&base, &sfx).unwrap();
        let result = endswith(&appended, &suffix).unwrap();
        let data = result.as_slice().unwrap();
        for &val in data {
            prop_assert!(val, "Expected endswith to be true after appending suffix");
        }
    }

    // 11. center(x, width) when stripped of fill chars contains the original string
    #[test]
    fn prop_center_preserves_content(
        s1 in "[a-z]{1,10}",
        s2 in "[a-z]{1,10}",
        s3 in "[a-z]{1,10}",
        extra_width in 0usize..20,
    ) {
        let a = array(&[&s1, &s2, &s3]).unwrap();
        // Use a width large enough to trigger padding
        let max_len = a.as_slice().iter().map(std::string::String::len).max().unwrap_or(0);
        let width = max_len + extra_width;
        let centered = center(&a, width, ' ').unwrap();
        let stripped = strip(&centered, None).unwrap();
        // After stripping whitespace fill, we should recover the original strings
        prop_assert_eq!(a.as_slice(), stripped.as_slice());
    }

    // 12. replace(x, sub, "", None) means count(result, sub) == 0
    #[test]
    fn prop_replace_removes_all(
        s1 in "[a-z]{1,10}",
        s2 in "[a-z]{1,10}",
        s3 in "[a-z]{1,10}",
        sub in "[a-z]{1,3}",
    ) {
        let a = array(&[&s1, &s2, &s3]).unwrap();
        let replaced = replace(&a, &sub, "", None).unwrap();
        let result = count(&replaced, &sub).unwrap();
        let data = result.as_slice().unwrap();
        for &val in data {
            prop_assert_eq!(val, 0_u64, "Expected count to be 0 after replacing all occurrences");
        }
    }
}
