// ferray-strings: Case manipulation operations (REQ-5)
//
// Implements upper, lower, capitalize, title — elementwise on StringArray.
//
// `capitalize` and `title` match Python `str.capitalize`/`str.title`
// (the semantics `numpy.char`/`numpy.strings` delegate to via `_vec_string`,
// numpy/_core/strings.py:1239 capitalize, :1282 title). The word-initial
// character uses the Unicode TITLECASE mapping (not uppercase): the ligature
// `ﬁ` titlecases to `Fi` (not `FI`), `ß` to `Ss` (not `SS`), and the Lt
// digraph `ǅ` titlecases to itself (not `Ǆ`). Tail characters use full
// lowercase with final-sigma context (word-final `Σ` -> `ς`). Rust's
// `str::to_lowercase` already implements the Unicode Final_Sigma rule, so the
// tail is routed through it; only the word-initial titlecase needs a table,
// since Rust's stdlib has no `to_titlecase`.

use ferray_core::dimension::Dimension;
use ferray_core::error::FerrayResult;

use crate::string_array::StringArray;

/// The Unicode titlecase mapping for the characters whose titlecase differs
/// from their uppercase. For every other character titlecase == uppercase, so
/// the fallback uses `char::to_uppercase`.
///
/// Derived live from CPython/Unicode (matches the numpy 2.4 oracle):
/// `[chr(c) for c in range(0x110000) if chr(c).title() != chr(c).upper()]`,
/// mapping each to `chr(c).title()`. Covers the case ligatures
/// (`ﬁ`->`Fi`, `ß`->`Ss`, …), the Latin/Greek titlecase digraphs (Lt chars
/// that titlecase to themselves), and the Armenian/Greek/Georgian forms.
fn titlecase_char(c: char) -> String {
    let mapped: &str = match c as u32 {
        0x00DF => "Ss",
        0x01C4 => "ǅ",
        0x01C5 => "ǅ",
        0x01C6 => "ǅ",
        0x01C7 => "ǈ",
        0x01C8 => "ǈ",
        0x01C9 => "ǈ",
        0x01CA => "ǋ",
        0x01CB => "ǋ",
        0x01CC => "ǋ",
        0x01F1 => "ǲ",
        0x01F2 => "ǲ",
        0x01F3 => "ǲ",
        0x0587 => "Եւ",
        0x10D0 => "ა",
        0x10D1 => "ბ",
        0x10D2 => "გ",
        0x10D3 => "დ",
        0x10D4 => "ე",
        0x10D5 => "ვ",
        0x10D6 => "ზ",
        0x10D7 => "თ",
        0x10D8 => "ი",
        0x10D9 => "კ",
        0x10DA => "ლ",
        0x10DB => "მ",
        0x10DC => "ნ",
        0x10DD => "ო",
        0x10DE => "პ",
        0x10DF => "ჟ",
        0x10E0 => "რ",
        0x10E1 => "ს",
        0x10E2 => "ტ",
        0x10E3 => "უ",
        0x10E4 => "ფ",
        0x10E5 => "ქ",
        0x10E6 => "ღ",
        0x10E7 => "ყ",
        0x10E8 => "შ",
        0x10E9 => "ჩ",
        0x10EA => "ც",
        0x10EB => "ძ",
        0x10EC => "წ",
        0x10ED => "ჭ",
        0x10EE => "ხ",
        0x10EF => "ჯ",
        0x10F0 => "ჰ",
        0x10F1 => "ჱ",
        0x10F2 => "ჲ",
        0x10F3 => "ჳ",
        0x10F4 => "ჴ",
        0x10F5 => "ჵ",
        0x10F6 => "ჶ",
        0x10F7 => "ჷ",
        0x10F8 => "ჸ",
        0x10F9 => "ჹ",
        0x10FA => "ჺ",
        0x10FD => "ჽ",
        0x10FE => "ჾ",
        0x10FF => "ჿ",
        0x1F80 => "ᾈ",
        0x1F81 => "ᾉ",
        0x1F82 => "ᾊ",
        0x1F83 => "ᾋ",
        0x1F84 => "ᾌ",
        0x1F85 => "ᾍ",
        0x1F86 => "ᾎ",
        0x1F87 => "ᾏ",
        0x1F88 => "ᾈ",
        0x1F89 => "ᾉ",
        0x1F8A => "ᾊ",
        0x1F8B => "ᾋ",
        0x1F8C => "ᾌ",
        0x1F8D => "ᾍ",
        0x1F8E => "ᾎ",
        0x1F8F => "ᾏ",
        0x1F90 => "ᾘ",
        0x1F91 => "ᾙ",
        0x1F92 => "ᾚ",
        0x1F93 => "ᾛ",
        0x1F94 => "ᾜ",
        0x1F95 => "ᾝ",
        0x1F96 => "ᾞ",
        0x1F97 => "ᾟ",
        0x1F98 => "ᾘ",
        0x1F99 => "ᾙ",
        0x1F9A => "ᾚ",
        0x1F9B => "ᾛ",
        0x1F9C => "ᾜ",
        0x1F9D => "ᾝ",
        0x1F9E => "ᾞ",
        0x1F9F => "ᾟ",
        0x1FA0 => "ᾨ",
        0x1FA1 => "ᾩ",
        0x1FA2 => "ᾪ",
        0x1FA3 => "ᾫ",
        0x1FA4 => "ᾬ",
        0x1FA5 => "ᾭ",
        0x1FA6 => "ᾮ",
        0x1FA7 => "ᾯ",
        0x1FA8 => "ᾨ",
        0x1FA9 => "ᾩ",
        0x1FAA => "ᾪ",
        0x1FAB => "ᾫ",
        0x1FAC => "ᾬ",
        0x1FAD => "ᾭ",
        0x1FAE => "ᾮ",
        0x1FAF => "ᾯ",
        0x1FB2 => "Ὰͅ",
        0x1FB3 => "ᾼ",
        0x1FB4 => "Άͅ",
        0x1FB7 => "ᾼ͂",
        0x1FBC => "ᾼ",
        0x1FC2 => "Ὴͅ",
        0x1FC3 => "ῌ",
        0x1FC4 => "Ήͅ",
        0x1FC7 => "ῌ͂",
        0x1FCC => "ῌ",
        0x1FF2 => "Ὼͅ",
        0x1FF3 => "ῼ",
        0x1FF4 => "Ώͅ",
        0x1FF7 => "ῼ͂",
        0x1FFC => "ῼ",
        0xFB00 => "Ff",
        0xFB01 => "Fi",
        0xFB02 => "Fl",
        0xFB03 => "Ffi",
        0xFB04 => "Ffl",
        0xFB05 => "St",
        0xFB06 => "St",
        0xFB13 => "Մն",
        0xFB14 => "Մե",
        0xFB15 => "Մի",
        0xFB16 => "Վն",
        0xFB17 => "Մխ",
        _ => return c.to_uppercase().collect(),
    };
    mapped.to_string()
}

/// Whether `c` has the Unicode `Cased` property — used to find word
/// boundaries for `title` (Python's `str.title` treats a character as
/// word-initial when the previous character is not cased). Rust's
/// `char::is_lowercase` / `char::is_uppercase` cover the `Lowercase` /
/// `Uppercase` properties but miss the titlecase (`Lt`) characters, which are
/// added explicitly. (The `Lt` set: U+01C5/01C8/01CB/01F2 and the Greek
/// titlecase ranges — derived from `category(chr(c)) == 'Lt'`.)
fn is_cased(c: char) -> bool {
    c.is_lowercase()
        || c.is_uppercase()
        || matches!(
            c,
            '\u{1C5}'
                | '\u{1C8}'
                | '\u{1CB}'
                | '\u{1F2}'
                | '\u{1F88}'..='\u{1F8F}'
                | '\u{1F98}'..='\u{1F9F}'
                | '\u{1FA8}'..='\u{1FAF}'
                | '\u{1FBC}'
                | '\u{1FCC}'
                | '\u{1FFC}'
        )
}

/// Convert each string element to uppercase.
///
/// # Errors
/// Returns an error if the internal array construction fails.
///
/// # Examples
/// ```ignore
/// let a = strings::array(&["hello", "world"]).unwrap();
/// let b = strings::upper(&a).unwrap();
/// assert_eq!(b.as_slice(), &["HELLO", "WORLD"]);
/// ```
pub fn upper<D: Dimension>(a: &StringArray<D>) -> FerrayResult<StringArray<D>> {
    a.map(str::to_uppercase)
}

/// Convert each string element to lowercase.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn lower<D: Dimension>(a: &StringArray<D>) -> FerrayResult<StringArray<D>> {
    a.map(str::to_lowercase)
}

/// Capitalize each string element: the first character is titlecased and the
/// rest is lowercased (final-sigma aware), matching Python `str.capitalize`.
///
/// The first character uses the Unicode titlecase mapping (`ﬁ`->`Fi`,
/// `ß`->`Ss`); the tail is the whole-string lowercase with the first
/// character's lowercase span removed, so the final-sigma decision sees the
/// (cased) first character as preceding context.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn capitalize<D: Dimension>(a: &StringArray<D>) -> FerrayResult<StringArray<D>> {
    a.map(|s| {
        let mut chars = s.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => {
                // `str::to_lowercase` resolves every final-sigma against the
                // whole-string context; the first character's lowercase is
                // context-free (nothing precedes it), so slicing it off the
                // front leaves the correctly-lowered tail.
                let lowered = s.to_lowercase();
                let first_lower_len: usize = first.to_lowercase().map(char::len_utf8).sum();
                let tail = &lowered[first_lower_len..];
                let head = titlecase_char(first);
                format!("{head}{tail}")
            }
        }
    })
}

/// Title-case each string element: each word starts with a titlecased
/// character and all remaining cased characters are lowercased, matching
/// Python `str.title` / `numpy.char.title`.
///
/// A character is word-initial when the preceding character is not cased
/// (digits and punctuation are not cased, so a letter following one is
/// titlecased). The tail uses the whole-string lowercase so a word-final
/// `Σ` becomes `ς` (final sigma) per the Unicode Final_Sigma rule.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn title<D: Dimension>(a: &StringArray<D>) -> FerrayResult<StringArray<D>> {
    a.map(|s| {
        // Whole-string lowercase resolves final-sigma at every position; each
        // original character maps to a span of `lowered` whose byte length
        // equals `char::to_lowercase(c)` (final sigma `ς` and `σ` are both two
        // bytes, so the only context-sensitive case preserves the span width).
        let lowered = s.to_lowercase();
        let mut result = String::with_capacity(lowered.len());
        let mut previous_is_cased = false;
        let mut lowered_offset = 0usize;
        for ch in s.chars() {
            let lower_len: usize = ch.to_lowercase().map(char::len_utf8).sum();
            if is_cased(ch) && !previous_is_cased {
                result.push_str(&titlecase_char(ch));
            } else {
                result.push_str(&lowered[lowered_offset..lowered_offset + lower_len]);
            }
            lowered_offset += lower_len;
            previous_is_cased = is_cased(ch);
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn test_upper() {
        let a = array(&["hello", "world"]).unwrap();
        let b = upper(&a).unwrap();
        assert_eq!(b.as_slice(), &["HELLO", "WORLD"]);
    }

    #[test]
    fn test_lower() {
        let a = array(&["HELLO", "World"]).unwrap();
        let b = lower(&a).unwrap();
        assert_eq!(b.as_slice(), &["hello", "world"]);
    }

    #[test]
    fn test_capitalize() {
        let a = array(&["hello world", "fOO BAR", ""]).unwrap();
        let b = capitalize(&a).unwrap();
        assert_eq!(b.as_slice(), &["Hello world", "Foo bar", ""]);
    }

    #[test]
    fn test_title() {
        let a = array(&["hello world", "foo bar baz"]).unwrap();
        let b = title(&a).unwrap();
        assert_eq!(b.as_slice(), &["Hello World", "Foo Bar Baz"]);
    }

    #[test]
    fn test_title_mixed_case() {
        let a = array(&["hELLO wORLD"]).unwrap();
        let b = title(&a).unwrap();
        assert_eq!(b.as_slice(), &["Hello World"]);
    }

    #[test]
    fn test_upper_empty() {
        let a = array(&[""]).unwrap();
        let b = upper(&a).unwrap();
        assert_eq!(b.as_slice(), &[""]);
    }

    // ---- Titlecase / final-sigma divergence regression (#915) ----------
    // Expected values come from Python `str.capitalize`/`str.title`, the
    // semantics numpy.char delegates to (R-CHAR-3).

    #[test]
    fn test_capitalize_ligature_titlecase() {
        // 'ﬁnery'.capitalize() == 'Finery' (ﬁ titlecases to "Fi", not "FI").
        let a = array(&["\u{FB01}nery"]).unwrap();
        let b = capitalize(&a).unwrap();
        assert_eq!(b.as_slice(), &["Finery"]);
    }

    #[test]
    fn test_capitalize_sharp_s_titlecase() {
        // 'ßeta'.capitalize() == 'Sseta' (ß -> "Ss", not "SS").
        let a = array(&["\u{DF}eta"]).unwrap();
        let b = capitalize(&a).unwrap();
        assert_eq!(b.as_slice(), &["Sseta"]);
    }

    #[test]
    fn test_capitalize_digraph_titlecase() {
        // 'ǅungla'.capitalize() == 'ǅungla' (U+01C5 titlecases to itself).
        let a = array(&["\u{1C5}ungla"]).unwrap();
        let b = capitalize(&a).unwrap();
        assert_eq!(b.as_slice(), &["\u{1C5}ungla"]);
    }

    #[test]
    fn test_capitalize_final_sigma() {
        // 'ΟΔΟΣ'.capitalize() == 'Οδος' (word-final Σ -> ς).
        let a = array(&["\u{39F}\u{394}\u{39F}\u{3A3}"]).unwrap();
        let b = capitalize(&a).unwrap();
        assert_eq!(b.as_slice(), &["\u{39F}\u{3B4}\u{3BF}\u{3C2}"]);
    }

    #[test]
    fn test_title_ligature_titlecase() {
        // 'ﬁle ﬁle'.title() == 'File File'.
        let a = array(&["\u{FB01}le \u{FB01}le"]).unwrap();
        let b = title(&a).unwrap();
        assert_eq!(b.as_slice(), &["File File"]);
    }

    #[test]
    fn test_title_sharp_s_titlecase() {
        // 'ßtraße ßeta'.title() == 'Sstraße Sseta'.
        let a = array(&["\u{DF}tra\u{DF}e \u{DF}eta"]).unwrap();
        let b = title(&a).unwrap();
        assert_eq!(b.as_slice(), &["Sstra\u{DF}e Sseta"]);
    }

    #[test]
    fn test_title_final_sigma() {
        // 'ΟΔΟΣ ΟΔΟΣ'.title() == 'Οδος Οδος' (word-final Σ -> ς per word).
        let a = array(&["\u{39F}\u{394}\u{39F}\u{3A3} \u{39F}\u{394}\u{39F}\u{3A3}"]).unwrap();
        let b = title(&a).unwrap();
        assert_eq!(
            b.as_slice(),
            &["\u{39F}\u{3B4}\u{3BF}\u{3C2} \u{39F}\u{3B4}\u{3BF}\u{3C2}"]
        );
    }

    #[test]
    fn test_title_digit_word_boundary() {
        // 'a1b c'.title() == 'A1B C' — a digit is not cased, so the letter
        // after it is word-initial (titlecased).
        let a = array(&["a1b c"]).unwrap();
        let b = title(&a).unwrap();
        assert_eq!(b.as_slice(), &["A1B C"]);
    }
}
