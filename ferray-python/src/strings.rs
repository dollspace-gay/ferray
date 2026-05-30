//! Registration for the `ferray.strings` and `ferray.char` namespaces.
//!
//! `numpy.strings` (the NumPy 2.0 canonical namespace) and the legacy
//! `numpy.char` namespace expose the *same* vectorized string operations.
//! ferray binds each operation once as a `#[pyfunction]` in [`crate::char`]
//! and registers the identical function set under both
//! `ferray.strings.*` and `ferray.char.*` here, so callers can use either
//! surface against the same ferray-strings library kernels.
//!
//! See `.design/ferray-strings.md` for the upstream contract.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::char as c;

/// Register every shared string operation onto `m`. Used to populate both
/// `ferray.strings` and `ferray.char`.
///
/// The list mirrors `numpy.strings.__all__` / `numpy.char`'s function surface
/// (minus the chararray subclass and the `array`/`asarray` constructors, which
/// are array-creation entry points rather than elementwise string ops).
pub fn register_string_ops<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    macro_rules! reg {
        ($($fn:ident),+ $(,)?) => {{
            $( m.add_function(wrap_pyfunction!(c::$fn, m)?)?; )+
            Ok::<(), PyErr>(())
        }};
    }
    reg!(
        // case / strip
        lower,
        upper,
        capitalize,
        title,
        swapcase,
        strip,
        lstrip,
        rstrip,
        // search / query
        count,
        find,
        rfind,
        index,
        rindex,
        startswith,
        endswith,
        str_len,
        // replace / concat / multiply
        replace,
        add,
        multiply,
        // comparisons
        equal,
        not_equal,
        less,
        less_equal,
        greater,
        greater_equal,
        compare_chararrays,
        // predicates
        isalpha,
        isdigit,
        isspace,
        isalnum,
        isupper,
        islower,
        isnumeric,
        isdecimal,
        istitle,
        // alignment / padding
        center,
        ljust,
        rjust,
        zfill,
        // misc string -> string
        expandtabs,
        slice,
        translate,
        // partition / split / join
        partition,
        rpartition,
        split,
        rsplit,
        splitlines,
        join,
        // codec
        encode,
        decode,
    )?;
    // `mod` is a Rust keyword; the binding is `r#mod`. Register it under the
    // numpy name "mod".
    m.add_function(wrap_pyfunction!(c::r#mod, m)?)?;
    Ok(())
}
