# ferrum-strings

Vectorized string operations on arrays of strings for the [ferrum](https://crates.io/crates/ferrum) scientific computing library.

## What's in this crate

- `StringArray` — N-dimensional array of strings
- **Case**: `upper`, `lower`, `title`, `capitalize`, `swapcase`
- **Alignment**: `center`, `ljust`, `rjust`, `zfill`
- **Trimming**: `strip`, `lstrip`, `rstrip`
- **Search**: `find`, `rfind`, `index`, `count`, `startswith`, `endswith`, `contains`
- **Split/Join**: `split`, `rsplit`, `splitlines`, `join`, `partition`, `rpartition`
- **Regex**: `replace`, `match_re`, `findall`
- **Comparison**: `equal`, `not_equal`, `greater`, `less`, etc.

Implements `numpy.strings` (NumPy 2.0+) for Rust.

## Usage

```rust
use ferrum_strings::StringArray;

let arr = StringArray::from_vec(vec!["hello", "world"])?;
let upper = arr.upper()?;
```

This crate is re-exported through the main [`ferrum`](https://crates.io/crates/ferrum) crate with the `strings` feature (enabled by default).

## License

MIT OR Apache-2.0
