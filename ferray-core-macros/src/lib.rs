// ferray-core-macros: Procedural macros for ferray-core
//
// Implements:
// - #[derive(FerrayRecord)] — generates FerrayRecord trait impl for #[repr(C)] structs
// - s![] — NumPy-style slice indexing macro
// - promoted_type!() — compile-time type promotion macro

// Macro implementations branch on parsed AST shapes; identical match arms
// reflect "different syntactic forms produce the same expansion" and are
// kept distinct on purpose for future divergence. `needless_pass_by_value`
// is a syn-API ergonomics tradeoff (the parser owns each TokenStream once).
#![allow(
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else
)]

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

// ---------------------------------------------------------------------------
// #[derive(FerrayRecord)]
// ---------------------------------------------------------------------------

/// Derive macro that generates an `unsafe impl FerrayRecord` for a `#[repr(C)]` struct.
///
/// # Requirements
/// - The struct must have `#[repr(C)]`.
/// - All fields must implement `ferray_core::dtype::Element`.
///
/// # Generated code
/// - `field_descriptors()` returns a static slice of `FieldDescriptor` with correct
///   name, dtype, offset, and size for each field.
/// - `record_size()` returns `std::mem::size_of::<Self>()`.
#[proc_macro_derive(FerrayRecord)]
pub fn derive_ferray_record(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match impl_ferray_record(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn impl_ferray_record(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    // Check for #[repr(C)]
    let has_repr_c = input.attrs.iter().any(|attr| {
        if !attr.path().is_ident("repr") {
            return false;
        }
        let mut found = false;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("C") {
                found = true;
            }
            Ok(())
        });
        found
    });

    if !has_repr_c {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "FerrayRecord requires #[repr(C)] on the struct",
        ));
    }

    // Only works on structs with named fields
    let fields = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    &input.ident,
                    "FerrayRecord only supports structs with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "FerrayRecord can only be derived for structs",
            ));
        }
    };

    let field_count = fields.len();
    let mut field_descriptors = Vec::with_capacity(field_count);

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Type-with-generics name used in field-offset / size-of expressions.
    // `#name` alone is the bare ident (`Mixed`), which won't compile in
    // `offset_of!` or `size_of::<>` for generic structs (#326). We
    // reconstruct `Mixed::<A, B>` by combining `#name` with the
    // turbofished form of `#ty_generics`.
    let ty_generics_turbofish = input.generics.split_for_impl().1.as_turbofish();
    let name_with_generics = quote! { #name #ty_generics_turbofish };

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let field_ty = &field.ty;

        field_descriptors.push(quote! {
            ferray_core::record::FieldDescriptor {
                name: #field_name_str,
                dtype: <#field_ty as ferray_core::dtype::Element>::dtype(),
                offset: std::mem::offset_of!(#name_with_generics, #field_name),
                size: std::mem::size_of::<#field_ty>(),
            }
        });
    }

    let expanded = quote! {
        unsafe impl #impl_generics ferray_core::record::FerrayRecord for #name #ty_generics #where_clause {
            fn field_descriptors() -> &'static [ferray_core::record::FieldDescriptor] {
                // #326: a plain `static FIELDS: LazyLock<Vec<…>>` here
                // can't reference the outer generic parameters
                // (`error[E0401]: use of generic parameter from outer
                // item` — statics are independent items). Use a
                // TypeId-keyed cache that lazily inserts the descriptor
                // vector on first call per monomorphization, then
                // returns the leaked slice. For non-generic structs
                // this lookups the same entry every call; for generic
                // structs each instantiation gets its own slot keyed
                // by `TypeId::of::<Self>()`.
                //
                // Issue #323 asked whether this could be a `const`
                // static array. `offset_of!` and `size_of::<T>()` are
                // both const, but `<T as Element>::dtype()` is a trait
                // method call and trait methods are not `const` on
                // stable Rust yet — so the runtime cache is
                // unavoidable. Re-evaluate once `const_trait_impl`
                // stabilises.
                use std::any::TypeId;
                use std::collections::HashMap;
                use std::sync::{OnceLock, Mutex};
                static CACHE: OnceLock<
                    Mutex<HashMap<TypeId, &'static [ferray_core::record::FieldDescriptor]>>,
                > = OnceLock::new();
                let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
                let mut guard = cache.lock().unwrap();
                *guard
                    .entry(TypeId::of::<#name_with_generics>())
                    .or_insert_with(|| {
                        let v: Vec<ferray_core::record::FieldDescriptor> = vec![
                            #(#field_descriptors),*
                        ];
                        Box::leak(v.into_boxed_slice())
                    })
            }

            fn record_size() -> usize {
                std::mem::size_of::<#name_with_generics>()
            }
        }
    };

    Ok(expanded)
}

// ---------------------------------------------------------------------------
// s![] macro — NumPy-style slice indexing
// ---------------------------------------------------------------------------

/// NumPy-style slice indexing macro.
///
/// Produces a `Vec<ferray_core::dtype::SliceInfoElem>` that can be passed
/// to array slicing methods.
///
/// # Syntax
/// - `s![0..3, 2]` — rows 0..3, column 2
/// - `s![.., 0..;2]` — all rows, every-other column starting from 0
/// - `s![1..5;2, ..]` — rows 1..5 step 2, all columns
/// - `s![3]` — single integer index
/// - `s![..]` — all elements along this axis
/// - `s![2..]` — from index 2 to end
/// - `s![..5]` — from start to index 5
/// - `s![1..5]` — from index 1 to 5
/// - `s![1..5;2]` — from index 1 to 5, step 2
///
/// Each component in the comma-separated list becomes one `SliceInfoElem`.
#[proc_macro]
pub fn s(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    let expanded = impl_s_macro(input2);
    match expanded {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn impl_s_macro(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    // We parse the input as a sequence of comma-separated slice expressions.
    // Each expression can be:
    //   - An integer literal or expression: `2` -> Index(2)
    //   - A full range `..` -> Slice { start: 0, end: None, step: 1 }
    //   - A range `a..b` -> Slice { start: a, end: Some(b), step: 1 }
    //   - A range from `a..` -> Slice { start: a, end: None, step: 1 }
    //   - A range to `..b` -> Slice { start: 0, end: Some(b), step: 1 }
    //   - Any of the above with `;step` suffix
    //
    // We'll output code that constructs a Vec<SliceInfoElem>.
    //
    // Since proc macros can't easily parse arbitrary Rust expressions with range syntax
    // mixed with custom `;step` syntax, we'll use a simpler token-based approach.
    //
    // FRAGILITY NOTE (#322): the implementation routes through
    // `TokenStream::to_string()` and string-parses the result. Rust's
    // pretty-printer formatting between tokens (whitespace placement
    // around `..`, `;`, parens) is technically allowed to drift across
    // compiler versions. The integration tests in
    // ferray-core/tests/macro_tests.rs (s_macro_*) pin a representative
    // set of inputs — variable bounds, arithmetic expressions, function
    // calls, typed literals — to catch any drift. A future cleanup
    // could walk the token tree directly via `into_iter()` and avoid
    // `to_string()` entirely; that requires writing a small recursive-
    // descent parser for `expr (.. expr)? (; expr)?` over `TokenTree`.

    let input_str = input.to_string();

    // Handle empty input
    if input_str.trim().is_empty() {
        return Ok(quote! {
            ::std::vec::Vec::<ferray_core::dtype::SliceInfoElem>::new()
        });
    }

    // Split by commas (respecting parentheses/brackets nesting)
    let components = split_top_level_commas(&input_str);
    let mut elems = Vec::new();

    for component in &components {
        let trimmed = component.trim();
        if trimmed.is_empty() {
            continue;
        }
        elems.push(parse_slice_component(trimmed)?);
    }

    Ok(quote! {
        vec![#(#elems),*]
    })
}

fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;

    for ch in s.chars() {
        match ch {
            '(' | '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                result.push(current.clone());
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

/// Find the last top-level semicolon (not inside brackets/braces/parens).
fn rfind_top_level_semicolon(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut last_idx = None;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ';' if depth == 0 => last_idx = Some(i),
            _ => {}
        }
    }
    last_idx
}

fn parse_slice_component(s: &str) -> syn::Result<proc_macro2::TokenStream> {
    let trimmed = s.trim();

    // Check for step suffix: `expr;step` (depth-aware to handle block expressions)
    let (range_part, step_part) = if let Some(idx) = rfind_top_level_semicolon(trimmed) {
        let (rp, sp) = trimmed.split_at(idx);
        (rp.trim(), Some(sp[1..].trim()))
    } else {
        (trimmed, None)
    };

    let step_expr = if let Some(step_str) = step_part {
        let step_tokens: proc_macro2::TokenStream = step_str.parse().map_err(|_| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("invalid step expression: {step_str}"),
            )
        })?;
        quote! { #step_tokens }
    } else {
        quote! { 1isize }
    };

    // Now parse range_part
    if range_part == ".." {
        // Full range: all elements
        return Ok(quote! {
            ferray_core::dtype::SliceInfoElem::Slice {
                start: 0,
                end: ::core::option::Option::None,
                step: #step_expr,
            }
        });
    }

    if let Some(rest) = range_part.strip_prefix("..") {
        // RangeTo: ..end
        let end_tokens: proc_macro2::TokenStream = rest.parse().map_err(|_| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("invalid end expression: {rest}"),
            )
        })?;
        return Ok(quote! {
            ferray_core::dtype::SliceInfoElem::Slice {
                start: 0,
                end: ::core::option::Option::Some(#end_tokens),
                step: #step_expr,
            }
        });
    }

    if let Some(idx) = range_part.find("..") {
        let start_str = range_part[..idx].trim();
        let end_str = range_part[idx + 2..].trim();

        let start_tokens: proc_macro2::TokenStream = start_str.parse().map_err(|_| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("invalid start expression: {start_str}"),
            )
        })?;

        if end_str.is_empty() {
            // RangeFrom: start..
            return Ok(quote! {
                ferray_core::dtype::SliceInfoElem::Slice {
                    start: #start_tokens,
                    end: ::core::option::Option::None,
                    step: #step_expr,
                }
            });
        }

        let end_tokens: proc_macro2::TokenStream = end_str.parse().map_err(|_| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("invalid end expression: {end_str}"),
            )
        })?;

        return Ok(quote! {
            ferray_core::dtype::SliceInfoElem::Slice {
                start: #start_tokens,
                end: ::core::option::Option::Some(#end_tokens),
                step: #step_expr,
            }
        });
    }

    // No `..` found — this is a single index (integer expression)
    if step_part.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("step ';' is not valid for integer indices: {trimmed}"),
        ));
    }

    let idx_tokens: proc_macro2::TokenStream = range_part.parse().map_err(|_| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("invalid index expression: {range_part}"),
        )
    })?;

    Ok(quote! {
        ferray_core::dtype::SliceInfoElem::Index(#idx_tokens)
    })
}

// ---------------------------------------------------------------------------
// promoted_type!() — compile-time type promotion
// ---------------------------------------------------------------------------

/// Compile-time type promotion macro.
///
/// Given two numeric types, resolves to the smallest type that can represent
/// both without precision loss, following `NumPy`'s promotion rules.
///
/// # Examples
/// ```ignore
/// type R = promoted_type!(f32, f64); // R = f64
/// type R = promoted_type!(i32, f32); // R = f64
/// type R = promoted_type!(u8, i8);   // R = i16
/// ```
#[proc_macro]
pub fn promoted_type(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    match impl_promoted_type(input2) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn impl_promoted_type(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let input_str = input.to_string();
    let parts: Vec<&str> = input_str.split(',').map(str::trim).collect();

    if parts.len() != 2 {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "promoted_type! expects exactly two type arguments: promoted_type!(T1, T2)",
        ));
    }

    let t1 = normalize_type(parts[0]);
    let t2 = normalize_type(parts[1]);

    let result = promote_types_static(&t1, &t2).ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("cannot promote types: {t1} and {t2}"),
        )
    })?;

    let result_tokens: proc_macro2::TokenStream = result.parse().map_err(|_| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("internal error: could not parse result type: {result}"),
        )
    })?;

    Ok(result_tokens)
}

fn normalize_type(s: &str) -> String {
    // Normalize Complex<f32> / Complex<f64> / num_complex::Complex<f32> etc.
    s.trim().replace(' ', "")
}

/// Static type promotion following `NumPy` rules.
///
/// Returns the promoted type as a string, or None if unknown.
fn promote_types_static(a: &str, b: &str) -> Option<&'static str> {
    // Assign a numeric "kind + rank" to each type, then pick the larger.
    //
    // NumPy promotion hierarchy (simplified):
    //   bool < u8 < u16 < u32 < u64 < u128
    //   bool < i8 < i16 < i32 < i64 < i128
    //   f32 < f64
    //   Complex<f32> < Complex<f64>
    //
    // Cross-kind rules:
    //   unsigned + signed -> next-size signed (e.g. u8 + i8 -> i16)
    //   any int + float -> float (ensure enough precision)
    //   any real + complex -> complex with appropriate float size

    let ra = type_rank(a)?;
    let rb = type_rank(b)?;

    // `promote_ranks` uses an empty string to signal "no lossless
    // common type exists" (currently the u128 + signed-int case); map
    // that to `None` so the outer proc-macro emits a compile error.
    match promote_ranks(ra, rb) {
        "" => None,
        other => Some(other),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TypeKind {
    Bool,
    Unsigned,
    Signed,
    Float,
    Complex,
}

#[derive(Clone, Copy)]
struct TypeRank {
    kind: TypeKind,
    /// Bit width within the kind (e.g., 8 for u8, 32 for f32, etc.)
    bits: u32,
}

fn type_rank(s: &str) -> Option<TypeRank> {
    let result = match s {
        "bool" => TypeRank {
            kind: TypeKind::Bool,
            bits: 1,
        },
        "u8" => TypeRank {
            kind: TypeKind::Unsigned,
            bits: 8,
        },
        "u16" => TypeRank {
            kind: TypeKind::Unsigned,
            bits: 16,
        },
        "u32" => TypeRank {
            kind: TypeKind::Unsigned,
            bits: 32,
        },
        "u64" => TypeRank {
            kind: TypeKind::Unsigned,
            bits: 64,
        },
        "u128" => TypeRank {
            kind: TypeKind::Unsigned,
            bits: 128,
        },
        "i8" => TypeRank {
            kind: TypeKind::Signed,
            bits: 8,
        },
        "i16" => TypeRank {
            kind: TypeKind::Signed,
            bits: 16,
        },
        "i32" => TypeRank {
            kind: TypeKind::Signed,
            bits: 32,
        },
        "i64" => TypeRank {
            kind: TypeKind::Signed,
            bits: 64,
        },
        "i128" => TypeRank {
            kind: TypeKind::Signed,
            bits: 128,
        },
        "f32" => TypeRank {
            kind: TypeKind::Float,
            bits: 32,
        },
        "f64" => TypeRank {
            kind: TypeKind::Float,
            bits: 64,
        },
        "Complex<f32>" | "num_complex::Complex<f32>" => TypeRank {
            kind: TypeKind::Complex,
            bits: 32,
        },
        "Complex<f64>" | "num_complex::Complex<f64>" => TypeRank {
            kind: TypeKind::Complex,
            bits: 64,
        },
        "f16" | "half::f16" => TypeRank {
            kind: TypeKind::Float,
            bits: 16,
        },
        "bf16" | "half::bf16" => TypeRank {
            kind: TypeKind::Float,
            bits: 16,
        },
        _ => return None,
    };
    Some(result)
}

fn promote_ranks(a: TypeRank, b: TypeRank) -> &'static str {
    use TypeKind::{Bool, Complex, Float, Signed, Unsigned};

    // Same type
    if a.kind == b.kind && a.bits == b.bits {
        return rank_to_type(a);
    }

    // Handle Bool: bool promotes to anything
    if a.kind == Bool {
        return rank_to_type(b);
    }
    if b.kind == Bool {
        return rank_to_type(a);
    }

    // Complex + anything -> Complex with max float precision
    if a.kind == Complex || b.kind == Complex {
        let float_bits_a = to_float_bits(a);
        let float_bits_b = to_float_bits(b);
        let bits = float_bits_a.max(float_bits_b);
        return if bits <= 32 {
            "num_complex::Complex<f32>"
        } else {
            "num_complex::Complex<f64>"
        };
    }

    // Float + anything -> Float with enough precision
    if a.kind == Float || b.kind == Float {
        let float_bits_a = to_float_bits(a);
        let float_bits_b = to_float_bits(b);
        let bits = float_bits_a.max(float_bits_b);
        return if bits <= 32 { "f32" } else { "f64" };
    }

    // Now both are integer types (Unsigned or Signed)
    match (a.kind, b.kind) {
        (Unsigned, Unsigned) => {
            let bits = a.bits.max(b.bits);
            uint_type(bits)
        }
        (Signed, Signed) => {
            let bits = a.bits.max(b.bits);
            int_type(bits)
        }
        (Unsigned, Signed) | (Signed, Unsigned) => {
            let (u, s) = if a.kind == Unsigned { (a, b) } else { (b, a) };
            // unsigned + signed: need a signed type that holds both ranges
            // u8 + i8 -> i16, u16 + i16 -> i32, etc.
            if u.bits < s.bits {
                // Signed type is strictly larger, it can hold the unsigned range
                int_type(s.bits)
            } else {
                // Need the next larger signed type
                let needed = u.bits.max(s.bits) * 2;
                if needed <= 128 {
                    int_type(needed)
                } else {
                    // u128 + any signed int: no lossless common type
                    // on stable Rust (see ferray-core/src/dtype/promotion.rs
                    // and issue #375). Return the empty string as a
                    // sentinel; `promote_types_static` maps that to
                    // `None`, which the outer proc-macro converts
                    // into a compile error with a clear message.
                    ""
                }
            }
        }
        _ => "f64", // fallback
    }
}

/// Convert any type rank to the float bit width it requires.
const fn to_float_bits(r: TypeRank) -> u32 {
    match r.kind {
        TypeKind::Bool => 32,
        TypeKind::Unsigned | TypeKind::Signed => {
            // Integers up to 24-bit mantissa fit in f32 (i.e., i8, i16, u8, u16).
            // Larger integers need f64 (53-bit mantissa).
            if r.bits <= 16 { 32 } else { 64 }
        }
        TypeKind::Float => r.bits,
        TypeKind::Complex => r.bits,
    }
}

const fn uint_type(bits: u32) -> &'static str {
    match bits {
        8 => "u8",
        16 => "u16",
        32 => "u32",
        64 => "u64",
        128 => "u128",
        _ => "u64",
    }
}

const fn int_type(bits: u32) -> &'static str {
    match bits {
        8 => "i8",
        16 => "i16",
        32 => "i32",
        64 => "i64",
        128 => "i128",
        _ => "i64",
    }
}

const fn rank_to_type(r: TypeRank) -> &'static str {
    match r.kind {
        TypeKind::Bool => "bool",
        TypeKind::Unsigned => uint_type(r.bits),
        TypeKind::Signed => int_type(r.bits),
        TypeKind::Float => {
            if r.bits <= 16 {
                "half::f16"
            } else if r.bits <= 32 {
                "f32"
            } else {
                "f64"
            }
        }
        TypeKind::Complex => {
            if r.bits <= 32 {
                "num_complex::Complex<f32>"
            } else {
                "num_complex::Complex<f64>"
            }
        }
    }
}
