//! Surface inventory tool for the ferray conformance suite.
//!
//! Walks a target crate's `src/` directory using `syn 2`, collects every
//! public item (functions, structs, traits, enums, type aliases, methods on
//! public impl blocks, and re-exports), and writes a deterministic JSON file
//! to `<crate>/tests/conformance/_surface.json`.
//!
//! # Usage
//!
//! ```text
//! cargo run -p ferray-test-oracle --bin surface-inventory -- <crate-name>
//! cargo run -p ferray-test-oracle --bin surface-inventory -- --all
//! ```
//!
//! # Determinism guarantees
//! - Items are sorted lexicographically by `path` before serialization.
//! - Source files are walked in sorted order (`walkdir` sorted by name).
//! - All intermediate keyed structures use `BTreeMap`.
//! - Output uses `serde_json::to_string_pretty` (2-space indent) + exactly
//!   one trailing newline.

// The surface-inventory binary processes arbitrary source trees and uses syn
// for code parsing. The allow list below mirrors the lib.rs allow list to
// keep clippy from flagging patterns that are acceptable in this context.
#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::match_same_arms,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use serde::Serialize;
use syn::visit::Visit;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A single item in the public surface inventory.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct SurfaceItem {
    /// Fully qualified Rust path (e.g., `ferray_window::taylor`).
    path: String,
    /// Kind: `fn`, `struct`, `trait`, `enum`, `type`, `method`, or `reexport`.
    kind: String,
    /// Full signature text as it appears in the source (whitespace-normalized).
    signature: String,
}

/// Top-level output written to `_surface.json`.
#[derive(Debug, Serialize)]
struct SurfaceInventory {
    /// Serialized as `"crate"` in JSON (Rust keyword requires rename).
    #[serde(rename = "crate")]
    crate_name: String,
    generated_by: String,
    items: Vec<SurfaceItem>,
}

// ---------------------------------------------------------------------------
// Visitor
// ---------------------------------------------------------------------------

/// State maintained while visiting a single source file with syn.
struct SurfaceVisitor<'a> {
    /// Items collected so far in this file.
    items: &'a mut Vec<SurfaceItem>,
    /// The module path prefix built from the file location / mod declarations.
    /// E.g., `["ferray_window", "windows"]`.
    module_path: Vec<String>,
    /// Whether we are inside a `#[cfg(test)]` block (skip everything inside).
    in_test_cfg: bool,
}

impl<'a> SurfaceVisitor<'a> {
    fn new(items: &'a mut Vec<SurfaceItem>, module_path: Vec<String>) -> Self {
        Self {
            items,
            module_path,
            in_test_cfg: false,
        }
    }

    fn current_path(&self, name: &str) -> String {
        if self.module_path.is_empty() {
            name.to_string()
        } else {
            format!("{}::{}", self.module_path.join("::"), name)
        }
    }

    fn push_item(&mut self, path: String, kind: &str, signature: String) {
        self.items.push(SurfaceItem {
            path,
            kind: kind.to_string(),
            signature,
        });
    }
}

/// Check whether a list of attributes contains `#[cfg(test)]`.
fn has_cfg_test(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("cfg")
            && let syn::Meta::List(ml) = &attr.meta
        {
            let tokens = ml.tokens.to_string();
            if tokens.contains("test") {
                return true;
            }
        }
    }
    false
}

/// Check whether a visibility is strictly `pub` (not `pub(crate)` / `pub(super)` /
/// `pub(in path)`). We want only fully-public items.
fn is_fully_pub(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

/// Stringify a function signature: `pub fn name<...>(...) -> ...`
fn fn_signature(func: &syn::ItemFn) -> String {
    let mut tokens = proc_macro2::TokenStream::new();
    // visibility
    func.vis.to_tokens(&mut tokens);
    // fn keyword + ident + generics
    func.sig.to_tokens(&mut tokens);
    normalize_whitespace(&tokens.to_string())
}

/// Stringify an item-level fn signature from a `TraitItemFn` or `ImplItemFn`.
fn impl_fn_signature(method: &syn::ImplItemFn) -> String {
    let mut tokens = proc_macro2::TokenStream::new();
    method.vis.to_tokens(&mut tokens);
    method.sig.to_tokens(&mut tokens);
    normalize_whitespace(&tokens.to_string())
}

/// Normalize a multi-line token string: collapse whitespace runs to a single
/// space, trim, and re-add one space after punctuation where needed.
fn normalize_whitespace(s: &str) -> String {
    // quote/proc_macro2 already produces space-separated tokens; we just
    // collapse multiple spaces and trim.
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Stringify the path portion of a `use` item (excluding `pub use`).
fn use_path_string(tree: &syn::UseTree) -> String {
    let mut tokens = proc_macro2::TokenStream::new();
    tree.to_tokens(&mut tokens);
    normalize_whitespace(&tokens.to_string())
}

impl<'ast, 'a> Visit<'ast> for SurfaceVisitor<'a> {
    // ---- Modules -----------------------------------------------------------
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if has_cfg_test(&node.attrs) {
            // Skip everything inside #[cfg(test)] modules.
            return;
        }
        // Only descend into inline modules we can see. File-level modules
        // (without a body) are handled by the file-walking logic.
        if let Some((_, items)) = &node.content {
            let mod_name = node.ident.to_string();
            self.module_path.push(mod_name);
            for item in items {
                self.visit_item(item);
            }
            self.module_path.pop();
        }
        // Non-inline (file-level) modules are handled by the file walker; we
        // do NOT recurse into them here to avoid double-counting.
    }

    // ---- Free functions ----------------------------------------------------
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        if self.in_test_cfg {
            return;
        }
        if !is_fully_pub(&node.vis) {
            return;
        }
        let name = node.sig.ident.to_string();
        let path = self.current_path(&name);
        let sig = fn_signature(node);
        self.push_item(path, "fn", sig);
    }

    // ---- Structs -----------------------------------------------------------
    fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
        if self.in_test_cfg {
            return;
        }
        if !is_fully_pub(&node.vis) {
            return;
        }
        let name = node.ident.to_string();
        let path = self.current_path(&name);
        let mut tokens = proc_macro2::TokenStream::new();
        node.vis.to_tokens(&mut tokens);
        let kw: proc_macro2::TokenStream = quote::quote! { struct };
        tokens.extend(kw);
        node.ident.to_tokens(&mut tokens);
        node.generics.to_tokens(&mut tokens);
        let sig = normalize_whitespace(&tokens.to_string());
        self.push_item(path, "struct", sig);
    }

    // ---- Enums -------------------------------------------------------------
    fn visit_item_enum(&mut self, node: &'ast syn::ItemEnum) {
        if self.in_test_cfg {
            return;
        }
        if !is_fully_pub(&node.vis) {
            return;
        }
        let name = node.ident.to_string();
        let path = self.current_path(&name);
        let mut tokens = proc_macro2::TokenStream::new();
        node.vis.to_tokens(&mut tokens);
        let kw: proc_macro2::TokenStream = quote::quote! { enum };
        tokens.extend(kw);
        node.ident.to_tokens(&mut tokens);
        node.generics.to_tokens(&mut tokens);
        let sig = normalize_whitespace(&tokens.to_string());
        self.push_item(path, "enum", sig);
    }

    // ---- Traits ------------------------------------------------------------
    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        if self.in_test_cfg {
            return;
        }
        if !is_fully_pub(&node.vis) {
            return;
        }
        let name = node.ident.to_string();
        let path = self.current_path(&name);
        let mut tokens = proc_macro2::TokenStream::new();
        node.vis.to_tokens(&mut tokens);
        let kw: proc_macro2::TokenStream = quote::quote! { trait };
        tokens.extend(kw);
        node.ident.to_tokens(&mut tokens);
        node.generics.to_tokens(&mut tokens);
        let sig = normalize_whitespace(&tokens.to_string());
        self.push_item(path, "trait", sig);
    }

    // ---- Type aliases ------------------------------------------------------
    fn visit_item_type(&mut self, node: &'ast syn::ItemType) {
        if self.in_test_cfg {
            return;
        }
        if !is_fully_pub(&node.vis) {
            return;
        }
        let name = node.ident.to_string();
        let path = self.current_path(&name);
        let mut tokens = proc_macro2::TokenStream::new();
        node.to_tokens(&mut tokens);
        let sig = normalize_whitespace(&tokens.to_string());
        self.push_item(path, "type", sig);
    }

    // ---- Impl blocks (pub methods) ----------------------------------------
    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        if self.in_test_cfg {
            return;
        }
        // Only handle inherent impls (`impl Foo { ... }`), not trait impls
        // (`impl Trait for Foo { ... }`). Trait impls' methods are part of
        // the trait contract, not directly part of the struct's surface.
        if node.trait_.is_some() {
            return;
        }
        // Extract the type name being implemented.
        let type_name = match type_name_from_type(&node.self_ty) {
            Some(n) => n,
            None => return,
        };
        let parent_path = self.current_path(&type_name);

        for item in &node.items {
            if let syn::ImplItem::Fn(method) = item {
                if !is_fully_pub(&method.vis) {
                    continue;
                }
                let method_name = method.sig.ident.to_string();
                let method_path = format!("{parent_path}::{method_name}");
                let sig = impl_fn_signature(method);
                self.push_item(method_path, "method", sig);
            }
        }
    }

    // ---- Re-exports (`pub use ...`) ----------------------------------------
    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        if self.in_test_cfg {
            return;
        }
        if !is_fully_pub(&node.vis) {
            return;
        }
        // Collect re-exported names from the use tree.
        let mut names: Vec<String> = Vec::new();
        collect_use_names(&node.tree, &mut names);

        for name in names {
            let path = self.current_path(&name);
            let sig = format!("pub use {}", use_path_string(&node.tree));
            self.push_item(path, "reexport", sig);
        }
    }
}

/// Extract a simple type name from a `syn::Type` for impl blocks.
/// Returns `None` for complex types (pointers, references, paths with
/// generics that we can't easily name).
fn type_name_from_type(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => {
            let last = tp.path.segments.last()?;
            Some(last.ident.to_string())
        }
        _ => None,
    }
}

/// Recursively collect all leaf names from a `UseTree`.
/// E.g., `{foo, bar::baz}` yields `["foo", "baz"]`.
fn collect_use_names(tree: &syn::UseTree, names: &mut Vec<String>) {
    match tree {
        syn::UseTree::Name(n) => {
            names.push(n.ident.to_string());
        }
        syn::UseTree::Rename(r) => {
            names.push(r.rename.to_string());
        }
        syn::UseTree::Glob(_) => {
            // `pub use foo::*` — we record it as a single "*" entry so
            // the surface shows the glob exists.
            names.push("*".to_string());
        }
        syn::UseTree::Group(g) => {
            for item in &g.items {
                collect_use_names(item, names);
            }
        }
        syn::UseTree::Path(p) => {
            collect_use_names(&p.tree, names);
        }
    }
}

// ---------------------------------------------------------------------------
// File-level parsing
// ---------------------------------------------------------------------------

/// Parse a single `.rs` source file and append all public items to `items`.
/// `module_path` is the fully-qualified module prefix for items in this file.
fn parse_file(path: &Path, module_path: Vec<String>, items: &mut Vec<SurfaceItem>) {
    let src = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("WARNING: could not read {}: {e}", path.display());
            return;
        }
    };
    let file = match syn::parse_file(&src) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("WARNING: could not parse {}: {e}", path.display());
            return;
        }
    };
    let mut visitor = SurfaceVisitor::new(items, module_path);
    syn::visit::visit_file(&mut visitor, &file);
}

// ---------------------------------------------------------------------------
// Crate-level walking
// ---------------------------------------------------------------------------

/// Walk `src/` for a crate rooted at `crate_dir`, collect all public items.
///
/// Files are walked in sorted order (lexicographic by full path). Module
/// paths are derived from the file path relative to `src/`.
fn collect_items_for_crate(crate_dir: &Path, crate_name: &str) -> Vec<SurfaceItem> {
    let src_dir = crate_dir.join("src");
    if !src_dir.is_dir() {
        return Vec::new();
    }

    // The crate root module name uses underscores (Rust crate name convention).
    let crate_root_mod = crate_name.replace('-', "_");

    let mut items: Vec<SurfaceItem> = Vec::new();

    // walkdir sorted by file name for determinism.
    let walker = walkdir::WalkDir::new(&src_dir)
        .sort_by_file_name()
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && e.path().extension().is_some_and(|ext| ext == "rs"));

    for entry in walker {
        let file_path = entry.path();

        // Derive module path from relative path under src/.
        // src/lib.rs          → [crate_root_mod]
        // src/windows/mod.rs  → [crate_root_mod, "windows"]
        // src/windows/foo.rs  → [crate_root_mod, "windows", "foo"]
        let rel = file_path.strip_prefix(&src_dir).unwrap();
        let module_path = path_to_module_path(&crate_root_mod, rel);

        parse_file(file_path, module_path, &mut items);
    }

    items
}

/// Convert a relative path under `src/` to a module path.
///
/// Examples:
/// - `lib.rs`          → `[crate_root]`
/// - `windows/mod.rs`  → `[crate_root, "windows"]`
/// - `windows/foo.rs`  → `[crate_root, "windows", "foo"]`
fn path_to_module_path(crate_root: &str, rel: &Path) -> Vec<String> {
    let mut parts: Vec<String> = vec![crate_root.to_string()];
    let components: Vec<_> = rel.components().collect();
    let n = components.len();
    for (i, comp) in components.iter().enumerate() {
        if let std::path::Component::Normal(os) = comp {
            let s = os.to_string_lossy();
            if i == n - 1 {
                // Last component — it's the file name.
                if s == "lib.rs" || s == "main.rs" {
                    // crate root: no extra component
                } else if s == "mod.rs" {
                    // Inline module root — the module name is the parent dir,
                    // already pushed. Do nothing here.
                } else {
                    // Strip the .rs extension.
                    let mod_name = s.trim_end_matches(".rs");
                    parts.push(mod_name.to_string());
                }
            } else {
                // Intermediate directory = module name.
                parts.push(s.to_string());
            }
        }
    }
    parts
}

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

/// After collecting all items across all source files, remove duplicates.
///
/// Duplicates arise when re-exports in `lib.rs` point to items already
/// collected from their definition files, but with `kind: reexport`. We keep
/// the definition (non-reexport) over the reexport when both exist for the
/// same path.
///
/// Items are deduplicated by `path`. Within a group of items sharing the same
/// path, the non-reexport entry wins; if all are reexports, keep only the
/// first (alphabetically by kind, then signature).
fn deduplicate(mut items: Vec<SurfaceItem>) -> Vec<SurfaceItem> {
    // Group by path using a BTreeMap to stay deterministic.
    let mut by_path: BTreeMap<String, Vec<SurfaceItem>> = BTreeMap::new();
    for item in items.drain(..) {
        by_path.entry(item.path.clone()).or_default().push(item);
    }

    let mut result: Vec<SurfaceItem> = Vec::new();
    for (_path, mut group) in by_path {
        // Sort within the group so the selection is deterministic.
        group.sort();
        // Prefer non-reexport entries.
        let non_reexport: Vec<_> = group.iter().filter(|i| i.kind != "reexport").collect();
        if !non_reexport.is_empty() {
            // Pick the first non-reexport (already sorted, so deterministic).
            result.push(non_reexport[0].clone());
        } else {
            result.push(group[0].clone());
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Write the inventory for `crate_name` rooted at `crate_dir`.
fn write_inventory(crate_dir: &Path, crate_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let items = collect_items_for_crate(crate_dir, crate_name);
    let mut items = deduplicate(items);

    // Final sort by path for determinism.
    items.sort_by(|a, b| a.path.cmp(&b.path));

    let inventory = SurfaceInventory {
        crate_name: crate_name.to_string(),
        generated_by: "surface-inventory v1".to_string(),
        items,
    };

    let json =
        serde_json::to_string_pretty(&inventory).expect("JSON serialization should not fail");

    // Write to <crate>/tests/conformance/_surface.json with exactly one
    // trailing newline.
    let out_dir = crate_dir.join("tests").join("conformance");
    fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join("_surface.json");
    fs::write(&out_path, format!("{json}\n"))?;

    println!(
        "Wrote {} items to {}",
        inventory.items.len(),
        out_path.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Workspace member enumeration
// ---------------------------------------------------------------------------

/// Read the workspace `Cargo.toml` and return the list of member crate names.
/// Returns them in sorted order (deterministic --all output order).
fn workspace_members(workspace_root: &Path) -> Vec<String> {
    let cargo_toml = workspace_root.join("Cargo.toml");
    let src = fs::read_to_string(&cargo_toml)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", cargo_toml.display()));

    let doc: toml::Value = src
        .parse()
        .unwrap_or_else(|e| panic!("cannot parse {}: {e}", cargo_toml.display()));

    let members = doc
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .unwrap_or_else(|| panic!("no [workspace].members in {}", cargo_toml.display()));

    let mut names: Vec<String> = members
        .iter()
        .filter_map(|v| v.as_str())
        .filter(|s| {
            // Exclude ferray-python (pre-publication, commented in workspace
            // but we also skip it explicitly in case it appears as a string).
            *s != "ferray-python"
        })
        .map(|s| s.to_string())
        .collect();

    names.sort();
    names
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Determine the workspace root relative to this binary's manifest.
    // CARGO_MANIFEST_DIR is set by Cargo when running via `cargo run`.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let manifest_path = PathBuf::from(&manifest_dir);
    // ferray-test-oracle is one level below the workspace root.
    let workspace_root = manifest_path
        .parent()
        .unwrap_or(&manifest_path)
        .to_path_buf();

    if args.len() < 2 {
        eprintln!("Usage:\n  surface-inventory <crate-name>\n  surface-inventory --all");
        std::process::exit(1);
    }

    let all_flag = args[1] == "--all";

    if all_flag {
        let members = workspace_members(&workspace_root);
        let mut any_error = false;
        for member in &members {
            let crate_dir = workspace_root.join(member);
            if !crate_dir.join("src").is_dir() {
                eprintln!("Skipping {member}: no src/ directory");
                continue;
            }
            if let Err(e) = write_inventory(&crate_dir, member) {
                eprintln!("ERROR processing {member}: {e}");
                any_error = true;
            }
        }
        if any_error {
            std::process::exit(1);
        }
    } else {
        let crate_name = &args[1];
        let crate_dir = workspace_root.join(crate_name);
        if !crate_dir.exists() {
            eprintln!(
                "ERROR: crate directory '{}' does not exist",
                crate_dir.display()
            );
            std::process::exit(1);
        }
        if let Err(e) = write_inventory(&crate_dir, crate_name) {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    }
}
