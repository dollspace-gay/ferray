//! Strict surface-coverage gate for ferray-ufunc.
//! Fails if any public item in `_surface.json` lacks a conformance test
//! reference, exclusion entry, or divergence entry.

use std::collections::BTreeSet;

#[derive(serde::Deserialize)]
struct SurfaceFile { items: Vec<SurfaceItem> }

#[derive(serde::Deserialize)]
struct SurfaceItem {
    path: String,
    #[allow(dead_code)] kind: String,
    #[allow(dead_code)] signature: String,
}

#[derive(serde::Deserialize)]
struct ExclusionsFile {
    #[serde(default, rename = "exclusion")]
    entries: Vec<ExclusionEntry>,
}

#[derive(serde::Deserialize)]
struct ExclusionEntry { path: String, reason: String, covered_by: String }

#[derive(serde::Deserialize)]
struct DivergencesFile {
    #[serde(default, rename = "divergence")]
    entries: Vec<DivergenceEntry>,
}

#[derive(serde::Deserialize)]
struct DivergenceEntry {
    ferray_path: String,
    #[allow(dead_code)] tracking_issue: u32,
}

#[test]
fn every_public_item_has_a_conformance_reference() {
    let manifest = env!("CARGO_MANIFEST_DIR");

    let surface_path = std::path::Path::new(manifest).join("tests/conformance/_surface.json");
    let surface: SurfaceFile = serde_json::from_str(
        &std::fs::read_to_string(&surface_path).expect("read _surface.json"),
    )
    .expect("parse _surface.json");

    let excl_path = std::path::Path::new(manifest).join("tests/conformance/_surface_exclusions.toml");
    let excl_text = std::fs::read_to_string(&excl_path).unwrap_or_default();
    let exclusions: ExclusionsFile = toml::from_str(&excl_text).expect("parse _surface_exclusions.toml");
    for e in &exclusions.entries {
        assert!(
            !e.reason.trim().is_empty(),
            "Exclusion for {} has empty reason — generic exclusions forbidden",
            e.path
        );
        assert!(
            !e.covered_by.trim().is_empty(),
            "Exclusion for {} has empty covered_by",
            e.path
        );
    }
    let excluded: BTreeSet<String> = exclusions.entries.iter().map(|e| e.path.clone()).collect();

    let div_path = std::path::Path::new(manifest).join("tests/conformance/_divergences.toml");
    let div_text = std::fs::read_to_string(&div_path).unwrap_or_default();
    let divergences: DivergencesFile = toml::from_str(&div_text).expect("parse _divergences.toml");
    let divergent: BTreeSet<String> = divergences.entries.iter().map(|d| d.ferray_path.clone()).collect();

    let tests_dir = std::path::Path::new(manifest).join("tests");
    let mut all_test_text = String::new();
    for entry in std::fs::read_dir(&tests_dir).expect("read tests/") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // Skip the gate itself (avoid trivial self-coverage) and non-rs files.
        if !file_name.starts_with("conformance_") || !file_name.ends_with(".rs") {
            continue;
        }
        if file_name == "conformance_surface_coverage.rs" {
            continue;
        }
        all_test_text.push_str(&std::fs::read_to_string(&path).expect("read test file"));
    }

    let mut uncovered = Vec::new();
    for item in &surface.items {
        if excluded.contains(&item.path) { continue; }
        if divergent.contains(&item.path) { continue; }
        if all_test_text.contains(&item.path) { continue; }
        uncovered.push(item.path.clone());
    }

    assert!(
        uncovered.is_empty(),
        "{} public items in ferray-ufunc lack a conformance reference:\n  {}\n\nFix: add a conformance test mentioning the path, OR add an exclusion entry with covered_by, OR add a divergence entry.",
        uncovered.len(),
        uncovered.join("\n  ")
    );
}
