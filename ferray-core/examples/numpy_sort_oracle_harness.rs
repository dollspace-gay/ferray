//! Critic-only batch harness for #2142 numpy-sort bit-identity audit.
//!
//! Reads newline-delimited JSON-ish records from stdin, one per line:
//!   S <n> v0 v1 ... v{n-1}            -> emit argsort_numpy indices
//!   P <kth> <n> v0 v1 ... v{n-1}     -> emit argpartition_numpy(kth) indices
//! Values are parsed as f64; the tokens `nan`, `inf`, `-inf`, `-0` are accepted.
//! Emits one line per record: space-separated u64 indices (or `NONE`).
//!
//! This is a TEST ARTIFACT (examples/, never linked into the library). It exists
//! so the Python numpy-2.4.5 oracle driver can stream thousands of arrays through
//! ferray's real `argsort_numpy`/`argpartition_numpy` and diff index-for-index.

use std::io::{self, BufRead, Write};

fn parse_f64(tok: &str) -> f64 {
    match tok {
        "nan" | "NaN" | "NAN" => f64::NAN,
        "inf" | "Inf" | "INF" | "+inf" => f64::INFINITY,
        "-inf" | "-Inf" | "-INF" => f64::NEG_INFINITY,
        "-0" | "-0.0" => -0.0_f64,
        _ => tok.parse::<f64>().expect("bad f64 token"),
    }
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let kind = it.next().unwrap();
        match kind {
            "S" => {
                let n: usize = it.next().unwrap().parse().unwrap();
                let a: Vec<f64> = (0..n).map(|_| parse_f64(it.next().unwrap())).collect();
                let idx = ferray_core::argsort_numpy(&a);
                let s: Vec<String> = idx.iter().map(|x| x.to_string()).collect();
                writeln!(out, "{}", s.join(" ")).unwrap();
            }
            "P" => {
                let kth: usize = it.next().unwrap().parse().unwrap();
                let n: usize = it.next().unwrap().parse().unwrap();
                let a: Vec<f64> = (0..n).map(|_| parse_f64(it.next().unwrap())).collect();
                match ferray_core::argpartition_numpy(&a, kth) {
                    Some(idx) => {
                        let s: Vec<String> = idx.iter().map(|x| x.to_string()).collect();
                        writeln!(out, "{}", s.join(" ")).unwrap();
                    }
                    None => writeln!(out, "NONE").unwrap(),
                }
            }
            other => panic!("bad record kind {other}"),
        }
    }
    out.flush().unwrap();
}
