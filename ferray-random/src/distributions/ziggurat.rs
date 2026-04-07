// ferray-random: Ziggurat sampling for the standard normal distribution.
//
// The Ziggurat algorithm of Marsaglia & Tsang (2000) samples from the
// half-normal distribution by decomposing the area under `f(x) = exp(-x^2/2)`
// into `N` equal-area layers. A random box is chosen uniformly and a point
// inside it is tested against the curve: in the vast majority of cases the
// point lies in a strictly-under-the-curve "fast region" and is accepted
// without any transcendental call. The sign bit of the same random word turns
// this into a full standard normal sample.
//
// With `N = 256` and `R ≈ 3.6541528853610088` (the tail boundary), the fast
// path is taken ~98% of the time, making Ziggurat roughly 3x faster than the
// Box-Muller pair we were using previously (which unconditionally performs a
// log, sqrt, sin, and cos for every two samples).
//
// The layer tables are generated on first use via `LazyLock` so there is no
// hardcoded table and no cost for programs that never use the normal
// distribution.

use std::sync::LazyLock;

use crate::bitgen::BitGenerator;

/// Number of ziggurat layers. Must be a power of two to allow the box index
/// to be taken directly from the low bits of a random `u64`.
const ZIG_N: usize = 256;

/// Tail boundary. For `x > ZIG_R` the Ziggurat falls back to truncated
/// exponential rejection sampling.
const ZIG_R: f64 = 3.654_152_885_361_009;

/// Total area of the bottom layer (and therefore every other layer). This
/// value is determined by `ZIG_R`: it equals `R * exp(-R^2/2)` plus the
/// area of the tail beyond `R`. Hardcoding it avoids needing an `erfc`
/// implementation at load time.
const ZIG_V: f64 = 0.00492867323399;

/// Precomputed ziggurat layer boundaries.
struct ZigguratTables {
    /// Layer widths. `x[0]` is the **extended** width of the bottom layer
    /// (`V / f(R)`) which lets us sample the tail via the same uniform draw
    /// used for all the other boxes. `x[1] = R`, and the entries strictly
    /// decrease to `x[ZIG_N] = 0`.
    x: [f64; ZIG_N + 1],
    /// `f[i] = exp(-x[i]^2 / 2)` with the convention `f[0] = f[1] = f(R)` and
    /// `f[ZIG_N] = 1`. Used by the wedge rejection test in the slow path.
    f: [f64; ZIG_N + 1],
}

/// Standard normal PDF, unnormalised: `f(x) = exp(-x^2 / 2)`.
#[inline(always)]
fn pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp()
}

fn build_tables() -> ZigguratTables {
    let mut x = [0.0f64; ZIG_N + 1];
    let mut f = [0.0f64; ZIG_N + 1];

    let f_r = pdf(ZIG_R);

    // Bottom layer: extended width covers the tail region in a single
    // uniform draw. `x[1] = R` is the true rectangle-to-tail boundary.
    x[0] = ZIG_V / f_r;
    f[0] = f_r;
    x[1] = ZIG_R;
    f[1] = f_r;

    // Walk up the stack. Each rectangle satisfies
    //   V = x[i] * (f[i] - f[i-1])   ⇒   f[i] = f[i-1] + V / x[i]
    // and x[i] = sqrt(-2 ln f[i]). These two equations close via fixed-point
    // iteration which converges in a few dozen rounds.
    for i in 2..ZIG_N {
        let f_prev = f[i - 1];
        let mut xi = x[i - 1];
        for _ in 0..200 {
            let fi = f_prev + ZIG_V / xi;
            // Guard against `fi >= 1` which would give a non-positive log.
            // This can happen only if the recurrence overshoots the peak; in
            // practice it does not for the first `ZIG_N - 1` layers with our
            // chosen R/V, but be defensive.
            if fi >= 1.0 {
                xi = 0.0;
                break;
            }
            let xi_new = (-2.0 * fi.ln()).sqrt();
            if (xi_new - xi).abs() < 1e-16 {
                xi = xi_new;
                break;
            }
            xi = xi_new;
        }
        x[i] = xi;
        f[i] = pdf(xi);
    }

    // Sentinel at the apex: sampling `|u * x[N-1]| < x[N]` always fails
    // (`x[N] = 0`), so the topmost layer always falls through to the wedge
    // test — which is the correct behaviour because its rectangle does not
    // fully cover the curve near `x = 0`.
    x[ZIG_N] = 0.0;
    f[ZIG_N] = 1.0;

    ZigguratTables { x, f }
}

static TABLES: LazyLock<ZigguratTables> = LazyLock::new(build_tables);

/// Sample a standard normal variate using the Ziggurat method.
///
/// ~98% of calls return after one `next_u64`, one multiplication, one
/// comparison and no transcendental functions. The slow path falls through
/// to a wedge rejection test (for ordinary layers) or to truncated
/// exponential tail sampling (for layer 0, roughly 0.3% of calls).
pub(crate) fn standard_normal_ziggurat<B: BitGenerator>(bg: &mut B) -> f64 {
    let tab = &*TABLES;
    loop {
        let bits = bg.next_u64();
        // Box index: low 8 bits. Power-of-two `N` guarantees a uniform choice.
        let i = (bits & 0xff) as usize;
        // Signed uniform in (-1, 1) from the top 56 bits. Using an i64 cast
        // preserves sign and gives us the symmetric sample for free.
        let j = (bits >> 8) as i64 - ((1i64) << 55);
        let u = (j as f64) * (1.0 / ((1u64 << 55) as f64));

        let x = u * tab.x[i];

        // Fast path: |x| < x[i+1] means we are in the region that is
        // strictly under the curve for this box (for i ≥ 1) or in the
        // rectangle part of the bottom layer (for i = 0).
        if x.abs() < tab.x[i + 1] {
            return x;
        }

        if i == 0 {
            // Truncated exponential rejection for the tail (|x| > R).
            loop {
                let u1 = bg.next_f64();
                let u2 = bg.next_f64();
                if u1 <= f64::EPSILON || u2 <= f64::EPSILON {
                    continue;
                }
                let xt = -u1.ln() / ZIG_R;
                let yt = -u2.ln();
                if 2.0 * yt > xt * xt {
                    return if u > 0.0 { ZIG_R + xt } else { -ZIG_R - xt };
                }
            }
        }

        // Wedge rejection: pick a random y between the two box f-values
        // and accept iff it lies under the true PDF at `x`.
        let y = tab.f[i - 1] + (tab.f[i] - tab.f[i - 1]) * bg.next_f64();
        if y < pdf(x) {
            return x;
        }
        // Otherwise loop.
    }
}

/// f32 variant. The tables and sampling arithmetic stay in f64 — the cost is
/// negligible on x86_64 and we preserve the full tail accuracy of the f64
/// path before casting.
#[inline]
pub(crate) fn standard_normal_ziggurat_f32<B: BitGenerator>(bg: &mut B) -> f32 {
    standard_normal_ziggurat(bg) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_rng_seeded;

    #[test]
    fn tables_are_monotone() {
        let t = &*TABLES;
        // x strictly decreasing, except possibly equal in degenerate spots.
        for i in 0..ZIG_N {
            assert!(
                t.x[i] >= t.x[i + 1],
                "x not monotone: x[{i}] = {} < x[{}] = {}",
                t.x[i],
                i + 1,
                t.x[i + 1]
            );
        }
        assert_eq!(t.x[ZIG_N], 0.0);
        assert_eq!(t.f[ZIG_N], 1.0);
    }

    #[test]
    fn tables_close_at_peak() {
        // The fixed-point recurrence should drive `x[N-1]` very close to 0
        // (and therefore `f[N-1]` very close to 1). We allow a small tolerance
        // because we hardcoded `V` rather than solving the full self-consistent
        // equation numerically.
        let t = &*TABLES;
        assert!(
            t.x[ZIG_N - 1] < 0.05,
            "x[N-1] = {} not close to 0",
            t.x[ZIG_N - 1]
        );
        assert!(
            (t.f[ZIG_N - 1] - 1.0).abs() < 1e-3,
            "f[N-1] = {} not close to 1",
            t.f[ZIG_N - 1]
        );
    }

    #[test]
    fn ziggurat_mean_and_variance_f64() {
        // 200k samples is enough to detect gross bugs but fast enough to run
        // in a unit test.
        let mut rng = default_rng_seeded(42);
        let n = 200_000usize;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for _ in 0..n {
            let x = standard_normal_ziggurat(&mut rng.bg);
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        // Standard error of the mean is 1/sqrt(n) ~ 0.0022, so allow ~5 sigma.
        assert!(mean.abs() < 0.015, "mean {mean} too far from 0");
        assert!((var - 1.0).abs() < 0.02, "var {var} too far from 1");
    }

    #[test]
    fn ziggurat_tail_is_reachable() {
        // Verify the tail sampler actually produces values outside ±R.
        let mut rng = default_rng_seeded(42);
        let mut saw_tail = 0usize;
        let n = 500_000usize;
        for _ in 0..n {
            let x = standard_normal_ziggurat(&mut rng.bg);
            if x.abs() > ZIG_R {
                saw_tail += 1;
            }
        }
        // P(|X| > 3.6541) ≈ 2 * (1 - Φ(3.6541)) ≈ 0.000258
        // Expect ~129 tail samples in 500k draws; require at least 30 to
        // rule out "tail path is never entered".
        assert!(
            saw_tail >= 30,
            "only {saw_tail} tail samples in {n} draws — tail path may be broken"
        );
    }

    #[test]
    fn ziggurat_higher_moments() {
        // Skewness and kurtosis for a normal should be ~0 and ~3.
        let mut rng = default_rng_seeded(7);
        let n = 200_000usize;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            samples.push(standard_normal_ziggurat(&mut rng.bg));
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let sd = var.sqrt();
        let skew = samples
            .iter()
            .map(|&x| ((x - mean) / sd).powi(3))
            .sum::<f64>()
            / n as f64;
        let kurt = samples
            .iter()
            .map(|&x| ((x - mean) / sd).powi(4))
            .sum::<f64>()
            / n as f64;
        assert!(skew.abs() < 0.1, "skew {skew} too large");
        assert!((kurt - 3.0).abs() < 0.2, "kurtosis {kurt} too far from 3");
    }

    #[test]
    fn ziggurat_f32_is_cast_of_f64() {
        // Since standard_normal_ziggurat_f32 is defined as `_ziggurat(..) as f32`,
        // the two functions should agree to within f32 rounding for the same
        // RNG state.
        let mut rng1 = default_rng_seeded(99);
        let mut rng2 = default_rng_seeded(99);
        for _ in 0..1000 {
            let a = standard_normal_ziggurat(&mut rng1.bg) as f32;
            let b = standard_normal_ziggurat_f32(&mut rng2.bg);
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
