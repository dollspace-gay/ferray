// ferray-core: NumPy-bit-identical argsort / argpartition for f64 (#2142).
//
// These are SEPARATE from the existing stable [`Array::argsort`] / [`Array::sorted`]
// (#366). Existing consumers stay on the stable sort; these functions reproduce
// NumPy 2.4.5's `np.argsort(kind='quicksort')` and `np.argpartition`
// (`kind='introselect'`) *bit-for-bit* on the INDEX array, including exact tie
// ordering. That bit-identity is the substrate for matching scikit-learn's KNN
// exact-tie neighbor selection.
//
// IMPORTANT — what NumPy 2.4.5 actually does for f64 arg*:
// We empirically verified (and confirmed against the numpy 2.4.5 sources +
// the pinned `intel/x86-simd-sort` commit 5adb334) that on an x86-64 CPU with
// AVX2 (X86_V3), `np.argsort`/`np.argpartition` for `double` do NOT use the
// scalar `npysort` introsort/introselect. They dispatch to
// `x86simdsortStatic::argsort` / `argselect` (the AVX2 path, `vtype::numlanes==4`).
// The scalar npysort path produces DIFFERENT tie orders (verified divergent on
// the #2141 KNN fixture). So bit-identity REQUIRES emulating the x86-simd-sort
// AVX2 algorithm, not the textbook npysort one.
//
// This module is a faithful SCALAR emulation of that AVX2 algorithm:
//   * a SIMD "register" of 4 doubles is modelled as `[f64; 4]` (lane 0 = low),
//   * COEX / cmp_merge / lane shuffles (`reverse`, `swap_n`, `reverse_n`) are
//     emulated exactly per the AVX2 intrinsic lane semantics
//     (`_mm256_permute4x64_pd`, `_mm256_blendv_pd`, `_mm256_min/max_pd`),
//   * the bitonic sorting network (`argsort_n` for N <= 256), the median-of-4
//     pivot (`get_pivot_64bit`), the unrolled Hoare-style `argpartition`, and
//     the `argsort_`/`argselect_` recursion are ported line-for-line from
//     `xss-common-argsort.h`, `xss-network-keyvaluesort.hpp`,
//     `xss-reg-networks.hpp`, and `avx2-64bit-qsort.hpp`.
//
// NaN handling: numpy/x86-simd-sort routes NaN-containing float arrays to
// `std::sort`/`std::nth_element` with a NaN-last comparator. We reproduce THAT
// path too: a faithful port of libstdc++'s `std::sort` (introsort) and
// `std::nth_element` (introselect) driven by numpy's NaN-last comparator, so
// NaN-containing inputs are also bit-identical (verified against the live
// numpy 2.4.5 oracle). See [`argsort_numpy`] / [`argpartition_numpy`] docs.

const NETWORK_MAX_N: usize = 256;
const NUMLANES: usize = 4;

/// A modelled AVX2 64-bit register: 4 `f64` lanes, lane 0 = low.
type Reg = [f64; NUMLANES];
/// The paired index register: 4 `u64` lanes.
type IReg = [u64; NUMLANES];

// --------------------------------------------------------------------------
// Lane primitives (exact AVX2 intrinsic semantics)
// --------------------------------------------------------------------------

/// `_mm256_min_pd(a, b)`: lanewise. Hardware semantics: returns `b` when the
/// comparison `a < b` is false (so for `a == b` or any NaN, returns `b`). The
/// no-NaN sort path never feeds NaN here.
#[inline]
fn vmin(a: Reg, b: Reg) -> Reg {
    let mut r = [0.0; NUMLANES];
    for i in 0..NUMLANES {
        r[i] = if a[i] < b[i] { a[i] } else { b[i] };
    }
    r
}

/// `_mm256_max_pd(a, b)`: lanewise; returns `b` when `a > b` is false.
#[inline]
fn vmax(a: Reg, b: Reg) -> Reg {
    let mut r = [0.0; NUMLANES];
    for i in 0..NUMLANES {
        r[i] = if a[i] > b[i] { a[i] } else { b[i] };
    }
    r
}

/// `_mm256_cmp_pd(a, b, _CMP_EQ_OQ)`: lanewise ordered equality → per-lane
/// boolean mask.
#[inline]
fn veq(a: Reg, b: Reg) -> [bool; NUMLANES] {
    let mut m = [false; NUMLANES];
    for i in 0..NUMLANES {
        m[i] = a[i] == b[i];
    }
    m
}

/// `_mm256_blendv_pd(x, y, mask)`: lane = `mask[i] ? y[i] : x[i]`.
#[inline]
fn blend_f(x: Reg, mask: [bool; NUMLANES], y: Reg) -> Reg {
    let mut r = [0.0; NUMLANES];
    for i in 0..NUMLANES {
        r[i] = if mask[i] { y[i] } else { x[i] };
    }
    r
}

/// Index-register blend, same mask semantics as [`blend_f`].
#[inline]
fn blend_i(x: IReg, mask: [bool; NUMLANES], y: IReg) -> IReg {
    let mut r = [0u64; NUMLANES];
    for i in 0..NUMLANES {
        r[i] = if mask[i] { y[i] } else { x[i] };
    }
    r
}

/// `convert_int_to_mask(m)`: bit `i` of `m` → lane-`i` mask boolean.
#[inline]
fn int_mask(m: u32) -> [bool; NUMLANES] {
    let mut r = [false; NUMLANES];
    for (i, slot) in r.iter_mut().enumerate() {
        *slot = (m >> i) & 1 == 1;
    }
    r
}

/// `vtype::reverse` = `_mm256_permute4x64_pd(reg, SHUFFLE_MASK(0,1,2,3))`,
/// i.e. full lane reverse `[a0,a1,a2,a3] -> [a3,a2,a1,a0]`.
#[inline]
fn reverse_f(a: Reg) -> Reg {
    [a[3], a[2], a[1], a[0]]
}
#[inline]
fn reverse_i(a: IReg) -> IReg {
    [a[3], a[2], a[1], a[0]]
}

/// `swap_n<2>` = `_mm256_permute4x64(reg, 0b10110001)` → `[a1,a0,a3,a2]`.
#[inline]
fn swap2_f(a: Reg) -> Reg {
    [a[1], a[0], a[3], a[2]]
}
#[inline]
fn swap2_i(a: IReg) -> IReg {
    [a[1], a[0], a[3], a[2]]
}

/// `swap_n<4>` = `_mm256_permute4x64(reg, 0b01001110)` → `[a2,a3,a0,a1]`.
#[inline]
fn swap4_f(a: Reg) -> Reg {
    [a[2], a[3], a[0], a[1]]
}
#[inline]
fn swap4_i(a: IReg) -> IReg {
    [a[2], a[3], a[0], a[1]]
}

/// `reverse_n<vtype, scale>`: scale==2 → `swap_n<2>`; scale==4 → `reverse`.
/// Callers only ever pass scale 2 or 4 (the 4-lane network uses no others);
/// any other value falls back to the scale-4 op rather than panicking.
#[inline]
fn reverse_n_f(a: Reg, scale: usize) -> Reg {
    if scale == 2 { swap2_f(a) } else { reverse_f(a) }
}
#[inline]
fn reverse_n_i(a: IReg, scale: usize) -> IReg {
    if scale == 2 { swap2_i(a) } else { reverse_i(a) }
}
#[inline]
fn swap_n_f(a: Reg, scale: usize) -> Reg {
    if scale == 2 { swap2_f(a) } else { swap4_f(a) }
}
#[inline]
fn swap_n_i(a: IReg, scale: usize) -> IReg {
    if scale == 2 { swap2_i(a) } else { swap4_i(a) }
}

/// `cmp_merge` (key+index): pick `min`/`max` per `mask`, carry indices.
///
/// Returns the merged key register; updates `idx1` in place to follow the keys
/// (when `tmp_keys == in1`, keep `idx1`, else take `idx2`).
#[inline]
fn cmp_merge(in1: Reg, in2: Reg, idx1: &mut IReg, idx2: IReg, mask: [bool; NUMLANES]) -> Reg {
    let mn = vmin(in2, in1);
    let mx = vmax(in2, in1);
    let tmp_keys = blend_f(mn, mask, mx);
    let eqm = veq(tmp_keys, in1);
    *idx1 = blend_i(idx2, eqm, *idx1);
    tmp_keys
}

/// `COEX(key1, key2, index1, index2)`: lanewise compare-exchange so `key1`
/// holds the per-lane min and `key2` the per-lane max, with indices following.
#[inline]
fn coex(key1: &mut Reg, key2: &mut Reg, idx1: &mut IReg, idx2: &mut IReg) {
    let kt1 = vmin(*key1, *key2);
    let kt2 = vmax(*key1, *key2);
    let eqm = veq(kt1, *key1);
    let it1 = blend_i(*idx2, eqm, *idx1);
    let it2 = blend_i(*idx1, eqm, *idx2);
    *key1 = kt1;
    *key2 = kt2;
    *idx1 = it1;
    *idx2 = it2;
}

// --------------------------------------------------------------------------
// In-register networks (avx2 64-bit, numlanes == 4)
// --------------------------------------------------------------------------

/// `sort_reg_4lanes`: sort a single 4-lane register ascending, carrying indices.
fn sort_reg_4lanes(key: &mut Reg, idx: &mut IReg) {
    let oxa = int_mask(0xA);
    let oxc = int_mask(0xC);
    let other = reverse_n_f(*key, 2);
    let oidx = reverse_n_i(*idx, 2);
    *key = cmp_merge(*key, other, idx, oidx, oxa);
    let other = reverse_n_f(*key, 4);
    let oidx = reverse_n_i(*idx, 4);
    *key = cmp_merge(*key, other, idx, oidx, oxc);
    let other = swap_n_f(*key, 2);
    let oidx = swap_n_i(*idx, 2);
    *key = cmp_merge(*key, other, idx, oidx, oxa);
}

/// `bitonic_merge_reg_4lanes`: half-cleaner merge of a bitonic 4-lane register.
fn bitonic_merge_reg_4lanes(key: &mut Reg, idx: &mut IReg) {
    let oxa = int_mask(0xA);
    let oxc = int_mask(0xC);
    let other = swap_n_f(*key, 4);
    let oidx = swap_n_i(*idx, 4);
    *key = cmp_merge(*key, other, idx, oidx, oxc);
    let other = swap_n_f(*key, 2);
    let oidx = swap_n_i(*idx, 2);
    *key = cmp_merge(*key, other, idx, oidx, oxa);
}

/// `bitonic_clean_n_vec`: cross-register half-cleaner over `numvecs` registers.
fn bitonic_clean_n_vec(keys: &mut [Reg], idxs: &mut [IReg], numvecs: usize) {
    let mut num = numvecs / 2;
    while num >= 2 {
        let mut j = 0;
        while j < numvecs {
            for i in 0..num / 2 {
                let i1 = i + j;
                let i2 = i + j + num / 2;
                let (mut k1, mut k2) = (keys[i1], keys[i2]);
                let (mut x1, mut x2) = (idxs[i1], idxs[i2]);
                coex(&mut k1, &mut k2, &mut x1, &mut x2);
                keys[i1] = k1;
                keys[i2] = k2;
                idxs[i1] = x1;
                idxs[i2] = x2;
            }
            j += num;
        }
        num /= 2;
    }
}

/// `bitonic_merge_n_vec`: merge a contiguous block of `numvecs` registers.
fn bitonic_merge_n_vec(keys: &mut [Reg], idxs: &mut [IReg], numvecs: usize) {
    if numvecs == 2 {
        keys[1] = reverse_f(keys[1]);
        idxs[1] = reverse_i(idxs[1]);
        let (mut k0, mut k1) = (keys[0], keys[1]);
        let (mut x0, mut x1) = (idxs[0], idxs[1]);
        coex(&mut k0, &mut k1, &mut x0, &mut x1);
        keys[0] = k0;
        keys[1] = k1;
        idxs[0] = x0;
        idxs[1] = x1;
        keys[1] = reverse_f(keys[1]);
        idxs[1] = reverse_i(idxs[1]);
    } else if numvecs > 2 {
        for i in 0..numvecs / 2 {
            let hi = numvecs - i - 1;
            keys[hi] = reverse_f(keys[hi]);
            idxs[hi] = reverse_i(idxs[hi]);
            let (mut ki, mut kh) = (keys[i], keys[hi]);
            let (mut xi, mut xh) = (idxs[i], idxs[hi]);
            coex(&mut ki, &mut kh, &mut xi, &mut xh);
            keys[i] = ki;
            keys[hi] = kh;
            idxs[i] = xi;
            idxs[hi] = xh;
            keys[hi] = reverse_f(keys[hi]);
            idxs[hi] = reverse_i(idxs[hi]);
        }
    }
    bitonic_clean_n_vec(keys, idxs, numvecs);
    for i in 0..numvecs {
        bitonic_merge_reg_4lanes(&mut keys[i], &mut idxs[i]);
    }
}

/// `bitonic_fullmerge_n_vec`: iteratively merge blocks of size 2, 4, ...,
/// numvecs.
fn bitonic_fullmerge_n_vec(keys: &mut [Reg], idxs: &mut [IReg], numvecs: usize) {
    let mut numper = 2;
    while numper <= numvecs {
        let mut i = 0;
        while i < numvecs / numper {
            let base = i * numper;
            bitonic_merge_n_vec(
                &mut keys[base..base + numper],
                &mut idxs[base..base + numper],
                numper,
            );
            i += 1;
        }
        numper *= 2;
    }
}

/// `get_partial_loadmask(num_to_read)` for numlanes 4: lanes `< num_to_read`
/// are active.
#[inline]
fn partial_loadmask(num_to_read: usize) -> [bool; NUMLANES] {
    let mut m = [false; NUMLANES];
    for (i, slot) in m.iter_mut().enumerate() {
        *slot = i < num_to_read;
    }
    m
}

/// `argsort_n_vec<numvecs>`: the recursive bitonic argsort over `indices[..n]`,
/// gathering keys from `arr` via `indices`. Operates on the `n`-element window
/// starting at `indices`.
fn argsort_n_vec(arr: &[f64], indices: &mut [u64], n: usize, numvecs: usize) {
    if numvecs > 1 && n * 2 <= numvecs * NUMLANES {
        argsort_n_vec(arr, indices, n, numvecs / 2);
        return;
    }

    let mut keyvecs = vec![[0.0f64; NUMLANES]; numvecs];
    let mut idxvecs = vec![[0u64; NUMLANES]; numvecs];

    // Masks for the upper half (masked load/store).
    let half = numvecs / 2;
    let mut iomasks = vec![[false; NUMLANES]; numvecs - half];
    for (j, i) in (half..numvecs).enumerate() {
        let lo = i * NUMLANES;
        let num_to_read = n.saturating_sub(lo).min(NUMLANES);
        iomasks[j] = partial_loadmask(num_to_read);
    }

    // Unmasked lower half: load indices, gather keys.
    for i in 0..half {
        let lo = i * NUMLANES;
        for lane in 0..NUMLANES {
            let ix = indices[lo + lane];
            idxvecs[i][lane] = ix;
            keyvecs[i][lane] = arr[ix as usize];
        }
    }
    // Masked upper half: padded lanes get index = u64::MAX, key = +inf.
    for i in half..numvecs {
        let lo = i * NUMLANES;
        let m = iomasks[i - half];
        for lane in 0..NUMLANES {
            if m[lane] {
                let ix = indices[lo + lane];
                idxvecs[i][lane] = ix;
                keyvecs[i][lane] = arr[ix as usize];
            } else {
                idxvecs[i][lane] = u64::MAX;
                keyvecs[i][lane] = f64::INFINITY;
            }
        }
    }

    for i in 0..numvecs {
        sort_reg_4lanes(&mut keyvecs[i], &mut idxvecs[i]);
    }

    bitonic_fullmerge_n_vec(&mut keyvecs, &mut idxvecs, numvecs);

    // Store back: lower half unmasked, upper half masked.
    for (i, iv) in idxvecs.iter().enumerate().take(half) {
        let lo = i * NUMLANES;
        indices[lo..lo + NUMLANES].copy_from_slice(&iv[..NUMLANES]);
    }
    for i in half..numvecs {
        let lo = i * NUMLANES;
        let m = iomasks[i - half];
        for lane in 0..NUMLANES {
            if m[lane] {
                indices[lo + lane] = idxvecs[i][lane];
            }
        }
    }
}

/// `argsort_n<256>`: dispatch into the bitonic network with `numvecs = 256/4`.
fn argsort_n(arr: &[f64], indices: &mut [u64], n: usize) {
    let numvecs = NETWORK_MAX_N / NUMLANES; // 64
    argsort_n_vec(arr, indices, n, numvecs);
}

// --------------------------------------------------------------------------
// Vectorized quicksort partition + pivot (for N > 256)
// --------------------------------------------------------------------------

/// `comparison_func<vtype>` for the no-NaN double path is `a < b`.
#[inline]
fn less(a: f64, b: f64) -> bool {
    a < b
}

/// `sort_vec`/median helper: sort 4 doubles ascending via the same 4-lane
/// network (used by `get_pivot_64bit`'s median-of-4).
fn sort4(vals: Reg) -> Reg {
    let mut k = vals;
    let mut idx = [0u64, 1, 2, 3];
    sort_reg_4lanes(&mut k, &mut idx);
    k
}

/// `get_pivot_64bit` for numlanes==4: median-of-4 of evenly spaced samples.
fn get_pivot_64bit(arr: &[f64], arg: &[u64], left: usize, right: usize) -> f64 {
    if right - left >= NUMLANES {
        let size = (right - left) / NUMLANES;
        // vtype::set(a,b,c,d) reverses, but we only need the median element,
        // and sort_vec sorts the whole register, so element order in is moot.
        let rand_vec = [
            arr[arg[left + size] as usize],
            arr[arg[left + 2 * size] as usize],
            arr[arg[left + 3 * size] as usize],
            arr[arg[left + 4 * size] as usize],
        ];
        let sorted = sort4(rand_vec);
        sorted[2]
    } else {
        arr[arg[right] as usize]
    }
}

/// `min`/`max` with `comparison_func` (= `a < b`), matching x86-simd-sort's
/// `std::min(*x, v, comparison_func)` accumulation in the partition prologue.
#[inline]
fn cmin(a: f64, b: f64) -> f64 {
    if less(b, a) { b } else { a }
}
#[inline]
fn cmax(a: f64, b: f64) -> f64 {
    if less(a, b) { b } else { a }
}

/// `avx2_compressstore_lut64`: for a 4-lane `ge`-mask, return the lane
/// permutation placing `lt`-lanes (in original order) at the low end and
/// `ge`-lanes (REVERSED) at the high end. Matches the LUT generated in
/// `avx2-emu-funcs.hpp` (`avx2_compressstore_lut64_gen`).
#[inline]
fn compress_perm(ge_mask: u32) -> [usize; NUMLANES] {
    let mut lt = [0usize; NUMLANES];
    let mut ge = [0usize; NUMLANES];
    let (mut nlt, mut nge) = (0usize, 0usize);
    for j in 0..NUMLANES {
        if (ge_mask >> j) & 1 == 1 {
            ge[nge] = j;
            nge += 1;
        } else {
            lt[nlt] = j;
            nlt += 1;
        }
    }
    let mut perm = [0usize; NUMLANES];
    let mut p = 0;
    for &l in lt.iter().take(nlt) {
        perm[p] = l;
        p += 1;
    }
    // ge lanes reversed
    for k in 0..nge {
        perm[p] = ge[nge - 1 - k];
        p += 1;
    }
    perm
}

/// `partition_vec_avx2` + `double_compressstore`: partition the 4-lane index
/// vector `arg_vec` (with keys gathered from `arr`) around `pivot`, writing the
/// compressed result to BOTH `l_store..l_store+4` and `r_store_plus-4..r_store_plus`.
/// Returns `amount_ge_pivot`.
#[allow(clippy::too_many_arguments)]
fn partition_vec(
    arr: &[f64],
    arg: &mut [u64],
    l_store: usize,
    r_store_plus: usize,
    arg_vec: IReg,
    pivot: f64,
    smallest: &mut f64,
    biggest: &mut f64,
) -> usize {
    let mut keys = [0.0f64; NUMLANES];
    for lane in 0..NUMLANES {
        keys[lane] = arr[arg_vec[lane] as usize];
    }
    let mut ge_mask = 0u32;
    for (lane, &k) in keys.iter().enumerate() {
        if !less(k, pivot) {
            ge_mask |= 1 << lane;
        }
    }
    let amount_ge = ge_mask.count_ones() as usize;
    let perm = compress_perm(ge_mask);
    let mut temp = [0u64; NUMLANES];
    for (t, slot) in temp.iter_mut().enumerate() {
        *slot = arg_vec[perm[t]];
    }
    arg[l_store..l_store + NUMLANES].copy_from_slice(&temp);
    arg[r_store_plus - NUMLANES..r_store_plus].copy_from_slice(&temp);
    for &k in &keys {
        *smallest = cmin(*smallest, k);
        *biggest = cmax(*biggest, k);
    }
    amount_ge
}

/// `argpartition<vtype>` (non-unrolled core). Partitions `arg[left..right]` so
/// keys `< pivot` come first; returns the split index. Faithful port of the
/// AVX2 `argpartition` in `xss-common-argsort.h`.
fn argpartition_core(
    arr: &[f64],
    arg: &mut [u64],
    mut left: usize,
    mut right: usize,
    pivot: f64,
    smallest: &mut f64,
    biggest: &mut f64,
) -> usize {
    let rem = (right - left) % NUMLANES;
    for _ in 0..rem {
        let v = arr[arg[left] as usize];
        *smallest = cmin(*smallest, v);
        *biggest = cmax(*biggest, v);
        if !less(v, pivot) {
            right -= 1;
            arg.swap(left, right);
        } else {
            left += 1;
        }
    }
    if left == right {
        return left;
    }
    if right - left == NUMLANES {
        let mut av = [0u64; NUMLANES];
        av.copy_from_slice(&arg[left..left + NUMLANES]);
        let amt = partition_vec(
            arr,
            arg,
            left,
            left + NUMLANES,
            av,
            pivot,
            smallest,
            biggest,
        );
        return left + (NUMLANES - amt);
    }
    let mut avl = [0u64; NUMLANES];
    avl.copy_from_slice(&arg[left..left + NUMLANES]);
    let mut avr = [0u64; NUMLANES];
    avr.copy_from_slice(&arg[right - NUMLANES..right]);
    let mut r_store = right - NUMLANES;
    let mut l_store = left;
    left += NUMLANES;
    right -= NUMLANES;
    while right - left != 0 {
        let av;
        if (r_store + NUMLANES) - right < left - l_store {
            right -= NUMLANES;
            let mut a = [0u64; NUMLANES];
            a.copy_from_slice(&arg[right..right + NUMLANES]);
            av = a;
        } else {
            let mut a = [0u64; NUMLANES];
            a.copy_from_slice(&arg[left..left + NUMLANES]);
            av = a;
            left += NUMLANES;
        }
        let amt = partition_vec(
            arr,
            arg,
            l_store,
            r_store + NUMLANES,
            av,
            pivot,
            smallest,
            biggest,
        );
        r_store = r_store.saturating_sub(amt);
        l_store += NUMLANES - amt;
    }
    let amt = partition_vec(
        arr,
        arg,
        l_store,
        r_store + NUMLANES,
        avl,
        pivot,
        smallest,
        biggest,
    );
    l_store += NUMLANES - amt;
    let amt = partition_vec(
        arr,
        arg,
        l_store,
        l_store + NUMLANES,
        avr,
        pivot,
        smallest,
        biggest,
    );
    l_store += NUMLANES - amt;
    l_store
}

/// `argpartition_unrolled<4>`: the unrolled (num_unroll=4) AVX2 partition that
/// `argsort_`/`argselect_` invoke. Falls back to [`argpartition_core`] for
/// small ranges.
fn argpartition_unrolled(
    arr: &[f64],
    arg: &mut [u64],
    mut left: usize,
    mut right: usize,
    pivot: f64,
    smallest: &mut f64,
    biggest: &mut f64,
) -> usize {
    const NU: usize = 4;
    if right - left <= 8 * NU * NUMLANES {
        return argpartition_core(arr, arg, left, right, pivot, smallest, biggest);
    }
    let rem = (right - left) % (NU * NUMLANES);
    for _ in 0..rem {
        let v = arr[arg[left] as usize];
        *smallest = cmin(*smallest, v);
        *biggest = cmax(*biggest, v);
        if !less(v, pivot) {
            right -= 1;
            arg.swap(left, right);
        } else {
            left += 1;
        }
    }
    if left == right {
        return left;
    }
    let mut vl = [[0u64; NUMLANES]; NU];
    let mut vr = [[0u64; NUMLANES]; NU];
    for ii in 0..NU {
        vl[ii].copy_from_slice(&arg[left + NUMLANES * ii..left + NUMLANES * ii + NUMLANES]);
        let base = right - NUMLANES * (NU - ii);
        vr[ii].copy_from_slice(&arg[base..base + NUMLANES]);
    }
    let mut r_store = right - NUMLANES;
    let mut l_store = left;
    left += NU * NUMLANES;
    right -= NU * NUMLANES;
    while right - left != 0 {
        let mut cur = [[0u64; NUMLANES]; NU];
        if (r_store + NUMLANES) - right < left - l_store {
            right -= NU * NUMLANES;
            for (ii, c) in cur.iter_mut().enumerate() {
                let b = right + ii * NUMLANES;
                c.copy_from_slice(&arg[b..b + NUMLANES]);
            }
        } else {
            for (ii, c) in cur.iter_mut().enumerate() {
                let b = left + ii * NUMLANES;
                c.copy_from_slice(&arg[b..b + NUMLANES]);
            }
            left += NU * NUMLANES;
        }
        for c in &cur {
            let amt = partition_vec(
                arr,
                arg,
                l_store,
                r_store + NUMLANES,
                *c,
                pivot,
                smallest,
                biggest,
            );
            l_store += NUMLANES - amt;
            r_store = r_store.saturating_sub(amt);
        }
    }
    // r_store only ever feeds the `r_store + NUMLANES` store address, which the
    // algorithm keeps in-bounds. The trailing decrements can conceptually pass
    // the array start (matching C's unsigned wraparound) but are never read
    // again, so saturate to satisfy the debug overflow check without altering
    // observed behaviour.
    for v in &vl {
        let amt = partition_vec(
            arr,
            arg,
            l_store,
            r_store + NUMLANES,
            *v,
            pivot,
            smallest,
            biggest,
        );
        l_store += NUMLANES - amt;
        r_store = r_store.saturating_sub(amt);
    }
    for v in &vr {
        let amt = partition_vec(
            arr,
            arg,
            l_store,
            r_store + NUMLANES,
            *v,
            pivot,
            smallest,
            biggest,
        );
        l_store += NUMLANES - amt;
        r_store = r_store.saturating_sub(amt);
    }
    l_store
}

/// `argsort_` recursion (no OpenMP): bitonic base case for `<= 256`, else
/// median pivot + partition + recurse both sides.
fn argsort_rec(arr: &[f64], arg: &mut [u64], left: usize, right: usize, max_iters: i64) {
    if max_iters <= 0 {
        std_argsort_range(arr, arg, left, right + 1);
        return;
    }
    if right + 1 - left <= NETWORK_MAX_N {
        let n = right + 1 - left;
        // Operate on the sub-window arg[left..left+n].
        let mut window: Vec<u64> = arg[left..left + n].to_vec();
        argsort_n(arr, &mut window, n);
        arg[left..left + n].copy_from_slice(&window);
        return;
    }
    let pivot = get_pivot_64bit(arr, arg, left, right);
    let mut smallest = f64::INFINITY;
    let mut biggest = f64::NEG_INFINITY;
    let pivot_index = argpartition_unrolled(
        arr,
        arg,
        left,
        right + 1,
        pivot,
        &mut smallest,
        &mut biggest,
    );
    if pivot != smallest {
        argsort_rec(arr, arg, left, pivot_index - 1, max_iters - 1);
    }
    if pivot != biggest {
        argsort_rec(arr, arg, pivot_index, right, max_iters - 1);
    }
}

/// `argselect_` recursion: like [`argsort_rec`] but recurses only into the
/// side containing `pos`.
fn argselect_rec(
    arr: &[f64],
    arg: &mut [u64],
    pos: usize,
    left: usize,
    right: usize,
    max_iters: i64,
) {
    if max_iters <= 0 {
        std_argsort_range(arr, arg, left, right + 1);
        return;
    }
    if right + 1 - left <= NETWORK_MAX_N {
        let n = right + 1 - left;
        let mut window: Vec<u64> = arg[left..left + n].to_vec();
        argsort_n(arr, &mut window, n);
        arg[left..left + n].copy_from_slice(&window);
        return;
    }
    let pivot = get_pivot_64bit(arr, arg, left, right);
    let mut smallest = f64::INFINITY;
    let mut biggest = f64::NEG_INFINITY;
    let pivot_index = argpartition_unrolled(
        arr,
        arg,
        left,
        right + 1,
        pivot,
        &mut smallest,
        &mut biggest,
    );
    if pivot != smallest && pos < pivot_index {
        argselect_rec(arr, arg, pos, left, pivot_index - 1, max_iters - 1);
    } else if pivot != biggest && pos >= pivot_index {
        argselect_rec(arr, arg, pos, pivot_index, right, max_iters - 1);
    }
}

/// `std_argsort` fallback used when quicksort makes no progress (`max_iters`
/// exhausted). x86-simd-sort calls `std::sort(arg+left, arg+right, comp)` here
/// (for BOTH argsort and argselect — `argselect_`'s fallback is a full sort,
/// not `nth_element`). To stay bit-identical we port libstdc++'s `std::sort`
/// (introsort: median-of-3 quicksort, heapsort depth fallback, then a final
/// insertion-sort pass). See [`std_sort_range`].
fn std_argsort_range(arr: &[f64], arg: &mut [u64], left: usize, right: usize) {
    std_sort_range(arr, arg, left, right);
}

// --------------------------------------------------------------------------
// libstdc++ std::sort (introsort) + std::nth_element (introselect) ports.
// Used for (a) the AVX2 depth-limit fallback (ascending `arr[a] < arr[b]`),
// and (b) the NaN-containing input fallback (NaN-last comparator). Both are
// generic over a comparator `cmp(a, b)` so the tie order matches GCC's
// `std::sort` / `std::nth_element` exactly.
// --------------------------------------------------------------------------

const STD_INSERTION_THRESHOLD: usize = 16;

/// libstdc++ `__lg`: floor(log2(n)).
#[inline]
fn std_lg(mut n: usize) -> u32 {
    let mut k = 0;
    while n > 1 {
        n >>= 1;
        k += 1;
    }
    k
}

/// libstdc++ `__move_median_to_first(result, a, b, c)`.
fn move_median_to_first<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    cmp: &C,
) {
    if cmp(v[a], v[b]) {
        if cmp(v[b], v[c]) {
            v.swap(result, b);
        } else if cmp(v[a], v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if cmp(v[a], v[c]) {
        v.swap(result, a);
    } else if cmp(v[b], v[c]) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

/// libstdc++ `__unguarded_partition(first, last, pivot)`.
fn unguarded_partition<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    cmp: &C,
) -> usize {
    loop {
        while cmp(v[first], v[pivot]) {
            first += 1;
        }
        last -= 1;
        while cmp(v[pivot], v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

/// libstdc++ `__unguarded_partition_pivot(first, last)`.
fn unguarded_partition_pivot<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    first: usize,
    last: usize,
    cmp: &C,
) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, cmp);
    unguarded_partition(v, first + 1, last, first, cmp)
}

/// libstdc++ `__insertion_sort(first, last)`.
fn std_insertion_sort<C: Fn(u64, u64) -> bool>(v: &mut [u64], first: usize, last: usize, cmp: &C) {
    if first == last {
        return;
    }
    let mut i = first + 1;
    while i != last {
        if cmp(v[i], v[first]) {
            let val = v[i];
            v.copy_within(first..i, first + 1);
            v[first] = val;
        } else {
            let val = v[i];
            let mut j = i;
            while cmp(val, v[j - 1]) {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = val;
        }
        i += 1;
    }
}

/// libstdc++ `__unguarded_insertion_sort(first, last)`.
fn std_unguarded_insertion_sort<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    first: usize,
    last: usize,
    cmp: &C,
) {
    let mut i = first;
    while i != last {
        let val = v[i];
        let mut j = i;
        while cmp(val, v[j - 1]) {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = val;
        i += 1;
    }
}

/// libstdc++ `__final_insertion_sort(first, last)`.
fn std_final_insertion_sort<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    first: usize,
    last: usize,
    cmp: &C,
) {
    if last - first > STD_INSERTION_THRESHOLD {
        std_insertion_sort(v, first, first + STD_INSERTION_THRESHOLD, cmp);
        std_unguarded_insertion_sort(v, first + STD_INSERTION_THRESHOLD, last, cmp);
    } else {
        std_insertion_sort(v, first, last, cmp);
    }
}

/// libstdc++ `__push_heap` (relative-index form over `v[first..]`).
fn heap_push<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    first: usize,
    mut hole: usize,
    top: usize,
    value: u64,
    cmp: &C,
) {
    let mut parent = if hole == 0 { 0 } else { (hole - 1) / 2 };
    while hole > top && cmp(v[first + parent], value) {
        v[first + hole] = v[first + parent];
        hole = parent;
        if hole == 0 {
            break;
        }
        parent = (hole - 1) / 2;
    }
    v[first + hole] = value;
}

/// libstdc++ `__adjust_heap`.
fn heap_adjust<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    first: usize,
    hole_in: usize,
    len: usize,
    value: u64,
    cmp: &C,
) {
    let top = hole_in;
    let mut hole = hole_in;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if cmp(v[first + second], v[first + second - 1]) {
            second -= 1;
        }
        v[first + hole] = v[first + second];
        hole = second;
    }
    if len.is_multiple_of(2) && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[first + hole] = v[first + second - 1];
        hole = second - 1;
    }
    heap_push(v, first, hole, top, value, cmp);
}

/// libstdc++ `__make_heap`.
fn make_heap<C: Fn(u64, u64) -> bool>(v: &mut [u64], first: usize, last: usize, cmp: &C) {
    let len = last - first;
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[first + parent];
        heap_adjust(v, first, parent, len, value, cmp);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// libstdc++ `__pop_heap`: move max to `result`, restore heap of `[first,last)`.
fn pop_heap<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    first: usize,
    last: usize,
    result: usize,
    cmp: &C,
) {
    let value = v[result];
    v[result] = v[first];
    heap_adjust(v, first, 0, last - first, value, cmp);
}

/// libstdc++ `__sort_heap`.
fn sort_heap<C: Fn(u64, u64) -> bool>(v: &mut [u64], first: usize, mut last: usize, cmp: &C) {
    while last - first > 1 {
        last -= 1;
        pop_heap(v, first, last, last, cmp);
    }
}

/// libstdc++ `partial_sort(first, last, last)` == heapsort of `[first,last)`.
fn std_heap_sort<C: Fn(u64, u64) -> bool>(v: &mut [u64], first: usize, last: usize, cmp: &C) {
    make_heap(v, first, last, cmp);
    sort_heap(v, first, last, cmp);
}

/// libstdc++ `__heap_select(first, middle, last)`.
fn heap_select<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    first: usize,
    middle: usize,
    last: usize,
    cmp: &C,
) {
    make_heap(v, first, middle, cmp);
    let mut i = middle;
    while i < last {
        if cmp(v[i], v[first]) {
            pop_heap(v, first, middle, i, cmp);
        }
        i += 1;
    }
}

/// libstdc++ `__introsort_loop`.
fn introsort_loop<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    first: usize,
    mut last: usize,
    mut depth_limit: u32,
    cmp: &C,
) {
    while last - first > STD_INSERTION_THRESHOLD {
        if depth_limit == 0 {
            std_heap_sort(v, first, last, cmp);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(v, first, last, cmp);
        introsort_loop(v, cut, last, depth_limit, cmp);
        last = cut;
    }
}

/// libstdc++ `std::sort(v[first..last], cmp)` (introsort).
fn std_sort_cmp<C: Fn(u64, u64) -> bool>(v: &mut [u64], first: usize, last: usize, cmp: &C) {
    if first != last {
        introsort_loop(v, first, last, 2 * std_lg(last - first), cmp);
        std_final_insertion_sort(v, first, last, cmp);
    }
}

/// libstdc++ `std::nth_element(first, nth, last, cmp)` (introselect).
fn nth_element_cmp<C: Fn(u64, u64) -> bool>(
    v: &mut [u64],
    mut first: usize,
    nth: usize,
    mut last: usize,
    cmp: &C,
) {
    if first == last || nth == last {
        return;
    }
    let mut depth_limit = 2 * std_lg(last - first);
    while last - first > 3 {
        if depth_limit == 0 {
            heap_select(v, first, nth + 1, last, cmp);
            v.swap(first, nth);
            return;
        }
        depth_limit -= 1;
        let mid = first + (last - first) / 2;
        move_median_to_first(v, first, first + 1, mid, last - 1, cmp);
        let cut = unguarded_partition(v, first + 1, last, first, cmp);
        if cut <= nth {
            first = cut;
        } else {
            last = cut;
        }
    }
    std_insertion_sort(v, first, last, cmp);
}

/// Ascending key comparator (`arr[a] < arr[b]`) for the no-NaN depth fallback.
fn std_sort_range(arr: &[f64], v: &mut [u64], first: usize, last: usize) {
    let cmp = |a: u64, b: u64| arr[a as usize] < arr[b as usize];
    std_sort_cmp(v, first, last, &cmp);
}

// --------------------------------------------------------------------------
// NaN detection + std::sort NaN-last fallback
// --------------------------------------------------------------------------

#[inline]
fn array_has_nan(arr: &[f64]) -> bool {
    arr.iter().any(|x| x.is_nan())
}

/// NaN-last comparator matching x86-simd-sort `std_argsort_withnan`:
/// `a < b` for non-NaN; NaN sorts as "greater".
#[inline]
fn nan_last_lt(va: f64, vb: f64) -> bool {
    if !va.is_nan() && !vb.is_nan() {
        va < vb
    } else {
        // va < vb under NaN-last is true iff va is non-NaN (and vb is NaN);
        // if va is NaN it is never "less".
        !va.is_nan()
    }
}

/// `std::is_sorted(arr)` ascending early-exit check (no-NaN path).
fn is_sorted_keys(arr: &[f64], arg: &[u64]) -> bool {
    // After iota init, arg[i] == i, so this checks arr itself in order.
    arg.windows(2)
        .all(|w| !less(arr[w[1] as usize], arr[w[0] as usize]))
}

// --------------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------------

/// NumPy 2.4.5-bit-identical `np.argsort(a)` for `f64` on an AVX2 (x86-64)
/// build.
///
/// Reproduces `x86simdsortStatic::argsort` (the AVX2 path numpy 2.4.5 actually
/// dispatches to for `double` on AVX2 CPUs), including exact tie order.
///
/// Guarantees:
/// * **No-NaN inputs**: bit-identical to `np.argsort(a, kind='quicksort')`.
/// * **NaN inputs**: bit-identical to numpy, reproduced via the libstdc++
///   `std::sort` introsort with numpy's NaN-last comparator (the path numpy
///   takes for NaN-containing float arrays). NaN indices sort to the end.
#[must_use]
pub fn argsort_numpy(a: &[f64]) -> Vec<u64> {
    let n = a.len();
    let mut arg: Vec<u64> = (0..n as u64).collect();
    if n <= 1 {
        return arg;
    }
    if array_has_nan(a) {
        // `std_argsort_withnan`: std::sort with the NaN-last comparator.
        let cmp = |i: u64, j: u64| nan_last_lt(a[i as usize], a[j as usize]);
        std_sort_cmp(&mut arg, 0, n, &cmp);
        return arg;
    }
    if is_sorted_keys(a, &arg) {
        return arg;
    }
    let max_iters = 2 * (n as f64).log2() as i64;
    argsort_rec(a, &mut arg, 0, n - 1, max_iters);
    arg
}

/// NumPy 2.4.5-bit-identical `np.argpartition(a, kth)` for `f64` on an AVX2
/// (x86-64) build (single `kth`, the KNN case).
///
/// Reproduces `x86simdsortStatic::argselect`. Returns `None` if `kth >= len`.
///
/// Guarantees mirror [`argsort_numpy`]: bit-identical to `np.argpartition(a, kth)`
/// for both no-NaN and NaN-containing inputs (the NaN path is reproduced via the
/// libstdc++ `std::nth_element` introselect with numpy's NaN-last comparator).
#[must_use]
pub fn argpartition_numpy(a: &[f64], kth: usize) -> Option<Vec<u64>> {
    let n = a.len();
    if kth >= n {
        return None;
    }
    let mut arg: Vec<u64> = (0..n as u64).collect();
    if n <= 1 {
        return Some(arg);
    }
    if array_has_nan(a) {
        // `std_argselect_withnan`: std::nth_element with the NaN-last comparator.
        let cmp = |i: u64, j: u64| nan_last_lt(a[i as usize], a[j as usize]);
        nth_element_cmp(&mut arg, 0, kth, n, &cmp);
        return Some(arg);
    }
    let max_iters = 2 * (n as f64).log2() as i64;
    argselect_rec(a, &mut arg, kth, 0, n - 1, max_iters);
    Some(arg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argsort_basic() {
        let a = [3.0, 1.0, 4.0, 1.0, 5.0];
        let idx = argsort_numpy(&a);
        let picked: Vec<f64> = idx.iter().map(|&i| a[i as usize]).collect();
        assert_eq!(picked, vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn argsort_nan_last() {
        let a = [2.0, f64::NAN, 0.5, 5.0];
        let idx = argsort_numpy(&a);
        assert_eq!(a[idx[0] as usize], 0.5);
        assert_eq!(a[idx[1] as usize], 2.0);
        assert_eq!(a[idx[2] as usize], 5.0);
        assert!(a[idx[3] as usize].is_nan());
    }

    #[test]
    fn argpartition_out_of_range() {
        assert!(argpartition_numpy(&[1.0, 2.0], 2).is_none());
    }

    #[test]
    fn knn_shaped_case() {
        // #2141 fixture: argpartition(dist, k-1)[:k] then argsort(dist[sel]).
        let dist = [0.0, 1.0, 1.0, 1.0, 1.0, 0.99998082, 0.99998082];
        let k = 4;
        let ap = argpartition_numpy(&dist, k - 1).unwrap();
        assert_eq!(ap, vec![0, 5, 6, 1, 3, 2, 4]);
        let sel: Vec<u64> = ap[..k].to_vec();
        let sub: Vec<f64> = sel.iter().map(|&i| dist[i as usize]).collect();
        let order = argsort_numpy(&sub);
        let neigh: Vec<u64> = order.iter().map(|&o| sel[o as usize]).collect();
        assert_eq!(neigh, vec![0, 5, 6, 1]);
    }

    // The following expected vectors are LIVE-ORACLE values: they were produced
    // by numpy 2.4.5 `np.argsort(a, kind='quicksort')` / `np.argpartition(a, k)`
    // on the system interpreter (python3 -c "import numpy; numpy.__version__" ==
    // 2.4.5), NOT literal-copied from ferray (R-CHAR-3). Each comment records the
    // exact oracle call.

    #[test]
    fn argsort_network_heavy_tie_oracle() {
        // np.argsort([0,1,1,0,2,2,1,0,2,1,0,1,2,0,1,2,0,1,2,0], kind='quicksort')
        let a = [
            0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
            1.0, 2.0, 0.0,
        ];
        let got = argsort_numpy(&a);
        assert_eq!(
            got,
            vec![
                0, 3, 7, 10, 13, 16, 19, 6, 14, 9, 11, 1, 2, 17, 4, 5, 15, 12, 8, 18
            ]
        );
    }

    #[test]
    fn argsort_allequal_network_is_identity_oracle() {
        // np.argsort([1,1,1,1,1], kind='quicksort') == [0,1,2,3,4] (stable identity)
        let a = [1.0; 5];
        assert_eq!(argsort_numpy(&a), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn argpartition_large_quicksort_path_oracle() {
        // a[i] = i % 3 for i in 0..300 (exercises the >256 vectorized quicksort).
        // np.argpartition(a, 150)[:8] == [0, 3, 6, 297, 30, 33, 27, 24]
        let a: Vec<f64> = (0..300).map(|i| (i % 3) as f64).collect();
        let got = argpartition_numpy(&a, 150).unwrap();
        assert_eq!(&got[..8], &[0, 3, 6, 297, 30, 33, 27, 24]);
        // kth element is in its final sorted position.
        let kv = a[got[150] as usize];
        for &i in &got[..150] {
            assert!(a[i as usize] <= kv);
        }
        for &i in &got[151..] {
            assert!(a[i as usize] >= kv);
        }
    }

    #[test]
    fn argsort_nan_oracle() {
        // np.argsort([2, nan, 0.5, 5, nan, 1], kind='quicksort') == [2,5,0,3,1,4]
        let a = [2.0, f64::NAN, 0.5, 5.0, f64::NAN, 1.0];
        assert_eq!(argsort_numpy(&a), vec![2, 5, 0, 3, 1, 4]);
    }

    #[test]
    fn argpartition_nan_oracle() {
        // np.argpartition([2, nan, 0.5, 5, nan, 1], 2) == [5, 2, 0, 3, 4, 1]
        let a = [2.0, f64::NAN, 0.5, 5.0, f64::NAN, 1.0];
        assert_eq!(argpartition_numpy(&a, 2).unwrap(), vec![5, 2, 0, 3, 4, 1]);
    }

    #[test]
    fn argsort_permutation_invariant() {
        // Applying the argsort permutation must yield a non-decreasing sequence
        // (sanity property, independent of the oracle).
        let a = [3.5, -1.0, 3.5, 0.0, 9.9, -1.0, 2.2, 2.2, 8.0, 0.0];
        let idx = argsort_numpy(&a);
        let picked: Vec<f64> = idx.iter().map(|&i| a[i as usize]).collect();
        assert!(picked.windows(2).all(|w| w[0] <= w[1]));
    }
}
