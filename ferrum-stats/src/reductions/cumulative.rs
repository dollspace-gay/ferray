// ferrum-stats: Cumulative operations — cumsum, cumprod, nancumsum, nancumprod (REQ-2a, REQ-2b)
//
// These are re-exports from ferrum-ufunc, gathered here for module completeness.
// The actual implementations live in the parent mod.rs and nan_aware.rs.

// This module exists for organizational completeness. The actual cumulative
// functions are re-exported from the parent `reductions` module (cumsum, cumprod)
// and from `nan_aware` (nancumsum, nancumprod).
//
// Users can access them via:
//   ferrum_stats::cumsum, ferrum_stats::cumprod
//   ferrum_stats::nancumsum, ferrum_stats::nancumprod
