// ferrum-core: Indexing module (REQ-12 through REQ-15a)
//
// Provides basic indexing (integer + slice → views), advanced indexing
// (fancy + boolean → copies), and extended indexing functions (take, put,
// choose, compress, etc.).

pub mod advanced;
pub mod basic;
pub mod extended;
