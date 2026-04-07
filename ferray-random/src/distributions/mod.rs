// ferray-random: Distribution sampling methods on Generator
//
// Each submodule implements distribution sampling as methods on Generator<B>.

pub mod discrete;
pub mod exponential;
pub mod gamma;
pub mod misc_continuous;
pub mod multivariate;
pub mod normal;
pub mod uniform;
pub(crate) mod ziggurat;
