// ferrum-ma: Masked reductions (REQ-4)
//
// mean, sum, min, max, var, std, count — all skip masked elements.

use ferrum_core::dimension::Dimension;
use ferrum_core::dtype::Element;
use ferrum_core::error::{FerrumError, FerrumResult};
use num_traits::Float;

use crate::MaskedArray;

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy,
    D: Dimension,
{
    /// Count the number of unmasked (valid) elements.
    ///
    /// # Errors
    /// This function does not currently error but returns `Result` for API
    /// consistency.
    pub fn count(&self) -> FerrumResult<usize> {
        let n = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .count();
        Ok(n)
    }
}

impl<T, D> MaskedArray<T, D>
where
    T: Element + Float,
    D: Dimension,
{
    /// Compute the sum of unmasked elements.
    ///
    /// Returns zero if all elements are masked.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn sum(&self) -> FerrumResult<T> {
        let zero = num_traits::zero::<T>();
        let s = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .fold(zero, |acc, (v, _)| acc + *v);
        Ok(s)
    }

    /// Compute the mean of unmasked elements.
    ///
    /// Returns `NaN` if no elements are unmasked.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn mean(&self) -> FerrumResult<T> {
        let zero = num_traits::zero::<T>();
        let one: T = num_traits::one();
        let (sum, count) = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .fold((zero, 0usize), |(s, c), (v, _)| (s + *v, c + 1));
        if count == 0 {
            return Ok(T::nan());
        }
        Ok(sum / T::from(count).unwrap_or(one))
    }

    /// Compute the minimum of unmasked elements.
    ///
    /// # Errors
    /// Returns `FerrumError::InvalidValue` if no elements are unmasked.
    pub fn min(&self) -> FerrumResult<T> {
        self.data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .map(|(v, _)| *v)
            .fold(None, |acc: Option<T>, v| {
                Some(match acc {
                    Some(a) => if v < a { v } else { a },
                    None => v,
                })
            })
            .ok_or_else(|| {
                FerrumError::invalid_value("min: all elements are masked")
            })
    }

    /// Compute the maximum of unmasked elements.
    ///
    /// # Errors
    /// Returns `FerrumError::InvalidValue` if no elements are unmasked.
    pub fn max(&self) -> FerrumResult<T> {
        self.data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .map(|(v, _)| *v)
            .fold(None, |acc: Option<T>, v| {
                Some(match acc {
                    Some(a) => if v > a { v } else { a },
                    None => v,
                })
            })
            .ok_or_else(|| {
                FerrumError::invalid_value("max: all elements are masked")
            })
    }

    /// Compute the variance of unmasked elements (population variance, ddof=0).
    ///
    /// Returns `NaN` if no elements are unmasked.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn var(&self) -> FerrumResult<T> {
        let mean = self.mean()?;
        if mean.is_nan() {
            return Ok(T::nan());
        }
        let zero = num_traits::zero::<T>();
        let one: T = num_traits::one();
        let (sum_sq, count) = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .fold((zero, 0usize), |(s, c), (v, _)| {
                let d = *v - mean;
                (s + d * d, c + 1)
            });
        if count == 0 {
            return Ok(T::nan());
        }
        Ok(sum_sq / T::from(count).unwrap_or(one))
    }

    /// Compute the standard deviation of unmasked elements (population, ddof=0).
    ///
    /// Returns `NaN` if no elements are unmasked.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn std(&self) -> FerrumResult<T> {
        Ok(self.var()?.sqrt())
    }
}
