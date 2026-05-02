//! Standard arithmetic operator overloads for polynomial types (#248).
//!
//! Each polynomial basis already exposes fallible `Poly::add`, `Poly::sub`,
//! and `Poly::mul` methods that return `Result<Self, FerrayError>` because
//! a domain/window mismatch between operands is rejected. The `std::ops`
//! traits cannot return `Result`, so the operator forms here panic on
//! mismatch — matching the convention used by `ndarray` for shape errors
//! in arithmetic operators. Code that wants to recover from mismatches
//! should keep using the fallible `Poly::*` methods.
//!
//! `Neg` doesn't have a `Poly`-trait counterpart; it negates each coefficient
//! while preserving the domain/window mapping.
//!
//! Both owned (`T op T`) and borrowed (`&T op &T`) forms are provided for
//! every basis: `Polynomial`, `Chebyshev`, `Legendre`, `Hermite`, `HermiteE`,
//! and `Laguerre`.

use crate::traits::Poly;

macro_rules! impl_poly_ops {
    ($ty:ident, $modname:ident) => {
        // ----- Add -----
        impl ::core::ops::Add<&crate::$modname::$ty> for &crate::$modname::$ty {
            type Output = crate::$modname::$ty;
            fn add(self, rhs: &crate::$modname::$ty) -> Self::Output {
                <crate::$modname::$ty as Poly>::add(self, rhs)
                    .expect(concat!(stringify!($ty), ": domain/window mismatch in `+`",))
            }
        }

        impl ::core::ops::Add for crate::$modname::$ty {
            type Output = crate::$modname::$ty;
            fn add(self, rhs: crate::$modname::$ty) -> Self::Output {
                <crate::$modname::$ty as Poly>::add(&self, &rhs)
                    .expect(concat!(stringify!($ty), ": domain/window mismatch in `+`",))
            }
        }

        // ----- Sub -----
        impl ::core::ops::Sub<&crate::$modname::$ty> for &crate::$modname::$ty {
            type Output = crate::$modname::$ty;
            fn sub(self, rhs: &crate::$modname::$ty) -> Self::Output {
                <crate::$modname::$ty as Poly>::sub(self, rhs)
                    .expect(concat!(stringify!($ty), ": domain/window mismatch in `-`",))
            }
        }

        impl ::core::ops::Sub for crate::$modname::$ty {
            type Output = crate::$modname::$ty;
            fn sub(self, rhs: crate::$modname::$ty) -> Self::Output {
                <crate::$modname::$ty as Poly>::sub(&self, &rhs)
                    .expect(concat!(stringify!($ty), ": domain/window mismatch in `-`",))
            }
        }

        // ----- Mul -----
        impl ::core::ops::Mul<&crate::$modname::$ty> for &crate::$modname::$ty {
            type Output = crate::$modname::$ty;
            fn mul(self, rhs: &crate::$modname::$ty) -> Self::Output {
                <crate::$modname::$ty as Poly>::mul(self, rhs)
                    .expect(concat!(stringify!($ty), ": domain/window mismatch in `*`",))
            }
        }

        impl ::core::ops::Mul for crate::$modname::$ty {
            type Output = crate::$modname::$ty;
            fn mul(self, rhs: crate::$modname::$ty) -> Self::Output {
                <crate::$modname::$ty as Poly>::mul(&self, &rhs)
                    .expect(concat!(stringify!($ty), ": domain/window mismatch in `*`",))
            }
        }

        // ----- Neg -----
        impl ::core::ops::Neg for &crate::$modname::$ty {
            type Output = crate::$modname::$ty;
            fn neg(self) -> Self::Output {
                let coeffs: Vec<f64> = <Self::Output as Poly>::coeffs(self)
                    .iter()
                    .map(|c| -c)
                    .collect();
                self.with_same_mapping(coeffs)
            }
        }

        impl ::core::ops::Neg for crate::$modname::$ty {
            type Output = crate::$modname::$ty;
            fn neg(self) -> Self::Output {
                let coeffs: Vec<f64> = <Self::Output as Poly>::coeffs(&self)
                    .iter()
                    .map(|c| -c)
                    .collect();
                self.with_same_mapping(coeffs)
            }
        }
    };
}

impl_poly_ops!(Polynomial, power);
impl_poly_ops!(Chebyshev, chebyshev);
impl_poly_ops!(Legendre, legendre);
impl_poly_ops!(Hermite, hermite);
impl_poly_ops!(HermiteE, hermite_e);
impl_poly_ops!(Laguerre, laguerre);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chebyshev, Hermite, HermiteE, Laguerre, Legendre, Polynomial};

    #[test]
    fn polynomial_add_owned_and_ref() {
        let p = Polynomial::new(&[1.0, 2.0]);
        let q = Polynomial::new(&[3.0, 4.0, 5.0]);
        let owned = p.clone() + q.clone();
        let by_ref = &p + &q;
        assert_eq!(owned.coeffs(), by_ref.coeffs());
        assert_eq!(owned.coeffs(), vec![4.0, 6.0, 5.0]);
    }

    #[test]
    fn polynomial_sub() {
        let p = Polynomial::new(&[5.0, 6.0, 7.0]);
        let q = Polynomial::new(&[1.0, 2.0]);
        let r = &p - &q;
        assert_eq!(r.coeffs(), vec![4.0, 4.0, 7.0]);
    }

    #[test]
    fn polynomial_mul() {
        // (1 + x) * (1 + x) = 1 + 2x + x^2
        let p = Polynomial::new(&[1.0, 1.0]);
        let r = &p * &p;
        for (got, want) in r.coeffs().iter().zip([1.0, 2.0, 1.0].iter()) {
            assert!((got - want).abs() < 1e-14);
        }
    }

    #[test]
    fn polynomial_neg_owned_and_ref() {
        let p = Polynomial::new(&[1.0, -2.0, 3.0]);
        let neg_owned = -p.clone();
        let neg_ref = -&p;
        assert_eq!(neg_owned.coeffs(), vec![-1.0, 2.0, -3.0]);
        assert_eq!(neg_ref.coeffs(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn polynomial_neg_then_add_is_sub() {
        let p = Polynomial::new(&[10.0, 20.0]);
        let q = Polynomial::new(&[1.0, 2.0]);
        let lhs = &p + &(-&q);
        let rhs = &p - &q;
        assert_eq!(lhs.coeffs(), rhs.coeffs());
    }

    #[test]
    fn chebyshev_add_and_neg() {
        let p = Chebyshev::new(&[1.0, 2.0, 3.0]);
        let q = Chebyshev::new(&[4.0, 5.0]);
        let s = &p + &q;
        assert_eq!(s.coeffs(), vec![5.0, 7.0, 3.0]);
        let n = -&p;
        assert_eq!(n.coeffs(), vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn legendre_sub_and_mul() {
        let p = Legendre::new(&[1.0, 2.0]);
        let q = Legendre::new(&[3.0, 4.0]);
        let d = &p - &q;
        assert_eq!(d.coeffs(), vec![-2.0, -2.0]);
        // mul through power-basis pivot; check non-empty result
        let m = &p * &q;
        assert!(!m.coeffs().is_empty());
    }

    #[test]
    fn hermite_add() {
        let p = Hermite::new(&[1.0, 2.0]);
        let q = Hermite::new(&[3.0, 4.0]);
        let r = &p + &q;
        assert_eq!(r.coeffs(), vec![4.0, 6.0]);
    }

    #[test]
    fn hermite_e_neg() {
        let p = HermiteE::new(&[1.0, 2.0, 3.0]);
        let n = -&p;
        assert_eq!(n.coeffs(), vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn laguerre_add_owned() {
        let p = Laguerre::new(&[1.0, 2.0]);
        let q = Laguerre::new(&[3.0, 4.0]);
        let r = p + q;
        assert_eq!(r.coeffs(), vec![4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "domain/window mismatch")]
    fn polynomial_add_panics_on_domain_mismatch() {
        let p = Polynomial::new(&[1.0, 2.0]);
        let q = Polynomial::new(&[1.0, 2.0])
            .with_domain([-2.0, 2.0])
            .unwrap();
        let _ = &p + &q;
    }
}
