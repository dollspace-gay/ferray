//! Direct surface anchors for the polynomial crate-root aliases and
//! per-type inherent methods that were previously carried as exclusions.

use ferray_polynomial::Poly;
use num_complex::Complex;

const POLYNOMIAL_SURFACE_PATHS: &[&str] = &[
    "ferray_polynomial::Chebyshev",
    "ferray_polynomial::ChebyshevF32",
    "ferray_polynomial::ComplexPolynomial",
    "ferray_polynomial::ConvertBasis",
    "ferray_polynomial::FromPowerBasis",
    "ferray_polynomial::Hermite",
    "ferray_polynomial::HermiteE",
    "ferray_polynomial::HermiteEF32",
    "ferray_polynomial::HermiteF32",
    "ferray_polynomial::Laguerre",
    "ferray_polynomial::LaguerreF32",
    "ferray_polynomial::Legendre",
    "ferray_polynomial::LegendreF32",
    "ferray_polynomial::Poly",
    "ferray_polynomial::Polynomial",
    "ferray_polynomial::PolynomialF32",
    "ferray_polynomial::ToPowerBasis",
    "ferray_polynomial::cheb2poly",
    "ferray_polynomial::chebfromroots",
    "ferray_polynomial::chebgauss",
    "ferray_polynomial::chebinterpolate",
    "ferray_polynomial::chebline",
    "ferray_polynomial::chebmulx",
    "ferray_polynomial::chebpts1",
    "ferray_polynomial::chebpts2",
    "ferray_polynomial::chebweight",
    "ferray_polynomial::chebyshev::Chebyshev::fit_with_domain",
    "ferray_polynomial::chebyshev::Chebyshev::new",
    "ferray_polynomial::chebyshev::Chebyshev::with_domain",
    "ferray_polynomial::chebyshev::Chebyshev::with_window",
    "ferray_polynomial::herm2poly",
    "ferray_polynomial::herme2poly",
    "ferray_polynomial::hermefromroots",
    "ferray_polynomial::hermegauss",
    "ferray_polynomial::hermeline",
    "ferray_polynomial::hermemulx",
    "ferray_polynomial::hermfromroots",
    "ferray_polynomial::hermgauss",
    "ferray_polynomial::hermite::Hermite::fit_with_domain",
    "ferray_polynomial::hermite::Hermite::new",
    "ferray_polynomial::hermite::Hermite::with_domain",
    "ferray_polynomial::hermite::Hermite::with_window",
    "ferray_polynomial::hermite_e::HermiteE::fit_with_domain",
    "ferray_polynomial::hermite_e::HermiteE::new",
    "ferray_polynomial::hermite_e::HermiteE::with_domain",
    "ferray_polynomial::hermite_e::HermiteE::with_window",
    "ferray_polynomial::hermline",
    "ferray_polynomial::hermmulx",
    "ferray_polynomial::lag2poly",
    "ferray_polynomial::lagfromroots",
    "ferray_polynomial::laggauss",
    "ferray_polynomial::lagguass",
    "ferray_polynomial::lagline",
    "ferray_polynomial::lagmulx",
    "ferray_polynomial::laguerre::Laguerre::fit_with_domain",
    "ferray_polynomial::laguerre::Laguerre::new",
    "ferray_polynomial::laguerre::Laguerre::with_domain",
    "ferray_polynomial::laguerre::Laguerre::with_window",
    "ferray_polynomial::leg2poly",
    "ferray_polynomial::legendre::Legendre::fit_with_domain",
    "ferray_polynomial::legendre::Legendre::new",
    "ferray_polynomial::legendre::Legendre::with_domain",
    "ferray_polynomial::legendre::Legendre::with_window",
    "ferray_polynomial::legfromroots",
    "ferray_polynomial::leggauss",
    "ferray_polynomial::legline",
    "ferray_polynomial::legmulx",
    "ferray_polynomial::poly2cheb",
    "ferray_polynomial::poly2herm",
    "ferray_polynomial::poly2herme",
    "ferray_polynomial::poly2lag",
    "ferray_polynomial::poly2leg",
    "ferray_polynomial::polyfromroots",
    "ferray_polynomial::polygrid2d",
    "ferray_polynomial::polygrid3d",
    "ferray_polynomial::polyline",
    "ferray_polynomial::polymulx",
    "ferray_polynomial::polyval2d",
    "ferray_polynomial::polyval3d",
    "ferray_polynomial::polyvalfromroots",
    "ferray_polynomial::polyvander2d",
    "ferray_polynomial::polyvander3d",
    "ferray_polynomial::power::Polynomial::basis",
    "ferray_polynomial::power::Polynomial::compose",
    "ferray_polynomial::power::Polynomial::fit_with_domain",
    "ferray_polynomial::power::Polynomial::from_roots",
    "ferray_polynomial::power::Polynomial::identity",
    "ferray_polynomial::power::Polynomial::integ_with_bounds",
    "ferray_polynomial::power::Polynomial::with_domain",
    "ferray_polynomial::power::Polynomial::with_window",
    "ferray_polynomial::power_complex::ComplexPolynomial",
    "ferray_polynomial::power_complex::ComplexPolynomial::add",
    "ferray_polynomial::power_complex::ComplexPolynomial::coeffs",
    "ferray_polynomial::power_complex::ComplexPolynomial::degree",
    "ferray_polynomial::power_complex::ComplexPolynomial::deriv",
    "ferray_polynomial::power_complex::ComplexPolynomial::eval",
    "ferray_polynomial::power_complex::ComplexPolynomial::eval_complex",
    "ferray_polynomial::power_complex::ComplexPolynomial::from_roots",
    "ferray_polynomial::power_complex::ComplexPolynomial::integ",
    "ferray_polynomial::power_complex::ComplexPolynomial::mul",
    "ferray_polynomial::power_complex::ComplexPolynomial::new",
    "ferray_polynomial::power_complex::ComplexPolynomial::sub",
    "ferray_polynomial::power_complex::ComplexPolynomial::trim",
    "ferray_polynomial::power_complex::ComplexPolynomial::truncate",
    "ferray_polynomial::power_f32::PolynomialF32",
    "ferray_polynomial::power_f32::PolynomialF32::add",
    "ferray_polynomial::power_f32::PolynomialF32::coeffs",
    "ferray_polynomial::power_f32::PolynomialF32::degree",
    "ferray_polynomial::power_f32::PolynomialF32::deriv",
    "ferray_polynomial::power_f32::PolynomialF32::eval",
    "ferray_polynomial::power_f32::PolynomialF32::fit",
    "ferray_polynomial::power_f32::PolynomialF32::from_f64",
    "ferray_polynomial::power_f32::PolynomialF32::integ",
    "ferray_polynomial::power_f32::PolynomialF32::mul",
    "ferray_polynomial::power_f32::PolynomialF32::new",
    "ferray_polynomial::power_f32::PolynomialF32::roots",
    "ferray_polynomial::power_f32::PolynomialF32::sub",
    "ferray_polynomial::power_f32::PolynomialF32::to_f64",
    "ferray_polynomial::power_f32::PolynomialF32::trim",
    "ferray_polynomial::power_f32::PolynomialF32::truncate",
    "ferray_polynomial::power_f32::PolynomialF32::with_domain",
    "ferray_polynomial::power_f32::PolynomialF32::with_window",
    "ferray_polynomial::traits::FromPowerBasis",
    "ferray_polynomial::traits::ToPowerBasis",
];

fn assert_close(actual: f64, expected: f64, name: &str) {
    let diff = (actual - expected).abs();
    let tol = 1e-10 * expected.abs().max(1.0);
    assert!(
        diff <= tol,
        "{name}: got {actual}, expected {expected}, diff={diff}, tol={tol}"
    );
}

fn assert_vec_close(actual: &[f64], expected: &[f64], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert_close(got, want, &format!("{name}[{i}]"));
    }
}

#[test]
fn previous_polynomial_exclusion_paths_are_explicitly_anchored() {
    assert_eq!(POLYNOMIAL_SURFACE_PATHS.len(), 124);
    assert!(
        POLYNOMIAL_SURFACE_PATHS
            .iter()
            .all(|path| path.starts_with("ferray_polynomial::"))
    );
}

#[test]
fn crate_root_type_and_trait_reexports_are_usable_surface() {
    let p = ferray_polynomial::Polynomial::new(&[1.0, 2.0, 3.0]);
    assert_close(p.eval(2.0).unwrap(), 17.0, "root Polynomial eval");

    let c = ferray_polynomial::Chebyshev::new(&[0.0, 1.0]);
    let l = ferray_polynomial::Legendre::new(&[0.0, 1.0]);
    let la = ferray_polynomial::Laguerre::new(&[0.0, 1.0]);
    let h = ferray_polynomial::Hermite::new(&[0.0, 0.5]);
    let he = ferray_polynomial::HermiteE::new(&[0.0, 1.0]);
    assert_close(c.eval(0.25).unwrap(), 0.25, "root Chebyshev eval");
    assert_close(l.eval(0.25).unwrap(), 0.25, "root Legendre eval");
    assert_close(la.eval(0.25).unwrap(), 0.75, "root Laguerre eval");
    assert_close(h.eval(0.25).unwrap(), 0.25, "root Hermite eval");
    assert_close(he.eval(0.25).unwrap(), 0.25, "root HermiteE eval");

    let _: ferray_polynomial::ComplexPolynomial =
        ferray_polynomial::ComplexPolynomial::new(&[Complex::new(1.0, 0.0)]);
    let _: ferray_polynomial::PolynomialF32 =
        ferray_polynomial::PolynomialF32::new(&[1.0_f32, 2.0]);
    let _: ferray_polynomial::ChebyshevF32 = ferray_polynomial::ChebyshevF32::new(&[0.0_f32, 1.0]);
    let _: ferray_polynomial::HermiteF32 = ferray_polynomial::HermiteF32::new(&[0.0_f32, 0.5]);
    let _: ferray_polynomial::HermiteEF32 = ferray_polynomial::HermiteEF32::new(&[0.0_f32, 1.0]);
    let _: ferray_polynomial::LaguerreF32 = ferray_polynomial::LaguerreF32::new(&[0.0_f32, 1.0]);
    let _: ferray_polynomial::LegendreF32 = ferray_polynomial::LegendreF32::new(&[0.0_f32, 1.0]);

    let power_coeffs = ferray_polynomial::ToPowerBasis::to_power_basis(&c).unwrap();
    let _: ferray_polynomial::Chebyshev =
        <ferray_polynomial::Chebyshev as ferray_polynomial::FromPowerBasis>::from_power_basis(
            &power_coeffs,
        )
        .unwrap();
    let _: ferray_polynomial::Polynomial = ferray_polynomial::ConvertBasis::convert(&c).unwrap();

    let inner_power =
        ferray_polynomial::traits::ToPowerBasis::to_power_basis(&l).expect("inner trait path");
    let _: ferray_polynomial::Legendre =
        <ferray_polynomial::Legendre as ferray_polynomial::traits::FromPowerBasis>::from_power_basis(
            &inner_power,
        )
        .unwrap();
}

#[test]
fn crate_root_extra_reexports_match_documented_polynomial_contracts() {
    assert_eq!(ferray_polynomial::polyline(2.0, 3.0).coeffs(), &[2.0, 3.0]);
    assert_close(
        ferray_polynomial::chebline(2.0, 3.0).eval(0.0).unwrap(),
        2.0,
        "chebline",
    );
    assert_close(
        ferray_polynomial::legline(2.0, 3.0).eval(0.0).unwrap(),
        2.0,
        "legline",
    );
    assert_close(
        ferray_polynomial::lagline(2.0, 3.0).eval(0.0).unwrap(),
        5.0,
        "lagline",
    );
    assert_close(
        ferray_polynomial::hermline(2.0, 3.0).eval(0.0).unwrap(),
        2.0,
        "hermline",
    );
    assert_close(
        ferray_polynomial::hermeline(2.0, 3.0).eval(0.0).unwrap(),
        2.0,
        "hermeline",
    );

    assert_close(
        ferray_polynomial::polyfromroots(&[1.0, 2.0])
            .eval(1.0)
            .unwrap(),
        0.0,
        "polyfromroots",
    );
    assert_close(
        ferray_polynomial::chebfromroots(&[-1.0, 1.0])
            .unwrap()
            .eval(1.0)
            .unwrap(),
        0.0,
        "chebfromroots",
    );
    assert_close(
        ferray_polynomial::legfromroots(&[-1.0, 1.0])
            .unwrap()
            .eval(1.0)
            .unwrap(),
        0.0,
        "legfromroots",
    );
    assert_close(
        ferray_polynomial::lagfromroots(&[1.0, 2.0])
            .unwrap()
            .eval(1.0)
            .unwrap(),
        0.0,
        "lagfromroots",
    );
    assert_close(
        ferray_polynomial::hermfromroots(&[-1.0, 1.0])
            .unwrap()
            .eval(1.0)
            .unwrap(),
        0.0,
        "hermfromroots",
    );
    assert_close(
        ferray_polynomial::hermefromroots(&[-1.0, 1.0])
            .unwrap()
            .eval(1.0)
            .unwrap(),
        0.0,
        "hermefromroots",
    );

    assert_close(
        ferray_polynomial::polyvalfromroots(0.0, &[1.0, 2.0]),
        2.0,
        "polyvalfromroots",
    );
    assert_eq!(
        ferray_polynomial::polymulx(&[1.0, 2.0, 3.0]),
        vec![0.0, 1.0, 2.0, 3.0]
    );
    assert!(!ferray_polynomial::chebmulx(&[1.0, 0.0]).is_empty());
    assert!(!ferray_polynomial::legmulx(&[1.0, 0.0]).is_empty());
    assert!(!ferray_polynomial::lagmulx(&[1.0, 0.0]).is_empty());
    assert!(!ferray_polynomial::hermmulx(&[1.0, 0.0]).is_empty());
    assert!(!ferray_polynomial::hermemulx(&[1.0, 0.0]).is_empty());

    let v2 = ferray_polynomial::polyval2d(&[2.0], &[3.0], &[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
    assert_close(v2[0], 7.0, "polyval2d");
    let v3 = ferray_polynomial::polyval3d(
        &[1.0],
        &[2.0],
        &[3.0],
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        2,
        2,
        2,
    )
    .unwrap();
    assert_close(v3[0], 7.0, "polyval3d");
    assert_eq!(
        ferray_polynomial::polygrid2d(&[1.0, 2.0], &[3.0, 4.0], &[1.0, 0.0, 0.0, 1.0], 2, 2)
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        ferray_polynomial::polygrid3d(
            &[1.0],
            &[2.0],
            &[3.0],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            2,
            2,
            2,
        )
        .unwrap()
        .len(),
        1
    );
    assert_eq!(
        ferray_polynomial::polyvander2d(&[1.0, 2.0], &[3.0, 4.0], 1, 1)
            .unwrap()
            .len(),
        8
    );
    assert_eq!(
        ferray_polynomial::polyvander3d(&[1.0], &[2.0], &[3.0], 1, 1, 1)
            .unwrap()
            .len(),
        8
    );

    assert_eq!(ferray_polynomial::chebpts1(5).unwrap().len(), 5);
    assert_eq!(ferray_polynomial::chebpts2(5).unwrap().len(), 5);
    assert_close(ferray_polynomial::chebweight(0.0), 1.0, "chebweight");
    assert_close(
        ferray_polynomial::chebinterpolate(|x| 1.0 + x + x * x, 2)
            .unwrap()
            .eval(0.5)
            .unwrap(),
        1.75,
        "chebinterpolate",
    );

    let (cheb_nodes, cheb_weights) = ferray_polynomial::chebgauss(4).unwrap();
    assert_eq!(cheb_nodes.len(), 4);
    assert_eq!(cheb_weights.len(), 4);
    let (_leg_nodes, leg_weights) = ferray_polynomial::leggauss(4).unwrap();
    assert_close(leg_weights.iter().sum::<f64>(), 2.0, "leggauss weights");
    let (_herm_nodes, herm_weights) = ferray_polynomial::hermgauss(4).unwrap();
    assert_close(
        herm_weights.iter().sum::<f64>(),
        std::f64::consts::PI.sqrt(),
        "hermgauss weights",
    );
    let (_herme_nodes, herme_weights) = ferray_polynomial::hermegauss(4).unwrap();
    assert_close(
        herme_weights.iter().sum::<f64>(),
        (2.0 * std::f64::consts::PI).sqrt(),
        "hermegauss weights",
    );
    let (_lag_nodes, lag_weights) = ferray_polynomial::laggauss(4).unwrap();
    assert_close(lag_weights.iter().sum::<f64>(), 1.0, "laggauss weights");
    let (lagguass_nodes, lagguass_weights) = ferray_polynomial::lagguass(4).unwrap();
    assert_eq!(lagguass_nodes.len(), 4);
    assert_eq!(lagguass_weights.len(), 4);

    let coeffs = [1.0, 2.0, 3.0];
    assert_vec_close(
        &ferray_polynomial::cheb2poly(&ferray_polynomial::poly2cheb(&coeffs).unwrap()).unwrap(),
        &coeffs,
        "root poly/cheb round trip",
    );
    assert_vec_close(
        &ferray_polynomial::leg2poly(&ferray_polynomial::poly2leg(&coeffs).unwrap()).unwrap(),
        &coeffs,
        "root poly/leg round trip",
    );
    assert_vec_close(
        &ferray_polynomial::lag2poly(&ferray_polynomial::poly2lag(&coeffs).unwrap()).unwrap(),
        &coeffs,
        "root poly/lag round trip",
    );
    assert_vec_close(
        &ferray_polynomial::herm2poly(&ferray_polynomial::poly2herm(&coeffs).unwrap()).unwrap(),
        &coeffs,
        "root poly/herm round trip",
    );
    assert_vec_close(
        &ferray_polynomial::herme2poly(&ferray_polynomial::poly2herme(&coeffs).unwrap()).unwrap(),
        &coeffs,
        "root poly/herme round trip",
    );
}

#[test]
fn canonical_inherent_polynomial_methods_are_directly_exercised() {
    let xs = [0.0, 1.0, 2.0, 3.0];
    let ys = [1.0, 3.0, 7.0, 13.0];

    let c = ferray_polynomial::chebyshev::Chebyshev::new(&[0.0, 1.0]);
    assert_close(c.eval(0.25).unwrap(), 0.25, "canonical Chebyshev::new");
    let c = ferray_polynomial::chebyshev::Chebyshev::with_domain(c, [0.0, 1.0]).unwrap();
    let _ = ferray_polynomial::chebyshev::Chebyshev::with_window(c, [-1.0, 1.0]).unwrap();
    let _ = ferray_polynomial::chebyshev::Chebyshev::fit_with_domain(&xs, &ys, 2).unwrap();

    let h = ferray_polynomial::hermite::Hermite::new(&[0.0, 0.5]);
    assert_close(h.eval(0.25).unwrap(), 0.25, "canonical Hermite::new");
    let h = ferray_polynomial::hermite::Hermite::with_domain(h, [0.0, 1.0]).unwrap();
    let _ = ferray_polynomial::hermite::Hermite::with_window(h, [-1.0, 1.0]).unwrap();
    let _ = ferray_polynomial::hermite::Hermite::fit_with_domain(&xs, &ys, 2).unwrap();

    let he = ferray_polynomial::hermite_e::HermiteE::new(&[0.0, 1.0]);
    assert_close(he.eval(0.25).unwrap(), 0.25, "canonical HermiteE::new");
    let he = ferray_polynomial::hermite_e::HermiteE::with_domain(he, [0.0, 1.0]).unwrap();
    let _ = ferray_polynomial::hermite_e::HermiteE::with_window(he, [-1.0, 1.0]).unwrap();
    let _ = ferray_polynomial::hermite_e::HermiteE::fit_with_domain(&xs, &ys, 2).unwrap();

    let la = ferray_polynomial::laguerre::Laguerre::new(&[0.0, 1.0]);
    assert_close(la.eval(0.25).unwrap(), 0.75, "canonical Laguerre::new");
    let la = ferray_polynomial::laguerre::Laguerre::with_domain(la, [0.0, 1.0]).unwrap();
    let _ = ferray_polynomial::laguerre::Laguerre::with_window(la, [0.0, 1.0]).unwrap();
    let _ = ferray_polynomial::laguerre::Laguerre::fit_with_domain(&xs, &ys, 2).unwrap();

    let l = ferray_polynomial::legendre::Legendre::new(&[0.0, 1.0]);
    assert_close(l.eval(0.25).unwrap(), 0.25, "canonical Legendre::new");
    let l = ferray_polynomial::legendre::Legendre::with_domain(l, [0.0, 1.0]).unwrap();
    let _ = ferray_polynomial::legendre::Legendre::with_window(l, [-1.0, 1.0]).unwrap();
    let _ = ferray_polynomial::legendre::Legendre::fit_with_domain(&xs, &ys, 2).unwrap();

    let p = ferray_polynomial::power::Polynomial::basis(2, None, None).unwrap();
    assert_close(p.eval(3.0).unwrap(), 9.0, "Polynomial::basis");
    let q = ferray_polynomial::power::Polynomial::new(&[1.0, 1.0]);
    let composed = ferray_polynomial::power::Polynomial::compose(&p, &q).unwrap();
    assert_vec_close(composed.coeffs(), &[1.0, 2.0, 1.0], "Polynomial::compose");
    let _ = ferray_polynomial::power::Polynomial::fit_with_domain(&xs, &ys, 2).unwrap();
    assert_vec_close(
        ferray_polynomial::power::Polynomial::from_roots(&[1.0, 2.0]).coeffs(),
        &[2.0, -3.0, 1.0],
        "Polynomial::from_roots",
    );
    assert_vec_close(
        ferray_polynomial::power::Polynomial::identity(None, None)
            .unwrap()
            .coeffs(),
        &[0.0, 1.0],
        "Polynomial::identity",
    );
    let anchored =
        ferray_polynomial::power::Polynomial::integ_with_bounds(&q, 1, &[0.0], 2.0, 1.0).unwrap();
    assert_close(
        anchored.eval(2.0).unwrap(),
        0.0,
        "Polynomial::integ_with_bounds",
    );
    let p = ferray_polynomial::power::Polynomial::with_domain(q.clone(), [0.0, 1.0]).unwrap();
    let _ = ferray_polynomial::power::Polynomial::with_window(p, [-1.0, 1.0]).unwrap();
}

#[test]
fn f32_and_complex_canonical_surfaces_match_power_basis_contracts() {
    let p32 = ferray_polynomial::power_f32::PolynomialF32::new(&[1.0_f32, 2.0, 3.0]);
    assert_eq!(
        ferray_polynomial::power_f32::PolynomialF32::coeffs(&p32),
        &[1.0_f32, 2.0, 3.0]
    );
    assert_eq!(ferray_polynomial::power_f32::PolynomialF32::degree(&p32), 2);
    assert!((ferray_polynomial::power_f32::PolynomialF32::eval(&p32, 2.0) - 17.0).abs() < 1e-5);
    let _ =
        ferray_polynomial::power_f32::PolynomialF32::with_domain(p32.clone(), [0.0, 1.0]).unwrap();
    let _ =
        ferray_polynomial::power_f32::PolynomialF32::with_window(p32.clone(), [-1.0, 1.0]).unwrap();
    assert_eq!(
        ferray_polynomial::power_f32::PolynomialF32::deriv(&p32, 1).coeffs(),
        &[2.0_f32, 6.0]
    );
    assert_eq!(
        ferray_polynomial::power_f32::PolynomialF32::integ(&p32, 1, &[0.0]).coeffs(),
        &[0.0_f32, 1.0, 1.0, 1.0]
    );
    let _ = ferray_polynomial::power_f32::PolynomialF32::trim(&p32, 1e-6).unwrap();
    assert_eq!(
        ferray_polynomial::power_f32::PolynomialF32::truncate(&p32, 2)
            .unwrap()
            .coeffs(),
        &[1.0_f32, 2.0]
    );
    assert_eq!(
        ferray_polynomial::power_f32::PolynomialF32::add(&p32, &p32)
            .unwrap()
            .coeffs(),
        &[2.0_f32, 4.0, 6.0]
    );
    assert_eq!(
        ferray_polynomial::power_f32::PolynomialF32::sub(&p32, &p32)
            .unwrap()
            .coeffs(),
        &[0.0_f32, 0.0, 0.0]
    );
    assert_eq!(
        ferray_polynomial::power_f32::PolynomialF32::mul(
            &p32,
            &ferray_polynomial::power_f32::PolynomialF32::new(&[1.0_f32]),
        )
        .unwrap()
        .coeffs(),
        p32.coeffs()
    );
    let fit32 = ferray_polynomial::power_f32::PolynomialF32::fit(
        &[0.0_f32, 1.0, 2.0, 3.0],
        &[1.0_f32, 3.0, 7.0, 13.0],
        2,
    )
    .unwrap();
    assert!((fit32.eval(2.0) - 7.0).abs() < 1e-4);
    assert!(
        !ferray_polynomial::power_f32::PolynomialF32::roots(
            &ferray_polynomial::power_f32::PolynomialF32::new(&[2.0_f32, -3.0, 1.0],)
        )
        .unwrap()
        .is_empty()
    );
    let p64 = ferray_polynomial::power_f32::PolynomialF32::to_f64(&p32);
    let p32_again = ferray_polynomial::power_f32::PolynomialF32::from_f64(&p64).unwrap();
    assert_eq!(p32_again.coeffs(), p32.coeffs());

    let cp = ferray_polynomial::power_complex::ComplexPolynomial::new(&[
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
    ]);
    assert_eq!(
        ferray_polynomial::power_complex::ComplexPolynomial::coeffs(&cp).len(),
        2
    );
    assert_eq!(
        ferray_polynomial::power_complex::ComplexPolynomial::degree(&cp),
        1
    );
    assert_eq!(
        ferray_polynomial::power_complex::ComplexPolynomial::eval(&cp, 2.0),
        Complex::new(1.0, 2.0)
    );
    assert_eq!(
        ferray_polynomial::power_complex::ComplexPolynomial::eval_complex(
            &cp,
            Complex::new(2.0, 0.0),
        ),
        Complex::new(1.0, 2.0)
    );
    assert_eq!(
        ferray_polynomial::power_complex::ComplexPolynomial::deriv(&cp, 1).coeffs(),
        &[Complex::new(0.0, 1.0)]
    );
    assert_eq!(
        ferray_polynomial::power_complex::ComplexPolynomial::integ(
            &cp,
            1,
            &[Complex::new(0.0, 0.0)],
        )
        .coeffs()
        .len(),
        3
    );
    assert_eq!(
        ferray_polynomial::power_complex::ComplexPolynomial::add(&cp, &cp).coeffs()[0],
        Complex::new(2.0, 0.0)
    );
    assert_eq!(
        ferray_polynomial::power_complex::ComplexPolynomial::sub(&cp, &cp).coeffs()[0],
        Complex::new(0.0, 0.0)
    );
    assert_eq!(
        ferray_polynomial::power_complex::ComplexPolynomial::mul(
            &cp,
            &ferray_polynomial::power_complex::ComplexPolynomial::new(&[Complex::new(1.0, 0.0)]),
        )
        .coeffs(),
        cp.coeffs()
    );
    let from_roots = ferray_polynomial::power_complex::ComplexPolynomial::from_roots(&[
        Complex::new(1.0, 0.0),
        Complex::new(-1.0, 0.0),
    ]);
    assert_close(
        from_roots.coeffs()[0].re,
        -1.0,
        "ComplexPolynomial::from_roots",
    );
    let _ = ferray_polynomial::power_complex::ComplexPolynomial::trim(&cp, 1e-12).unwrap();
    assert_eq!(
        ferray_polynomial::power_complex::ComplexPolynomial::truncate(&cp, 1)
            .unwrap()
            .coeffs()
            .len(),
        1
    );
}
