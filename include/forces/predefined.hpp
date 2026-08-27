/**
 * @file predefined.hpp
 * @brief Built-in force laws for center-based multicellular modeling.
 *
 * Forces follow the recipes from:
 *   "Recipes for Centre-Based Modelling of Multicellular Systems"
 *
 * @par Convention
 * The force magnitude \f$F = -dU/dd\f$ follows:
 * - \f$F > 0\f$ → repulsive  (particles pushed apart: \f$d < r_0\f$)
 * - \f$F < 0\f$ → attractive (particles pulled together: \f$d > r_0\f$)
 *
 * The force vector is then `displacement * F / distance`, where
 * `displacement = position_j - position_i` (points from agent \a i to agent \a j).
 *
 * Each `Scalar XxxPotential(...)` returns the force \e magnitude \f$F\f$
 * (not the potential energy). Each `Vector Xxx(...)` returns the full force vector.
 *
 * @par Summary of forces from the paper
 *
 * | Force               | Potential \f$U(d)\f$                                                                 | Force \f$F(d) = -dU/dd\f$                                  |
 * |---------------------|---------------------------------------------------------------------------------------|------------------------------------------------------------|
 * | Linear Spring       | \f$\frac{\mu}{2} (d - s_{ij})^2\f$                                                   | \f$\mu (s_{ij} - d)\f$                                     |
 * | Cubic               | \f$\frac{\mu}{3} (d - s_{ij})^3\f$ for \f$d < r_c\f$                                 | \f$\mu (s_{ij} - d)^2\f$ for \f$d < r_c\f$                 |
 * | Morse               | \f$u_0 e^{-2\rho(d - r_{\rm eq})} - 2u_0 e^{-\rho(d - r_{\rm eq})}\f$                | \f$2u_0\rho(e^{-2\rho(d - r_{\rm eq})} - e^{-\rho(d - r_{\rm eq})})\f$ |
 * | Hertz               | (pure contact, no closed-form potential)                                              | \f$\frac{4}{3}E^*\sqrt{R^*}(R_i+R_j-d)^{3/2}\f$ for \f$d < R_i+R_j\f$ |
 * | Lennard-Jones (LJ)  | \f$4\varepsilon\bigl[(\sigma/d)^{12} - (\sigma/d)^6\bigr]\f$                          | \f$\frac{24\varepsilon}{d}\bigl[2(\sigma/d)^{12} - (\sigma/d)^6\bigr]\f$ |
 *
 * @note Additional forces (piecewise-linear, piecewise-quadratic) are provided
 *       as convenience utilities that are commonly used but not explicitly in the paper.
 */

#ifndef TECS_FORCES_PREDEFINED_HPP
#define TECS_FORCES_PREDEFINED_HPP

#include <Kokkos_Core.hpp>

#include "detail.hpp"

namespace tecs::forces {

  // ===========================================================================
  //  1. Linear Spring (Harmonic)
  // ===========================================================================

  /**
   * @brief Linear spring force magnitude.
   *
   * \f$U = \frac{\mu}{2} (d - s_{ij})^2\f$,
   * \f$F = -dU/dd = \mu (s_{ij} - d)\f$.
   *
   * Repulsive for \f$d < s_{ij}\f$, attractive for \f$d > s_{ij}\f$.
   *
   * @tparam Scalar floating-point type (float / double).
   * @param distance            separation distance between the two agents.
   * @param homeostatic_radius  rest length \f$s_{ij}\f$ where force is zero.
   * @param scale               spring constant \f$\mu\f$.
   * @return Force magnitude \f$F = \mu (s_{ij} - d)\f$.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar SpringPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar scale = Scalar(1)
  ) {
    return scale * (homeostatic_radius - distance);
  }

  /**
   * @brief Linear spring force vector.
   * @copydetails SpringPotential
   * @param displacement vector from agent \a i to agent \a j.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector Spring(
    const Vector& displacement,
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar scale = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = SpringPotential(distance, homeostatic_radius, scale);
    return displacement * F / distance;
  }

  // ===========================================================================
  //  6. Piecewise-Linear (convenience; common in center-based models)
  // ===========================================================================

  /**
   * @brief Piecewise-linear force magnitude.
   *
   * A simple combination of linear repulsion (below `repulsion_radius`) and
   * linear adhesion (above `adhesion_radius`), with an inactive zone between.
   *
   * @par Regions
   * | Distance             | Force                                         | Sign       |
   * |----------------------|-----------------------------------------------|------------|
   * | \f$d < r_{\rm rep}\f$  | \f$s_{\rm rep}(r_{\rm rep} - d)\f$          | repulsive  |
   * | \f$d > r_{\rm adh}\f$  | \f$-s_{\rm adh}(d - r_{\rm adh})\f$         | attractive |
   * | otherwise            | 0                                             | neutral    |
   *
   * Potentials: \f$U_{\rm rep} = \frac{1}{2}s_{\rm rep}\max(r_{\rm rep}-d,0)^2\f$,
   * \f$U_{\rm adh} = \frac{1}{2}s_{\rm adh}\max(d-r_{\rm adh},0)^2\f$.
   *
   * @tparam Scalar floating-point type.
   * @param distance         separation distance.
   * @param repulsion_radius distance below which repulsion activates.
   * @param adhesion_radius  distance above which adhesion activates.
   * @param repulsion_scale  repulsion stiffness \f$s_{\rm rep}\f$.
   * @param adhesion_scale   adhesion stiffness \f$s_{\rm adh}\f$.
   * @return Force magnitude \f$F = -dU/dd\f$.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar PiecewiseLinearPotential(
    const Scalar distance,
    const Scalar repulsion_radius,
    const Scalar adhesion_radius,
    const Scalar repulsion_scale = Scalar(1),
    const Scalar adhesion_scale = Scalar(1)
  ) {
    return repulsion_scale * Kokkos::fmax(repulsion_radius - distance, Scalar(0))
         - adhesion_scale  * Kokkos::fmax(distance - adhesion_radius, Scalar(0));
  }

  /**
   * @brief Piecewise-linear force vector.
   * @copydetails PiecewiseLinearPotential
   * @param displacement vector from agent \a i to agent \a j.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector PiecewiseLinear(
    const Vector& displacement,
    const Scalar distance,
    const Scalar repulsion_radius,
    const Scalar adhesion_radius,
    const Scalar repulsion_scale = Scalar(1),
    const Scalar adhesion_scale = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = PiecewiseLinearPotential(
      distance,
      repulsion_radius,
      adhesion_radius,
      repulsion_scale,
      adhesion_scale
    );
    return displacement * F / distance;
  }

  // ===========================================================================
  //  7. Piecewise-Quadratic (convenience; used in epithelia/tissue models)
  // ===========================================================================

  /**
   * @brief Piecewise-quadratic force magnitude.
   *
   * Uses quadratic potentials so the force is piecewise-linear:
   *
   * @par Repulsion (\f$d < r_0\f$)
   * \f$U_{\rm rep} = s_{\rm rep} (d - r_{\rm rep})^2\f$,
   * \f$F_{\rm rep} = 2 s_{\rm rep} (r_{\rm rep} - d)\f$,
   * active only for \f$d < r_0\f$ and \f$d < r_{\rm rep}\f$.
   *
   * @par Adhesion (\f$d > r_0\f$)
   * \f$U_{\rm adh} = -s_{\rm adh} (d - r_{\rm adh})^2\f$,
   * \f$F_{\rm adh} = 2 s_{\rm adh} (d - r_{\rm adh})\f$
   * (negative when \f$d < r_{\rm adh}\f$),
   * active for \f$r_0 < d < r_{\rm adh}\f$.
   *
   * This is commonly used in vertex-like cell-center models where the force
   * transitions linearly across the homeostatic radius.
   *
   * @tparam Scalar floating-point type.
   * @param distance           separation distance.
   * @param homeostatic_radius equilibrium \f$r_0\f$ where \f$F = 0\f$.
   * @param repulsion_radius   distance at which repulsion begins \f$r_{\rm rep}\f$.
   * @param adhesion_radius    distance up to which adhesion acts \f$r_{\rm adh}\f$.
   * @param repulsion_scale    repulsion stiffness \f$s_{\rm rep}\f$.
   * @param adhesion_scale     adhesion stiffness \f$s_{\rm adh}\f$.
   * @return Force magnitude \f$F = -dU/dd\f$.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar PiecewiseQuadraticPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar repulsion_radius,
    const Scalar adhesion_radius,
    const Scalar repulsion_scale = Scalar(1),
    const Scalar adhesion_scale = Scalar(1)
  ) {
    // Repulsion: U_rep = s_rep * (d - r_rep)²
    // F_rep = -dU/dd = 2 * s_rep * (r_rep - d)  for d < r_0 and d < r_rep
    Scalar repulsion = Scalar(0);
    if (distance < homeostatic_radius) {
      repulsion = Scalar(2) * repulsion_scale * Kokkos::fmax(repulsion_radius - distance, Scalar(0));
    }

    // Adhesion: U_adh = -s_adh * (d - r_adh)²
    // F_adh = -dU/dd = 2 * s_adh * (d - r_adh)  for r_0 < d < r_adh
    // (negative when d < r_adh)
    Scalar adhesion = Scalar(0);
    if (distance > homeostatic_radius && distance < adhesion_radius) {
      adhesion = Scalar(2) * adhesion_scale * (distance - adhesion_radius);
    }

    return repulsion + adhesion;
  }

  /**
   * @brief Piecewise-quadratic force vector.
   * @copydetails PiecewiseQuadraticPotential
   * @param displacement vector from agent \a i to agent \a j.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector PiecewiseQuadratic(
    const Vector& displacement,
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar repulsion_radius,
    const Scalar adhesion_radius,
    const Scalar repulsion_scale = Scalar(1),
    const Scalar adhesion_scale = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = PiecewiseQuadraticPotential(
      distance,
      homeostatic_radius,
      repulsion_radius,
      adhesion_radius,
      repulsion_scale,
      adhesion_scale
    );
    return displacement * F / distance;
  }

  // ===========================================================================
  //  2. Cubic (from the paper: U ∝ (d - s_ij)³)
  // ===========================================================================

  /**
   * @brief Cubic force magnitude.
   *
   * From the paper (Table 1 / eq. 3.12):
   * \f$U = \frac{\mu}{3} (d - s_{ij})^3\f$ for \f$d < r_c\f$, 0 otherwise.
   *
   * \f$F = -dU/dd = \mu (s_{ij} - d)^2\f$ for \f$d < r_c\f$, 0 otherwise.
   *
   * The force is always non-negative (repulsive when \f$d < s_{ij}\f$,
   * zero at \f$d = s_{ij}\f$, attractive and growing when \f$s_{ij} < d < r_c\f$).
   * The sign is determined by the sign of \f$(s_{ij} - d)\f$ via the
   * displacement vector direction; the squared magnitude is always repulsive
   * for \f$d < s_{ij}\f$ and attractive for \f$d > s_{ij}\f$.
   *
   * @note This differs from the piecewise-cubic used in some other codes.
   *       Here adhesion and repulsion both use the same \f$(s_{ij} - d)^2\f$ term.
   *
   * @tparam Scalar floating-point type.
   * @param distance           separation distance.
   * @param homeostatic_radius rest length \f$s_{ij}\f$.
   * @param cutoff_distance    distance \f$r_c\f$ beyond which the force is zero.
   * @param scale              overall strength \f$\mu\f$.
   * @return Force magnitude \f$F = \mu (s_{ij} - d)^2\f$ if \f$d < r_c\f$, else 0.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar CubicPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar cutoff_distance,
    const Scalar scale = Scalar(1)
  ) {
    // Force = -dU/dd = μ (s_ij - d)²  for d < r_c,  0 otherwise
    if (distance >= cutoff_distance)
      return Scalar(0);

    Scalar diff = homeostatic_radius - distance;
    return scale * diff * diff;
  }

  /**
   * @brief Cubic force vector.
   * @copydetails CubicPotential
   * @param displacement vector from agent \a i to agent \a j.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector Cubic(
    const Vector& displacement,
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar cutoff_distance,
    const Scalar scale = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = CubicPotential(
      distance,
      homeostatic_radius,
      cutoff_distance,
      scale
    );
    return displacement * F / distance;
  }

  // ===========================================================================
  //  3. Morse
  // ===========================================================================

  /**
   * @brief Morse force magnitude.
   *
   * From the paper (eq. 2.12 / Table 1):
   * \f$U = u_0 e^{-2\rho(d - r_{\rm eq})} - 2u_0 e^{-\rho(d - r_{\rm eq})}\f$
   *
   * \f$F = -dU/dd = 2u_0\rho\bigl[e^{-2\rho(d - r_{\rm eq})} - e^{-\rho(d - r_{\rm eq})}\bigr]\f$
   *
   * The potential has a minimum (\f$F = 0\f$) at \f$d = r_{\rm eq}\f$ with
   * depth \f$-u_0\f$. The force is repulsive for \f$d < r_{\rm eq}\f$ and
   * attractive for \f$d > r_{\rm eq}\f$.
   *
   * @tparam Scalar floating-point type.
   * @param distance   separation distance.
   * @param equilibrium_distance  equilibrium distance \f$r_{\rm eq}\f$ where force vanishes.
   * @param steepness  steepness parameter \f$\rho\f$ (controls the width of the well).
   * @param well_depth well depth \f$u_0\f$ (depth of the attractive minimum).
   * @return Force magnitude \f$F = -dU/dd\f$.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar MorsePotential(
    const Scalar distance,
    const Scalar equilibrium_distance,
    const Scalar steepness = Scalar(1),
    const Scalar well_depth = Scalar(1)
  ) {
    // F = -dU/dd = 2 u_0 ρ [exp(-2ρ(d - r_eq)) - exp(-ρ(d - r_eq))]
    Scalar rho = steepness;
    Scalar u0 = well_depth;
    Scalar dx = distance - equilibrium_distance;

    Scalar rep_term = Kokkos::exp(-Scalar(2) * rho * dx);
    Scalar adh_term = Kokkos::exp(-rho * dx);

    return Scalar(2) * u0 * rho * (rep_term - adh_term);
  }

  /**
   * @brief Morse force vector.
   * @copydetails MorsePotential
   * @param displacement vector from agent \a i to agent \a j.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector Morse(
    const Vector& displacement,
    const Scalar distance,
    const Scalar equilibrium_distance,
    const Scalar steepness = Scalar(1),
    const Scalar well_depth = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = MorsePotential(
      distance,
      equilibrium_distance,
      steepness,
      well_depth
    );
    return displacement * F / distance;
  }

  // ===========================================================================
  //  5. Lennard-Jones (12-6)
  // ===========================================================================

  /**
   * @brief 12-6 Lennard-Jones force magnitude.
   *
   * From the paper (eq. 2.9 / Table 1):
   * \f$U = 4\varepsilon\bigl[(\sigma/d)^{12} - (\sigma/d)^6\bigr]\f$.
   *
   * \f$F = -dU/dd = \frac{24\varepsilon}{d}\bigl[2(\sigma/d)^{12} - (\sigma/d)^6\bigr]\f$.
   *
   * Force is repulsive for \f$d < 2^{1/6}\sigma \approx 1.122\sigma\f$,
   * attractive for \f$d > 2^{1/6}\sigma\f$, with a minimum at
   * \f$d = 2^{1/6}\sigma\f$ of depth \f$-\varepsilon\f$.
   *
   * At \f$d = \sigma\f$ the potential energy is zero (but the force is repulsive).
   *
   * @tparam Scalar floating-point type.
   * @param distance separation distance.
   * @param sigma    length scale \f$\sigma\f$ (distance where \f$U = 0\f$).
   * @param epsilon  well depth \f$\varepsilon\f$ (depth of the attractive minimum).
   * @return Force magnitude \f$F = -dU/dd\f$, or 0 if `distance == 0`.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar LennardJonesPotential(
    const Scalar distance,
    const Scalar sigma,
    const Scalar epsilon = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Scalar(0);

    Scalar inv_d = sigma / distance;
    Scalar inv_d6  = Kokkos::pow(inv_d, Scalar(6));
    Scalar inv_d12 = inv_d6 * inv_d6;

    // F = -dU/dd = (24 ε / d) * [2 (σ/d)¹² - (σ/d)⁶]
    //            = (24 ε / distance) * [2 inv_d12 - inv_d6]
    return (Scalar(24) * epsilon / distance) * (Scalar(2) * inv_d12 - inv_d6);
  }

  /**
   * @brief Lennard-Jones force vector.
   * @copydetails LennardJonesPotential
   * @param displacement vector from agent \a i to agent \a j.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector LennardJones(
    const Vector& displacement,
    const Scalar distance,
    const Scalar sigma,
    const Scalar epsilon = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = LennardJonesPotential(
      distance,
      sigma,
      epsilon
    );
    return displacement * F / distance;
  }

  // ===========================================================================
  //  4. Hertz Contact (repulsion only, elastic spheres)
  // ===========================================================================

  /**
   * @brief Hertzian elastic contact force magnitude (repulsion only).
   *
   * From the paper (eq. 2.16):
   * \f$F = \frac{4}{3} E^* \sqrt{R^*} (R_i + R_j - d)^{3/2}\f$
   * for \f$d < R_i + R_j\f$, 0 otherwise.
   *
   * where the composite elastic modulus and effective radius are:
   * \f$\frac{1}{E^*} = \frac{1 - \nu_i^2}{E_i} + \frac{1 - \nu_j^2}{E_j}\f$,
   * \f$\frac{1}{R^*} = \frac{1}{R_i} + \frac{1}{R_j}\f$.
   *
   * This is a pure repulsive contact force with no adhesion.
   *
   * @tparam Scalar floating-point type.
   * @param distance           separation distance.
   * @param radius_i           radius of agent \a i \f$R_i\f$.
   * @param radius_j           radius of agent \a j \f$R_j\f$.
   * @param young_modulus_i    Young's modulus of agent \a i \f$E_i\f$.
   * @param young_modulus_j    Young's modulus of agent \a j \f$E_j\f$.
   * @param poisson_ratio_i    Poisson's ratio of agent \a i \f$\nu_i\f$.
   * @param poisson_ratio_j    Poisson's ratio of agent \a j \f$\nu_j\f$.
   * @return Force magnitude \f$F\f$, or 0 if `distance >= R_i + R_j`.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar HertzContactPotential(
    const Scalar distance,
    const Scalar radius_i,
    const Scalar radius_j,
    const Scalar young_modulus_i = Scalar(1),
    const Scalar young_modulus_j = Scalar(1),
    const Scalar poisson_ratio_i = Scalar(0),
    const Scalar poisson_ratio_j = Scalar(0)
  ) {
    // Sum of radii is the contact distance
    Scalar contact_distance = radius_i + radius_j;

    // No overlap → no force
    if (distance >= contact_distance)
      return Scalar(0);

    // Composite Young's modulus: 1/E* = (1-ν_i²)/E_i + (1-ν_j²)/E_j
    Scalar ni2 = Kokkos::pow(poisson_ratio_i, Scalar(2));
    Scalar nj2 = Kokkos::pow(poisson_ratio_j, Scalar(2));

    Scalar composite_youngs_modulus = young_modulus_i * young_modulus_j /
      ((Scalar(1) - ni2) * young_modulus_j +
       (Scalar(1) - nj2) * young_modulus_i);

    // Effective radius: 1/R* = 1/R_i + 1/R_j
    Scalar effective_radius = (radius_i * radius_j) / contact_distance;

    // Overlap (indentation depth)
    Scalar overlap = contact_distance - distance;

    // F = (4/3) * E* * sqrt(R*) * overlap^(3/2)
    return (Scalar(4) / Scalar(3)) * composite_youngs_modulus
           * Kokkos::sqrt(effective_radius)
           * Kokkos::pow(overlap, Scalar(3) / Scalar(2));
  }

  /**
   * @brief Hertzian contact force vector.
   * @copydetails HertzContactPotential
   * @param displacement vector from agent \a i to agent \a j.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector HertzContact(
    const Vector& displacement,
    const Scalar distance,
    const Scalar radius_i,
    const Scalar radius_j,
    const Scalar young_modulus_i = Scalar(1),
    const Scalar young_modulus_j = Scalar(1),
    const Scalar poisson_ratio_i = Scalar(0),
    const Scalar poisson_ratio_j = Scalar(0)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = HertzContactPotential(
      distance,
      radius_i,
      radius_j,
      young_modulus_i,
      young_modulus_j,
      poisson_ratio_i,
      poisson_ratio_j
    );
    return displacement * F / distance;
  }
} // namespace tecs::forces

#endif // TECS_FORCES_PREDEFINED_HPP
