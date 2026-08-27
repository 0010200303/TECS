/**
 * @file predefined.hpp
 * @brief Built-in force laws for center-based modeling.
 *
 * Forces transcribed from:
 *   T. Liebisch, J. Sharpe, and F. Matthäus, “Recipes for Centre-Based
 *   Modelling of Multicellular Systems”, Section 2.2, Eqs. (5)–(13).
 *
 * @par Sign convention (paper Eq. (4) and Fig. 1)
 * The pairwise force exerted on agent \a i by agent \a j is
 * \f$
 *   \mathbf{F}_{ij} = F(d_{ij})\,\frac{\mathbf{r}_{ij}}{d_{ij}},
 *   \qquad \mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j,
 * \f$
 * i.e. `displacement * F / distance` with `displacement = position_i - position_j`
 * (a vector pointing from agent \a j to agent \a i). The scalar magnitude obeys
 * - \f$F > 0\f$ -> repulsion  (agents pushed apart, \f$d_{ij} < d_c\f$),
 * - \f$F = 0\f$ -> equilibrium (\f$d_{ij} = d_c\f$),
 * - \f$F < 0\f$ -> adhesion   (agents pulled together, \f$d_c < d_{ij} < d_\sigma\f$),^
 * - \f$F = 0\f$ -> no interaction beyond the cut-off distance \f$d_{ij} \ge d_\sigma\f$.
 *
 * Each `Scalar XxxPotential(...)` returns the force \e magnitude \f$F(d)\f$
 * (not the potential energy). Each `Vector Xxx(...)` returns the full force vector.
 *
 * @par Summary of the force functions from the paper
 *
 * | Force            | \f$F(d)\f$                                                                                                                               |
 * |------------------|------------------------------------------------------------------------------------------------------------------------------------------|
 * | Linear spring    | \f$k\,(d_c - d)\f$                                                                                                                       |
 * | Piecewise-linear | \f$k_{rep}\max(d_c-d_{ij},0) - k_{adh}\max(d_{ij}-d_c,0)\f$                                                                              |
 * | Piecewise-quadr. | \f$k_{rep}(d_{ij}-d_c)(d_{ij}-d_{rep})\f$ for \f$d<d_c\f$; \f$k_{adh}(d_{ij}-d_c)(d_{ij}-d_\sigma)\f$ for \f$d_c\le d_{ij} < d_\sigma\f$ |
 * | Cubic            | \f$\mu(d_\sigma - d)^2(d_c - d_{ij})\f$ for \f$d_{ij} < d_\sigma\f$                                                                      |
 * | Morse            | \f$2F_0\left(e^{-2a(d_{ij}-d_c)} - e^{-a(d_{ij}-d_c)}\right)\f$                                                                          |
 * | Lennard-Jones    | \f$\frac{24\varepsilon}{d_{ij}}\left[2(\sigma/d_{ij})^{12} - (d_c/d_{ij})^6\right]\f$                                                    |
 * | Hertz            | \f$\frac{4}{3}E^*\sqrt{R^*}\,\delta^{3/2}\f$, \f$\delta = \max(d_c - d_{ij}, 0)\f$                                                       |
 */

#ifndef TECS_FORCES_PREDEFINED_HPP
#define TECS_FORCES_PREDEFINED_HPP

#include <Kokkos_Core.hpp>

#include "detail.hpp"

namespace tecs::forces {
  /**
   * @brief Linear spring force magnitude.
   *
   * Paper Eq. (5):
   * \f$F(d_{ij}) = k \cdot (d_c - d_{ij})\f$,
   *
   * Repulsive for \f$d_c < d_{ij}\f$, attractive for \f$d_c > d_{ij}\f$.
   *
   * @tparam Scalar Floating-point type (e.g., \c double).
   * @param distance            Distance between the two agents.
   * @param homeostatic_radius  Rest length \f$d_c\f$ where force is zero.
   * @param scale               Spring constant \f$\mu\f$.
   * @return Force magnitude.
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
   * @param displacement Vector from agent \a j to agent \a i.
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

  /**
   * @brief Piecewise-linear force magnitude.
   *
   * Paper Eq. (6):
   * \f$F(d_{ij}) = k_{rep} \cdot \max(d_c - d_{ij}, 0) - k_{adh} \cdot \max(d_{ij} - d_c, 0)\f$
   *
   * Repulsive for \f$d < d_c\f$, attractive for \f$d > d_c\f$, and zero at the
   * equilibrium distance \f$d_c\f$.
   *
   * @tparam Scalar Floating-point type (e.g., \c double).
   * @param distance           Distance between the two agents.
   * @param homeostatic_radius Equilibrium distance \f$d_c\f$ where the force vanishes.
   * @param repulsion_scale    Repulsion stiffness \f$k_{rep}\f$.
   * @param adhesion_scale     Adhesion stiffness \f$k_{adh}\f$.
   * @return Force magnitude.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar PiecewiseLinearPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar repulsion_scale = Scalar(1),
    const Scalar adhesion_scale = Scalar(1)
  ) {
    return repulsion_scale * Kokkos::fmax(homeostatic_radius - distance, Scalar(0))
         - adhesion_scale  * Kokkos::fmax(distance - homeostatic_radius, Scalar(0));
  }

  /**
   * @brief Piecewise-linear force vector.
   * @copydetails PiecewiseLinearPotential
   * @param displacement Vector from agent \a j to agent \a i.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector PiecewiseLinear(
    const Vector& displacement,
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar repulsion_scale = Scalar(1),
    const Scalar adhesion_scale = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = PiecewiseLinearPotential(
      distance,
      homeostatic_radius,
      repulsion_scale,
      adhesion_scale
    );
    return displacement * F / distance;
  }

  /**
   * @brief Piecewise-quadratic force magnitude (quadratic in \f$d\f$).
   *
   * Paper Eq. (7):
   * \f$
   * F(d) = \begin{cases}
   *   k_{rep}(d_{ij} - d_c)(d_{ij} - d_{rep}) & d_{ij} < d_c, \\
   *   k_{adh}(d_{ij} - d_c)(d_{ij} - d_\sigma)    & d_c \le d_{ij} < d_\sigma, \\
   *   0                                           & d_{ij} \ge d_\sigma,
   * \end{cases}
   * \f$
   *
   * @tparam Scalar Floating-point type (e.g., \c double).
   * @param distance           Distance between the two agents.
   * @param homeostatic_radius Equilibrium distance \f$d_c\f$ where the force vanishes.
   * @param repulsion_radius   Reference distance \f$d_{rep}\f$ shaping the repulsive piece (\f$> d_c\f$).
   * @param cutoff_distance    Cut-off distance \f$d_\sigma\f$ where the force vanishes.
   * @param repulsion_scale    Repulsion stiffness \f$k_{rep}\f$.
   * @param adhesion_scale     Adhesion stiffness \f$k_{adh}\f$.
   * @return Force magnitude.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar PiecewiseQuadraticPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar repulsion_radius,
    const Scalar cutoff_distance,
    const Scalar repulsion_scale = Scalar(1),
    const Scalar adhesion_scale = Scalar(1)
  ) {
    if (distance < homeostatic_radius) {
      // Repulsion: k_rep * (d_ij - d_c) * (d_ij - d_rep), positive for d_ij < d_c.
      return repulsion_scale * (distance - homeostatic_radius) * (distance - repulsion_radius);
    }
    if (distance < cutoff_distance) {
      // Adhesion: k_adh (d_ij - d_c) (d_ij - d_sigma), negative for d_c < d_ij < d_sigma.
      return adhesion_scale * (distance - homeostatic_radius) * (distance - cutoff_distance);
    }
    return Scalar(0);
  }

  /**
   * @brief Piecewise-quadratic force vector.
   * @copydetails PiecewiseQuadraticPotential
   * @param displacement Vector from agent \a j to agent \a i.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector PiecewiseQuadratic(
    const Vector& displacement,
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar repulsion_radius,
    const Scalar cutoff_distance,
    const Scalar repulsion_scale = Scalar(1),
    const Scalar adhesion_scale = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = PiecewiseQuadraticPotential(
      distance,
      homeostatic_radius,
      repulsion_radius,
      cutoff_distance,
      repulsion_scale,
      adhesion_scale
    );
    return displacement * F / distance;
  }

  /**
   * @brief Cubic force magnitude.
   *
   * Paper Eq. (9):
   * \f$F(d_{ij}) = \mu (d_\sigma - d_{ij})^2 \cdot (d_c - d_{ij})\f$
   *
   * Repulsive for \f$d_{ij} < d_c\f$, attractive for \f$d_c < d_{ij} < d_\sigma\f$, and
   * vanishing at both \f$d_c\f$ and \f$d_\sigma\f$.
   *
   * @tparam Scalar Floating-point type (e.g., \c double).
   * @param distance           Distance between the two agents.
   * @param homeostatic_radius Equilibrium distance \f$d_c\f$.
   * @param cutoff_distance    Cut-off distance \f$d_\sigma\f$ (must exceed \f$d_c\f$).
   * @param scale              Overall strength \f$\mu\f$.
   * @return Force magnitude \f$F(d_{ij})\f$.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar CubicPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar cutoff_distance,
    const Scalar scale = Scalar(1)
  ) {
    if (distance >= cutoff_distance)
      return Scalar(0);

    // F = mu * (d_sigma - d_ij)^2 * (d_c - d_ij)
    Scalar overlap = cutoff_distance - distance;
    Scalar diff = homeostatic_radius - distance;
    return scale * overlap * overlap * diff;
  }

  /**
   * @brief Cubic force vector.
   * @copydetails CubicPotential
   * @param displacement Vector from agent \a j to agent \a i.
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

  /**
   * @brief Morse force magnitude.
   *
   * Paper Eq. (10):
   * \f$F(d_{ij}) = 2 F_0 \left(e^{-2a(d_{ij} - d_c)} - e^{-a(d_{ij} - d_c)}\right)\f$
   *
   * The first exponential is the repulsive part and the second the adhesive
   * part. \f$F_0\f$ is a scalar scaling factor and \f$a\f$ the slope (stiffness
   * of the cell–cell connection). The force vanishes at \f$d_{ij} = d_c\f$, is
   * repulsive for \f$d_{ij} < d_c\f$ and attractive for \f$d_{ij} > d_c\f$.
   *
   * @tparam Scalar Floating-point type (e.g., \c double).
   * @param distance             Distance between the two agents.
   * @param equilibrium_distance Equilibrium distance \f$d_c\f$ where the force vanishes.
   * @param slope                Slope parameter \f$a\f$ (cell–cell connection stiffness).
   * @param scaling_factor       Scalar scaling factor \f$F_0\f$.
   * @return Force magnitude \f$F(d)\f$.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar MorsePotential(
    const Scalar distance,
    const Scalar equilibrium_distance,
    const Scalar slope = Scalar(1),
    const Scalar scaling_factor = Scalar(1)
  ) {
    Scalar dx = distance - equilibrium_distance;

    Scalar rep_term = Kokkos::exp(-Scalar(2) * slope * dx);
    Scalar adh_term = Kokkos::exp(-slope * dx);

    return Scalar(2) * scaling_factor * (rep_term - adh_term);
  }

  /**
   * @brief Morse force vector.
   * @copydetails MorsePotential
   * @param displacement Vector from agent \a j to agent \a i.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector Morse(
    const Vector& displacement,
    const Scalar distance,
    const Scalar equilibrium_distance,
    const Scalar slope = Scalar(1),
    const Scalar scaling_factor = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = MorsePotential(
      distance,
      equilibrium_distance,
      slope,
      scaling_factor
    );
    return displacement * F / distance;
  }

  /**
   * @brief Lennard-Jones force magnitude.
   *
   * Paper Eq. (8):
   * \f$U(d_{ij}) = 4\epsilon\left[\left(\frac{\sigma}{d_{ij}}\right)^{12} - \left(\frac{\sigma}{d_{ij}}\right)^6\right]\f$
   * from which the force is derived as
   * \f$F(d_{ij}) = -\frac{dU(d_{ij})}{d(d_{ij})} = \frac{24\varepsilon}{d_{ij}}\left[2\left(\frac{\sigma}{d_{ij}}\right)^{12} - \left(\frac{\sigma}{d_{ij}}\right)^6\right]\f$
   *
   * The potential crosses zero at \f$d_{ij} = \sigma\f$ and its well minimum lies at
   * \f$d_{ij} = 2^{1/6}\sigma\f$; the force is repulsive for \f$d_{ij} < \sigma\f$
   * and attractive for \f$d_{ij} > \sigma\f$.
   *
   * @tparam Scalar Floating-point type (e.g., \c double).
   * @param distance Distance between the two agents.
   * @param sigma    Length scale \f$\sigma\f$ (distance where \f$U = 0\f$).
   * @param epsilon  Well depth \f$\varepsilon\f$.
   * @return Force magnitude \f$F(d)\f$, or 0 if `distance == 0`.
   */
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar LennardJonesPotential(
    const Scalar distance,
    const Scalar sigma,
    const Scalar equilibrium_distance,
    const Scalar epsilon = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Scalar(0);

    Scalar s_over_d = sigma / distance;
    Scalar s6  = Kokkos::pow(s_over_d, Scalar(6));
    Scalar s12 = s6 * s6;

    Scalar c_over_d = equilibrium_distance / distance;
    Scalar e6 = Kokkos::pow(c_over_d, Scalar(6));

    return (Scalar(24) * epsilon / distance) * (Scalar(2) * s12 - e6);
  }

  /**
   * @brief Lennard-Jones force vector.
   * @copydetails LennardJonesPotential
   * @param displacement Vector from agent \a j to agent \a i.
   * @return Force vector `displacement * F / distance` (zero vector when `distance == 0`).
   */
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector LennardJones(
    const Vector& displacement,
    const Scalar distance,
    const Scalar sigma,
    const Scalar equilibrium_distance,
    const Scalar epsilon = Scalar(1)
  ) {
    if (distance == Scalar(0))
      return Vector{Scalar(0)};

    Scalar F = LennardJonesPotential(distance, sigma, equilibrium_distance, epsilon);
    return displacement * F / distance;
  }

  /**
   * @brief Hertz contact force magnitude (repulsion only).
   *
   * Paper Eqs. (11)–(13):
   * \f$F(d_{ij}) = \frac{4}{3} E^* \sqrt{R^*} \delta_{ij}^{3/2}\f$,
   * with overlap \f$\delta_{ij} = \max(d_c - d_{ij}, 0)\f$.
   *
   * where the composite elastic modulus and effective radius are:
   * \f$\frac{1}{E^*} = \frac{1 - \nu_i^2}{E_i} + \frac{1 - \nu_j^2}{E_j}\f$,
   * \f$\frac{1}{R^*} = \frac{1}{R_i} + \frac{1}{R_j}\f$.
   *
   * This is a pure repulsive contact force with no adhesion.
   *
   * @tparam Scalar Floating-point type (e.g., \c double).
   * @param distance           Distance between the two agents.
   * @param radius_i           Radius of agent \a i \f$R_i\f$.
   * @param radius_j           Radius of agent \a j \f$R_j\f$.
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

    // No overlap means no force
    if (distance >= contact_distance)
      return Scalar(0);

    // Composite Young's modulus: 1/E* = (1 - v_i^2)/E_i + (1 - v_j^2)/E_j
    Scalar ni2 = Kokkos::pow(poisson_ratio_i, Scalar(2));
    Scalar nj2 = Kokkos::pow(poisson_ratio_j, Scalar(2));

    Scalar composite_youngs_modulus = young_modulus_i * young_modulus_j /
      ((Scalar(1) - ni2) * young_modulus_j + (Scalar(1) - nj2) * young_modulus_i);

    // Effective radius: 1/R^* = 1/R_i + 1/R_j
    Scalar effective_radius = (radius_i * radius_j) / contact_distance;

    Scalar overlap = contact_distance - distance;

    // F = (4/3) * E^* * sqrt(R^*) * overlap^(3/2)
    return (Scalar(4) / Scalar(3)) * composite_youngs_modulus
           * Kokkos::sqrt(effective_radius)
           * Kokkos::pow(overlap, Scalar(3) / Scalar(2));
  }

  /**
   * @brief Hertzian contact force vector.
   * @copydetails HertzContactPotential
   * @param displacement Vector from agent \a j to agent \a i.
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
