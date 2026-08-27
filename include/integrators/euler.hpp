#ifndef TECS_INTEGRATORS_EULER_HPP
#define TECS_INTEGRATORS_EULER_HPP

#include <Kokkos_Core.hpp>

#include "base.hpp"

namespace tecs::integrators {

  /**
   * @brief Forward Euler integrator (first-order explicit).
   *
   * Uses two integration stages:
   * - Stage 0: current simulation state (e.g. positions).
   * - Stage 1: force/velocity increments.
   *
   * The update rule is:
   * @f[
   *   \mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{v}_n \cdot \Delta t
   * @f]
   *
   * @tparam PairFinder          Algorithm for finding pairwise interactions.
   * @tparam ComFixer            Centre-of-mass fixing strategy.
   * @tparam GenericForceFields  Pack of field references for per-agent forces.
   * @tparam PairwiseForceFields Pack of field references for pair forces.
   * @tparam LinkForceFields     Pack of field references for link forces.
   * @tparam Views               View types that hold the simulation data.
   */
  template<
    typename PairFinder,
    typename ComFixer,
    typename GenericForceFields,
    typename PairwiseForceFields,
    typename LinkForceFields,
    typename... Views
  >
  struct Euler : public Base<PairFinder, ComFixer, 2, GenericForceFields, PairwiseForceFields, LinkForceFields, Views...> {
    using Base<PairFinder, ComFixer, 2, GenericForceFields, PairwiseForceFields, LinkForceFields, Views...>::Base;

    /**
     * @brief Apply the Euler step: update positions using the computed deltas.
     * 
     * Optionally corrects for centre-of-mass drift, saves the old
     * velocities, then advances the current state by delta time (dt).
     * The delta buffers are cleared after the update.
     *
     * @param dt Time step size.
     */
    void apply_euler(double dt) {
      const auto com_fix_delta = this->com_fixer.fix(this->stage_pack[1].first());

      Kokkos::parallel_for(
        "apply_euler",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          this->stage_pack[1].first()(i) -= com_fix_delta;

          // write old velocities
          this->old_velocities(i) = this->stage_pack[1].first()(i);

          this->stage_pack[0].zip_apply([&](auto& current, const auto& delta) {
            current(i) += delta(i) * dt;

            // clear delta buffers (faster than new deep_copy call)
            delta(i) = std::remove_cv_t<std::remove_reference_t<decltype(delta(i))>>{};
          }, this->stage_pack[1]);
        }
      );
    }

    /**
     * @brief Perform one full Euler integration step.
     *
     * Evaluates all forces (generic, pairwise, link) into the stage-1
     * increment buffers, then applies the Euler update.
     *
     * @param dt          Time step size.
     * @param random_pool Kokkos random number pool for stochastic processes.
     * @param forces      The force functors to evaluate.
     */
    template<typename RandomPool, typename... Forces>
    void integrate(double dt, RandomPool& random_pool, Forces... forces) {
      this->evaluate_forces(true, random_pool, this->stage_pack[0], this->stage_pack[1], forces...);
      apply_euler(dt);
    }
  };
} // namespace tecs::integrators

#endif // TECS_INTEGRATORS_EULER_HPP
