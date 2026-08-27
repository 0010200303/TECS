#ifndef TECS_INTEGRATORS_HEUN_HPP
#define TECS_INTEGRATORS_HEUN_HPP

#include <type_traits>
#include <Kokkos_Core.hpp>

#include "base.hpp"

namespace tecs::integrators {

  /**
   * @brief Heun's method (second-order explicit Runge–Kutta) integrator.
   *
   * Uses four integration stages:
   * - Stage 0: current simulation state (e.g. positions).
   * - Stage 1: first force evaluation.
   * - Stage 2: predicted state (Euler predictor).
   * - Stage 3: second force evaluation (corrector).
   *
   * The update rule is:
   * @f[
   *   \tilde{\mathbf{x}} = \mathbf{x}_n + f(\mathbf{x}_n) \cdot \Delta t
   * @f]
   * @f[
   *   \mathbf{x}_{n+1} = \mathbf{x}_n + \frac{f(\mathbf{x}_n) + f(\tilde{\mathbf{x}})}{2} \cdot \Delta t
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
  struct Heun : public Base<PairFinder, ComFixer, 4, GenericForceFields, PairwiseForceFields, LinkForceFields, Views...> {
    using Base<PairFinder, ComFixer, 4, GenericForceFields, PairwiseForceFields, LinkForceFields, Views...>::Base;

    /**
     * @brief Euler predictor: compute a first-guess state into stage 2.
     *
     * Optionally corrects for centre-of-mass drift, then computes the predicted
     * state @f$ \tilde{\mathbf{x}} = \mathbf{x}_n + \mathbf{v}_n \cdot \Delta t @f$.
     *
     * @param dt Time step size.
     */
    void apply_euler_predictor(double dt) {
      const auto com_fix_delta = this->com_fixer.fix(this->stage_pack[1].first());

      Kokkos::parallel_for(
        "apply_euler_predictor",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          this->stage_pack[1].first()(i) -= com_fix_delta;

          this->stage_pack[2].zip_apply([&](auto& predicted, const auto& current, const auto& delta) {
            predicted(i) = current(i) + delta(i) * dt;
          }, this->stage_pack[0], this->stage_pack[1]);
        }
      );
    }

    /**
     * @brief Heun corrector: average the first and second force evaluations.
     *
     * Computes the final velocity as the average of the two evaluations,
     * then updates the current state and clears both delta buffers.
     *
     * @param dt Time step size.
     */
    void apply_heun_corrector(double dt) {
      const auto com_fix_delta = this->com_fixer.fix(this->stage_pack[3].first());

      Kokkos::parallel_for(
        "apply_heun_corrector",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          this->stage_pack[3].first()(i) -= com_fix_delta;
          
          this->old_velocities(i) = (this->stage_pack[1].first()(i) + this->stage_pack[3].first()(i)) * 0.5;

          this->stage_pack[0].zip_apply([&](auto& current, auto& delta_0, auto& delta_1) {
            current(i) += (delta_0(i) + delta_1(i)) * 0.5 * dt;

            // clear delta buffers (faster than new deep_copy call)
            delta_0(i) = std::remove_cv_t<std::remove_reference_t<decltype(delta_0(i))>>{};
            delta_1(i) = std::remove_cv_t<std::remove_reference_t<decltype(delta_1(i))>>{};
          }, this->stage_pack[1], this->stage_pack[3]);
        }
      );
    }

    /**
     * @brief Perform one full Heun integration step.
     *
     * 1. Evaluate forces at the current state into stage 1.
     * 2. Compute the Euler predictor into stage 2.
     * 3. Evaluate forces at the predicted state into stage 3.
     * 4. Apply the Heun corrector (average + update).
     *
     * @param dt          Time step size.
     * @param random_pool Kokkos random number pool for stochastic processes.
     * @param forces      The force functors to evaluate.
     */
    template<typename RandomPool, typename... Forces>
    void integrate(double dt, RandomPool& random_pool, Forces... forces) {
      this->evaluate_forces(true, random_pool, this->stage_pack[0], this->stage_pack[1], forces...);
      apply_euler_predictor(dt);

      this->evaluate_forces(false, random_pool, this->stage_pack[2], this->stage_pack[3], forces...);
      apply_heun_corrector(dt);
    }
  };
} // namespace tecs::integrators

#endif // TECS_INTEGRATORS_HEUN_HPP