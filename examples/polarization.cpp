// example translated from https://github.com/germannp/yalla/blob/main/examples/polarization.cu
// Simulate tissue polarization

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity)
  )
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 200;
const int steps = 300;
const double dt = 0.025;
const Scalar r_max = 1.0;
const Scalar r_min = 0.6;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/polarization", r_max);
  auto& polarities_view = sim.get_view<FIELD(Polarity, polarity)>();
  auto initial_conditions = INIT_FUNC(
    polarities_view(i) = Polarity{
      Kokkos::acos(2.0 * rng.drand(0.0, 1.0) - 1.0),
      2.0 * Kokkos::numbers::pi_v<Scalar> * rng.drand(0.0, 1.0)
    };
  );

  sim.init_random_filled_sphere(1.5, initial_conditions());
  sim.write(0.0);

  auto polarization = PAIRWISE_FORCE(
    Scalar F = 2.0 * (r_min - distance) * (r_max - distance) + Kokkos::pow(r_max - distance, 2);
    ctx.position.delta += displacement * F / distance;

    // U_Pol = -(Σ(n_i . n_j)^2) / 2
    ctx.polarity.delta += ctx.polarity.self.bidirectional_polarization_force(ctx.polarity.other);
  );

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, polarization());
    sim.write(i * dt);
  }

  return 0;
}
