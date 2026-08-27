// example translated from https://github.com/germannp/yalla/blob/main/examples/apical_constriction.cu
// Simulate a apical constriction in an epithelial layer

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity)
  )
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 225;
const int steps = 6000;
const double dt = 0.1;
const Scalar r_max = 1.0;
const auto preferential_angle_deviation = 20.f * Kokkos::numbers::pi_v<Scalar> / 180.f; // in radiants
const int save_every_nth = 40;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/apical_constriction", r_max);
  auto& polarities = sim.get_view<FIELD(Polarity, polarity)>();
  auto init_polarities = INIT_FUNC(
    polarities(i) = Polarity(Kokkos::acos(1.0f), Kokkos::atan2(1.0f, 1.0f));
  );
  sim.init_regular_rectangle(0.8, 15, init_polarities());
  sim.write(0.0);

  auto constriction_force = PAIRWISE_FORCE(
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.8f, 0.8f, 2.0f, 2.0f);

    auto bending_force = ctx.polarity.self.apical_constriction_force(displacement, distance, ctx.polarity.other, 
      Kokkos::numbers::pi_v<Scalar> / 2.0f - preferential_angle_deviation);
    ctx.position.delta += bending_force.vector * 0.6f;
    ctx.polarity.delta += bending_force.polarity * 0.6f;
  );

  for (int i = 1; i < steps + 1; ++i) {
    sim.take_step(dt, constriction_force);
    if (i % save_every_nth == 0)
      sim.write(i * dt);
  }

  return 0;
}
