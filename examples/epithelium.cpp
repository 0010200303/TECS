// example translated from https://github.com/germannp/yalla/blob/main/examples/epithelium.cu
// Simulate a mesenchyme-to-epithelium transition

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity)
  )
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 250;
const int steps = 100;
const double dt = 0.05;
const Scalar r_max = 1.0;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/epithelium", r_max);
  auto& positions = sim.get_view<FIELD(Vector, position)>();
  auto& polarities = sim.get_view<FIELD(Polarity, polarity)>();
  auto init_polarities = INIT_FUNC(
    polarities(i) = Polarity(positions(i));
  );
  sim.init_relaxed_sphere(0.8, init_polarities());

  // ReLU forces plus k*(n_i . r_ij/r)^2/2 for all r_ij <= r_max
  auto layer_force = PAIRWISE_FORCE(
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.7f, 0.8f, 1.0f);

    auto bending_force = ctx.polarity.self.bending_force(displacement, ctx.polarity.other, distance);
    ctx.position.delta += bending_force.vector;
    ctx.polarity.delta += bending_force.polarity;
  );

  for (int i = 0; i < steps + 1; ++i) {
    sim.take_step(dt, layer_force());
    sim.write(i * dt);
  }

  return 0;
}
