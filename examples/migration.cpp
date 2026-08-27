// example translated from https://github.com/germannp/yalla/blob/main/examples/migration.cu
// Simulate mono-polar migration

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity)
  )
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 262;
const int steps = 100;
const double dt = 0.05;
const Scalar r_max = 1.0;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/migration", r_max);
  auto& positions_view = sim.get_view<FIELD(Vector, position)>();
  auto& polarities_view = sim.get_view<FIELD(Polarity, polarity)>();
  auto initial_conditions = INIT_FUNC(
    if (i == 0) {
      positions_view(i) = Vector{0.0};
      polarities_view(i) = Polarity{0.0, 0.01};
    }
  );

  sim.init_relaxed_cuboid(Vector{-1.5, -1.5, 0.0}, Vector{1.5, 1.5, 10.0}, 2000, initial_conditions());
  sim.write(0.0);

  auto relu_w_migration = PAIRWISE_FORCE(
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.7f, 1.0f, 2.0f) +
      ctx.polarity.self.migration_force(displacement, ctx.polarity.other, distance);
  );

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, relu_w_migration());
    sim.write(i * dt);
  }

  return 0;
}
