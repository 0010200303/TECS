// example translated from https://github.com/germannp/yalla/blob/main/examples/gradient.cu
// Simulate gradient formation

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_DIMENSIONS(2)
  CONFIG_FIELDS(
    (Vector, position),
    (Scalar, gradient)
  )
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 61;
const int steps = 200;
const double dt = 0.005;
const Scalar r_max = 1.0;
const Scalar D = 10;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/gradient", r_max);
  auto& gradient_view = sim.get_view<FIELD(Scalar, gradient)>();
  auto gradient_init = INIT_FUNC(
    if (i == 11)
      gradient_view(i) = Scalar(1);
  );
  sim.init_regular_hexagon(0.75, gradient_init());
  sim.write(0.0);

  auto diffusion = PAIRWISE_FORCE(
    if (i == 11)
      return;
    ctx.gradient.delta += -(ctx.gradient.self - ctx.gradient.other) * D;
  );

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, diffusion());
    sim.write(i * dt);
  }

  return 0;
}
