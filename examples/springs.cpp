// example translated from https://github.com/germannp/yalla/blob/main/examples/springs.cu
// Integrate N-body problem with springs between all bodies

#include "../include/tecs.hpp"

using namespace tecs;
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(DefaultSimulationConfig)

const int n_bodies = 800;
const int steps = 100;
const double dt = 0.001;
const Scalar L_0 = 0.5f;

int main() {
  Simulation<DefaultSimulationConfig> sim(n_bodies, "./output/springs");
  sim.init_random_filled_sphere(3.0);
  sim.write(0.0);

  auto spring = PAIRWISE_FORCE(
    ctx.position.delta += forces::Spring(displacement, distance, L_0);
  );

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, spring());
    sim.write(i * dt);
  }

  return 0;
}
