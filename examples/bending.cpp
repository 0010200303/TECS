// example translated from https://github.com/germannp/yalla/blob/main/examples/bending.cu
// Relax bent epithelium

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity)
  )
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 91;
const int steps = 2000;
const double dt = 0.1;
const Scalar r_max = 1.0;
const Scalar pi_6 = Kokkos::numbers::pi_v<Scalar> / 6;
const Scalar radius = 1.6;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/bending", r_max);
  auto& positions = sim.get_view<FIELD(Vector, position)>();
  auto& polarities = sim.get_view<FIELD(Polarity, polarity)>();
  auto wrap_hexagon = INIT_FUNC(
    auto& x = positions(i).x();
    auto& y = positions(i).y();
    auto& z = positions(i).z();

    // Rotate by pi/6 to reduce negative curvature from tips
    x = Kokkos::cos(pi_6) * x - Kokkos::sin(pi_6) * y;
    y = Kokkos::sin(pi_6) * x + Kokkos::cos(pi_6) * y;

    // Wrap around cylinder
    auto phi = x / radius;
    if (phi == 0)
      phi = 0.01;
    x = radius * Kokkos::sin(phi);
    z = radius * Kokkos::cos(phi);
    polarities(i).theta() = phi;
  );
  sim.init_regular_hexagon(0.75, wrap_hexagon());

  // ReLU forces plus k*(n_i . r_ij/r)^2/2 for all r_ij <= r_max
  auto layer_force = PAIRWISE_FORCE(
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.7f, 2.0f, 0.8f, 1.0f);

    auto bending_force = ctx.polarity.self.bending_force(displacement, ctx.polarity.other, distance);
    ctx.position.delta += bending_force.vector * 0.5f;
    ctx.polarity.delta += bending_force.polarity * 0.5f;
  );

  for (int i = 0; i < steps + 1; ++i) {
    sim.take_step(dt, layer_force());
    sim.write(i * dt);
  }

  return 0;
}
