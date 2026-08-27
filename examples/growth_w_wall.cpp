// example translated from https://github.com/germannp/yalla/blob/main/examples/growth_w_wall.cu
// Simulate growing mesenchyme constrained by a planar wall

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_PAIR_FINDER(pair_finders::BinnedGabriel)
)
EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)

// pre allocate enough for the rest of the simulation
// see passive_growth.cpp for dynamic approach using resizing
const int n_max = 10000;
const int n_cells = 500;
const int steps = 500;
const int save_every_nth = steps / 100;
const double dt = 0.1;
const Scalar r_max = 1.0;
const Scalar mean_distance = 0.75;
const Scalar proliferation_rate = 0.005;

int main() {
  Plane wall(0.0f, 0.0f, 1.0f, 0.0f);

  Simulation<SimulationConfig> sim(n_max, "./output/growth_w_wall", r_max);
  sim.set_agent_count(n_cells);
  auto& positions = sim.get_view<FIELD(Vector, position)>();
  auto move_cells = INIT_FUNC(
    if (positions(i).z() < 0.0f)
      positions(i).z() *= -1.0f;
  );
  sim.init_random_filled_sphere(3.0f, move_cells());
  sim.write(0.0);

  auto relu_force = PAIRWISE_FORCE(
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.7f, 0.8f);
  );

  auto repulsion_from_wall = GENERIC_FORCE(
    Scalar sd = wall.signed_distance(ctx.position.self);
    if (sd < 0.0f)
      ctx.position.delta += wall.normal() * (-sd * 10.0f);
  );

  DeviceVar<int> counter = sim.get_agent_count();
  auto proliferate = UPDATE_FUNC(
    if (rng.drand(0.0, 1.0) > proliferation_rate)
      return;

    int n = Kokkos::atomic_fetch_add(counter.data(), 1);

    Polarity temp_polarity = Polarity(
      Kokkos::acos(2.0 * rng.drand(0.0, 1.0) - 1),
      rng.drand(0.0, 1.0) * 2.0 * Kokkos::numbers::pi_v<Scalar>
    );
    positions(n) = positions(i) + mean_distance / 4 * temp_polarity.to_vector3();
  );

  for (int i = 1; i < steps + 1; ++i) {
    sim.take_step(dt, relu_force(), repulsion_from_wall());
    sim.init(proliferate());
    sim.set_agent_count(counter);

    if (i % save_every_nth == 0)
      sim.write(i * dt);
  }

  return 0;
}
