// example translated from https://github.com/germannp/yalla/blob/main/examples/lineage_tracing.cu
// Simulate lineage tracing of a group of dividing cells

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_INTEGRATOR(integrators::Euler)
  CONFIG_FIELDS(
    (Vector, position)
  )
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_max = 5000;
const int n_cells = 5;
const int steps = 1000;
const double dt = 0.1;
const Scalar r_max = 1.0;
const Scalar mean_distance = 0.75;
const Scalar proliferation_rate = 0.005;

int main() {
  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/lineage_tracing");
  settings.capacity = n_max;
  settings.cutoff_distance = r_max;
  settings.link_capacity = n_max;

  Simulation<SimulationConfig> sim(settings);
  auto& positions = sim.get_view<FIELD(Vector, position)>();
  auto& links = sim.get_links();
  View<int> families("families", n_cells, n_max);
  View<int> generations("generations", n_cells, n_max);
  auto init = INIT_FUNC(
    families(i) = i;
    generations(i) = 0;
  );
  sim.init_line(mean_distance, init());

  auto relaxation_force = PAIRWISE_FORCE(
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.8f, 2.0f, 1.0f);
  );

  DeviceVar<int> counter = sim.get_agent_count();
  DeviceVar<int> link_counter = links.get_active_count();
  auto proliferate = UPDATE_FUNC(
    if (rng.drand(1.0) > proliferation_rate)
      return;
    
    int n = Kokkos::atomic_fetch_add(counter.data(), 1);
    int link_n = Kokkos::atomic_fetch_add(link_counter.data(), 1);

    Polarity temp_polarity = Polarity(
      Kokkos::acos(2.0 * rng.drand(0.0, 1.0) - 1),
      rng.drand(0.0, 1.0) * 2.0 * Kokkos::numbers::pi_v<Scalar>
    );
    positions(n) = positions(i) + mean_distance / 4 * temp_polarity.to_vector3();

    // cell lineage tracing
    families(n) = families(i);
    generations(n) = generations(i) + 1;
    links(link_n) = Link(i, n);
  );

  for (int i = 0; i < steps + 1; ++i) {
    sim.take_step(dt, relaxation_force());
    sim.run(proliferate());

    sim.set_agent_count(counter, families, generations);
    links.set_active_count(link_counter);

    sim.write(i * dt, families, generations);
  }

  return 0;
}
