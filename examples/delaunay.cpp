// visualize delauney triangulation using links

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_PAIR_FINDER(pair_finders::NaiveDelaunay)
  CONFIG_INTEGRATOR(integrators::Euler)
  CONFIG_DIMENSIONS(2)
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const unsigned int n_cells = 4096;
const double dt = 1.0;
const Scalar r_max = 1.0f;

int main() {
  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/delaunay");
  settings.cutoff_distance = r_max;
  settings.link_capacity = n_cells * n_cells;

  Simulation<SimulationConfig> sim(settings);
  auto& links = sim.get_links();
  sim.init_random_disk(0.5);

  DeviceVar<int> link_counter = links.get_active_count();
  auto generate_links = PAIRWISE_FORCE(
    int link_n = Kokkos::atomic_fetch_add(link_counter.data(), 1);
    links(link_n) = Link(i, j);
  );

  sim.take_step(0.0, generate_links);
  links.set_active_count(link_counter);
  sim.write(0.0);

  return 0;
}
