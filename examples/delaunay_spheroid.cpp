// Relax a 3D cell spheroid using Delaunay-neighbor forces.
//
// Cells interact only with their Delaunay neighbours — the natural topology
// for a dense 3D aggregate — via adhesion and repulsion.  In 3D the
// Delaunay triangulation connects cells into tetrahedra, capturing the
// correct local neighbourhood for a confluent spheroid.
//
// The Delaunay triangulation is recomputed at every time step so that the
// interaction graph evolves correctly as cells move.

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_PAIR_FINDER(pair_finders::NaiveDelaunay)
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 100;
const int steps = 300;
const double dt = 0.001;
const Scalar r_max = 1.0f;

int main() {
  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/delaunay_spheroid");
  settings.cutoff_distance = r_max;
  settings.link_capacity = n_cells * n_cells * n_cells / 2;

  Simulation<SimulationConfig> sim(settings);
  auto& links = sim.get_links();
  sim.init_random_filled_sphere(3.0);

  DeviceVar<int> link_counter = links.get_active_count();
  auto generate_links = PAIRWISE_FORCE(
    if (is_full_step == false)
      return;

    int link_n = Kokkos::atomic_fetch_add(link_counter.data(), 1);
    links(link_n) = Link(i, j);
  );
  sim.take_step(0.0, generate_links);
  links.set_active_count(link_counter);
  sim.write(0.0);

  auto spheroid_forces = PAIRWISE_FORCE(
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.5f) * 3.0;
  );

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, spheroid_forces());
    sim.write(i * dt);
  }

  return 0;
}
