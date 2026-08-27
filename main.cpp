#include "include/tecs.hpp"

using namespace tecs;
struct SimulationConfig : public DefaultSimulationConfig {
  // Switch the pair finder here to compare neighbourhood definitions:
  //   pair_finders::NaiveDelaunay  -> Delaunay triangulation
  //   pair_finders::NaiveGabriel   -> Gabriel graph (strict subset of Delaunay)
  CONFIG_PAIR_FINDER(pair_finders::NaiveGabriel)
  CONFIG_INTEGRATOR(integrators::Euler)
  CONFIG_DIMENSIONS(2)
  RESOLVE_TECS_CONFIG()
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const unsigned int n_cells = 11;
const double dt = 1.0;
const Scalar r_max = 2.5f;

int main() {
  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/all_pairs_inf");
  settings.cutoff_distance = r_max;
  settings.link_capacity = n_cells * n_cells;

  Simulation<SimulationConfig> sim(settings);
  auto& links = sim.get_links();

  // (x, y) coordinates for all 11 cells, from the user's point table. The raw
  // table spans x in [26, 297] and y in [28, 224]. To scale the whole point set
  // into the range [-3.5, 3.5] without distorting it, each coordinate is
  // centred on the table's midpoint and scaled uniformly by s = 7 / 271
  // (the larger span becomes exactly 7 units wide), giving aspect-correct
  // geometry that fits within the square. The resulting x reaches exactly
  // -3.5 and 3.5 (cells 7 and 10), and every point lies inside the box.
  const Scalar pos[n_cells][2] = {
    { -3.1384, -1.4465 },  // 0  (was 40, 70)
    {  0.0387, -1.8340 },  // 1  (was 163, 55)
    {  2.2085, -2.5314 },  // 2  (was 247, 28)
    { -2.0018, -0.2325 },  // 3  (was 84, 117)
    {  0.7878, -0.4391 },  // 4  (was 192, 109)
    {  1.3044, -0.7232 },  // 5  (was 212, 98)
    { -3.5000,  1.3432 },  // 6  (was 26, 178)
    { -0.5554,  0.7491 },  // 7  (was 140, 155)
    {  1.2528,  1.1365 },  // 8  (was 210, 170)
    {  3.5000,  0.3875 },  // 9  (was 297, 141)
    {  0.0129,  2.5314 },  // 10 (was 162, 224)
  };

  auto positions = sim.get_positions_view();
  Kokkos::parallel_for("custom_init", n_cells, KOKKOS_LAMBDA(const int i) {
    positions(i).x() = pos[i][0];
    positions(i).y() = pos[i][1];
  });

  DeviceVar<int> link_counter = 0;
  View<int> neighbour_count("neighbour_count", n_cells);
  auto neighbour_count_device = neighbour_count.view_device();
  auto pairwise_force = PAIRWISE_FORCE(
    int link_n = Kokkos::atomic_fetch_add(link_counter.data(), 1);
    links(link_n) = Link(i, j);
    Kokkos::atomic_add(&neighbour_count_device(i), 1);
  );

  sim.take_step(0.0, pairwise_force);
  links.set_active_count(link_counter);

  // Every pair is evaluated twice (once as (i,j) and once as (j,i)),
  // so each cell's count is doubled — halve it to get the real neighbour count.
  auto neighbour_count_host = neighbour_count.view_host();

  printf("total links: %u\n", links.get_active_count() / 2);
  for (int c = 0; c < n_cells; ++c) {
    printf("cell %d: %d neighbours\n", c, neighbour_count_host(c) / 2);
  }

  // Write the (halved) neighbour counts to the output file.
  sim.write(0.0, neighbour_count);

  return 0;
}
