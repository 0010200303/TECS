// example translated from https://github.com/germannp/yalla/blob/main/examples/sorting_prot.cu
// Simulate cell sorting by protrusions

#include "../include/tecs.hpp"

using namespace tecs;
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(DefaultSimulationConfig)

const int n_cells = 200;
const int steps = 300;
const double dt = 0.05;
const Scalar r_max = 1.0;
const Scalar r_min = 0.5;
const int n_protrusions = n_cells * 5;

const Scalar min_link_length_squared = 1.0;
const Scalar max_link_length_squared = 2.0 * 2.0;
const Scalar link_strength = 0.2;

KOKKOS_INLINE_FUNCTION
constexpr bool get_type(const unsigned int i) {
  return i < n_cells / 2;
}

int main() {
  Simulation<DefaultSimulationConfig>::Settings settings(n_cells, "./output/sorting_prot");
  settings.cutoff_distance = r_max;
  settings.link_capacity = n_protrusions;
  settings.link_active_count = n_protrusions;

  Simulation<DefaultSimulationConfig> sim(settings);
  auto& positions = sim.get_view<FIELD(Vector, position)>();
  auto& links = sim.get_links();
  View<bool> types("types", n_cells);
  auto types_init = INIT_FUNC(
    types(i) = !get_type(i);
  );
  sim.init_random_filled_sphere(2.0, types_init());
  sim.write_static(types);

  unsigned int agent_count = sim.get_agent_count();
  auto update_protrusions = UPDATE_FUNC(
    Link& link = links(i);

    // check if link has to be destroyed due to distance changes
    Scalar distance_squared = positions(link.a).distance_to_squared(positions(link.b));
    if (distance_squared < min_link_length_squared || distance_squared > max_link_length_squared) {
      link.a = 0;
      link.b = 0;
    }

    // probability testing to keep link
    if (get_type(link.a) == true && get_type(link.b) == true) {
      if (rng.drand(0.0, 1.0) > 0.025)
        return;
    }
    else if (get_type(link.a) == false && get_type(link.b) == false) {
      if (rng.drand(0.0, 1.0) > 0.25)
        return;
    }
    else {
      if (rng.drand(0.0, 1.0) > 0.25)
        return;
    }

    // create new link
    int new_a = rng.urand(0, agent_count - 1);
    int new_b = rng.urand(0, agent_count - 1);
    if (new_a == new_b)
      return;

    distance_squared = positions(new_a).distance_to_squared(positions(new_b));
    if (distance_squared > min_link_length_squared && distance_squared < max_link_length_squared) {
      link.a = new_a;
      link.b = new_b;
    }
  );

  auto clipped_cubic = PAIRWISE_FORCE(
    Scalar tmp = r_max - distance;
    ctx.position.delta += (2.0 * (r_min - distance) * tmp + tmp * tmp) * displacement / distance;
  );

  auto prot_forces = LINK_FORCE(
    Vector displacement = ctx.position.b - ctx.position.a;
    Scalar distance = displacement.length();

    ctx.position.delta_a +=  link_strength * displacement / distance;
    ctx.position.delta_b += -link_strength * displacement / distance;
  );

  for (int i = 0; i < steps + 1; ++i) {
    sim.run_links(update_protrusions());
    sim.take_step(dt, prot_forces(), clipped_cubic());
    sim.write(i * dt);
  }

  return 0;
}
