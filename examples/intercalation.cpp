// example translated from https://github.com/germannp/yalla/blob/main/examples/intercalation.cu
// Simulate intercalating cells

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_COM_FIXER(com_fixers::GlobalComFixer)
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(DefaultSimulationConfig)

const int n_cells = 500;
const int steps = 250;
const double dt = 0.2;
const Scalar r_max = 1.0;
const Scalar r_min = 0.5;
const int protrusions_per_cell = 1;

const Scalar min_link_length = 1.0;
const Scalar max_link_length = 2.0 * 2.0;
const Scalar link_strength = 0.2;

int main() {
  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/intercalation");
  settings.cutoff_distance = r_max;
  settings.link_capacity = n_cells * protrusions_per_cell;
  settings.link_active_count = n_cells * protrusions_per_cell;

  Simulation<SimulationConfig> sim(settings);
  auto& positions = sim.get_view<FIELD(Vector, position)>();
  auto& links = sim.get_links();
  sim.init_random_filled_sphere(2.5);

  unsigned int agent_count = sim.get_agent_count();
  auto update_protrusions = UPDATE_FUNC(
    Link& link = links(i);

    // check if link has to be destroyed due to distance changes
    Scalar distance = positions(link.a).distance_to(positions(link.b));
    if (distance < min_link_length || distance > max_link_length) {
      link.a = 0;
      link.b = 0;
    }

    // create new link
    unsigned int new_a = (i + 0.5) / protrusions_per_cell;
    unsigned int new_b = rng.urand(agent_count - 1);
    if (new_a == new_b)
      return;

    Vector displacement = positions(new_a) - positions(new_b);
    distance = displacement.length();
    if (distance > min_link_length && distance < max_link_length && Kokkos::abs(displacement.x()) / distance < 0.2) {
      link.a = new_a;
      link.b = new_b;
    }
  );

  auto clipped_cubic = PAIRWISE_FORCE(
    Scalar tmp = r_max - distance;
    ctx.position.delta += (2.0 * (r_min - distance) * tmp + tmp * tmp) * displacement / distance;
    drag += 1.0;
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
