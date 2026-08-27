// example translated from https://github.com/germannp/yalla/blob/main/examples/epithelia_double_polarity.cu

#include "../include/tecs.hpp"

using namespace tecs;

// Epithelial cells with two polarity vectors,
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_COM_FIXER(com_fixers::GlobalComFixer)
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity_a),
    (Polarity, polarity_b)
  )
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 300;
const int steps = 500;
const double dt = 0.1;
const Scalar r_max = 1.0;
const int save_every_nth = 5; // skip_step

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/epithelia_double_polarity", r_max);
  auto& positions = sim.get_view<FIELD(Vector, position)>();
  auto& polarities_a = sim.get_view<FIELD(Polarity, polarity_a)>();
  auto& polarities_b = sim.get_view<FIELD(Polarity, polarity_b)>();
  auto init_polarities = INIT_FUNC(
    polarities_a(i) = Polarity(positions(i));
    polarities_b(i) = Polarity(Kokkos::acos(0.0f), Kokkos::atan2(0.0f, 1.0f));
  );
  sim.init_random_filled_sphere(3.0f, init_polarities());

  auto force_a = PAIRWISE_FORCE(
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.8f, 1.0f, 1.5f);

    auto bending_force = ctx.polarity_a.self.bending_force(displacement, ctx.polarity_a.other, distance);
    ctx.position.delta += bending_force.vector * 0.3f;
    ctx.polarity_a.delta += bending_force.polarity * 0.3f;

    drag += 1.0f;
  );

  auto force_b = PAIRWISE_FORCE(
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.8f, 1.0f, 1.5f);

    auto bending_force = ctx.polarity_b.self.bending_force(displacement, ctx.polarity_b.other, distance);
    ctx.position.delta += bending_force.vector * 0.3f;
    ctx.polarity_b.delta += bending_force.polarity * 0.3f;

    drag += 1.0f;
  );

  for (int i = 0; i < steps / 2; ++i) {
    sim.take_step(dt, force_a());
    if (i % save_every_nth == 0)
      sim.write(i * dt);
  }
  for (int i = steps / 2; i < steps + 1; ++i) {
    sim.take_step(dt, force_b());
    if (i % save_every_nth == 0)
      sim.write(i * dt);
  }

  return 0;
}
