// example translated from https://github.com/germannp/yalla/blob/main/examples/passive_growth.cu
// Simulates branching on a spheroid induced by Turing mechanism on surface

#include "../include/tecs.hpp"

enum CellType {
  Mesenchyme,
  Epithelium
};

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_PAIR_FINDER(pair_finders::BinnedAllPairs)
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity),
    (Scalar, u),
    (Scalar, v)
  )
)
EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_max = 500'000;
const int n_cells = 500;
const int steps = 4000;
const int save_every_nth = 10;
const double dt = 0.2;
const Scalar r_max = 1.0;

// turing parameters
const Scalar lambda = 0.0075;
const Scalar D_u = 0.001;
const Scalar D_v = 0.2;
const Scalar f_v = 1.0;
const Scalar f_u = 80.0;
const Scalar g_u = 80.0;
const Scalar m_u = 0.25;
const Scalar m_v = 0.75;
const Scalar s_u = 0.05;

const Scalar mean_distance = 0.75;
const Scalar mesenchyme_proliferation_rate = 0.1;
const Scalar epithelium_proliferation_rate = 0.2;
// Threshold of v that allows mesenchyme cells to divide
const auto proliferation_threshold = 1150.0f;

int main() {
  PairFinder::Settings pair_finder_settings;
  pair_finder_settings.min_bounds = Vector(-40.0, -40.0, -40.0);
  pair_finder_settings.max_bounds = Vector( 40.0,  40.0,  40.0);

  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/branching_lineage_tracing_longeer");
  settings.capacity = n_max;
  settings.cutoff_distance = r_max;
  settings.pair_finder_settings = pair_finder_settings;
  settings.link_capacity = n_max;

  Simulation<SimulationConfig> sim(settings);
  auto& positions = sim.get_view<FIELD(Vector, position)>();
  auto& polarities = sim.get_view<FIELD(Polarity, polarity)>();
  auto& us = sim.get_view<FIELD(Scalar, u)>();
  auto& vs = sim.get_view<FIELD(Scalar, v)>();
  auto& links = sim.get_links();
  View<int> types("types", n_cells, n_max);
  View<int> mesenchyme_neighbours("mesenchyme_neighbours", n_cells, n_max);
  View<int> epithelium_neighbours("epithelium_neighbours", n_cells, n_max);
  View<int> families("families", n_cells, n_max);
  View<int> generations("generations", n_cells, n_max);
  View<Scalar> spawn_times("spawn_times", n_cells, n_max);

  auto count_neighbours = PAIRWISE_FORCE(
    if (is_full_step == false)
      return;

    if (types(j) == CellType::Mesenchyme)
      Kokkos::atomic_add(&mesenchyme_neighbours(i), 1);
    else
      Kokkos::atomic_add(&epithelium_neighbours(i), 1);
  );
  sim.init_relaxed_sphere(3.0);
  sim.take_step(0.0, count_neighbours());

  auto init = INIT_FUNC(
    families(i) = i;
    generations(i) = 0;
    spawn_times(i) = 0.0f;

    if (mesenchyme_neighbours(i) >= 10)
      return;
    
    types(i) = CellType::Epithelium;
    polarities(i) = Polarity(positions(i));
    us(i) = rng.drand(-0.1, 0.1);
    vs(i) = rng.drand(-0.1, 0.1);
  );
  sim.init(init());

  auto meinhardt_equations = GENERIC_FORCE(
    if (types(i) != CellType::Epithelium)
      return;

    Scalar u_delta = lambda * ((f_u * ctx.u.self * ctx.u.self) / (1.0f + f_v * ctx.v.self) - m_u * ctx.u.self + s_u);
    Scalar v_delta = lambda * (g_u * ctx.u.self * ctx.u.self - m_v * ctx.v.self);;

    // prevent negative values
    if (-u_delta <= ctx.u.self)
      ctx.u.delta += u_delta;
    if (-v_delta <= ctx.v.self)
      ctx.v.delta += v_delta;
  );

  auto epithelium_w_turing = PAIRWISE_FORCE(
    int type_i = types(i);
    int type_j = types(j);

    if (type_i == type_j)
      ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.7f, 0.8f, 2.0f, 1.0f);
    else
      ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.8f, 0.9f, 2.0f, 1.0f);

    // diffusion
    if (type_i == CellType::Epithelium && type_j == CellType::Epithelium) {
      Scalar u_delta = -D_u * (ctx.u.self - ctx.u.other);
      Scalar v_delta = -D_v * (ctx.v.self - ctx.v.other);

      // prevent negative values
      if (-u_delta <= ctx.u.self)
        ctx.u.delta += u_delta;
      if (-v_delta <= ctx.v.self)
        ctx.v.delta += v_delta;

      auto bending_force = ctx.polarity.self.bending_force(displacement, ctx.polarity.other, distance);
      ctx.position.delta += bending_force.vector * 0.2;
      ctx.polarity.delta += bending_force.polarity * 0.2;
    }
    else {
      // diffuses into mesenchyme to induce proliferation
      ctx.v.delta -= D_v * (ctx.v.self - ctx.v.other);
    }

    if (is_full_step == false)
      return;

    if (types(j) == CellType::Mesenchyme)
      Kokkos::atomic_add(&mesenchyme_neighbours(i), 1);
    else
      Kokkos::atomic_add(&epithelium_neighbours(i), 1);
  );

  DeviceVar<int> counter("counter", sim.get_agent_count());
  DeviceVar<int> link_counter = links.get_active_count();
  DeviceVar<Scalar> current_time = 0.0f;
  auto proliferate = UPDATE_FUNC(
    // dividing new cells is problematic
    if (i >= counter * (1.0 - epithelium_proliferation_rate))
      return;

    int type_i = types(i);
    if (type_i == CellType::Mesenchyme) {
      if (vs(i) < proliferation_threshold || rng.drand(1.0) > mesenchyme_proliferation_rate)
        return;
    }
    else {
      if (mesenchyme_neighbours(i) < 1 || epithelium_neighbours(i) > 5 || rng.drand(1.0) > epithelium_proliferation_rate)
        return;
    }

    int n = Kokkos::atomic_fetch_add(counter.data(), 1);
    int link_n = Kokkos::atomic_fetch_add(link_counter.data(), 1);

    Polarity temp_polarity = Polarity(
      Kokkos::acos(2.0 * rng.drand(0.0, 1.0) - 1),
      rng.drand(0.0, 1.0) * 2.0 * Kokkos::numbers::pi_v<Scalar>
    );
    positions(n) = positions(i) + mean_distance / 4 * temp_polarity.to_vector3();
    polarities(n) = polarities(i);
    types(n) = types(i);

    us(i) *= 0.5;
    vs(i) *= 0.5;
    us(n) = us(i);
    vs(n) = vs(i);

    families(n) = families(i);
    generations(n) = generations(i) + 1;
    spawn_times(n) = current_time;
    links(link_n) = Link(i, n);
  );

  for (int i = 0; i < steps + 1; ++i) {
    sim.run(proliferate());
    sim.set_agent_count(counter, types, mesenchyme_neighbours, epithelium_neighbours,
      families, generations, spawn_times);

    mesenchyme_neighbours.deep_copy(0);
    epithelium_neighbours.deep_copy(0);

    sim.take_step(dt, meinhardt_equations(), epithelium_w_turing());

    if (i % save_every_nth == 0) {
      sim.write(i * dt, types, families, generations, spawn_times);
      std::cout << "step " << i << ": " << sim.get_agent_count() << " agents" << std::endl;
    }
    current_time = i * dt;
  }

  return 0;
}
