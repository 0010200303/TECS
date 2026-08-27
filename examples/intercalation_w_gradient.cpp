// example translated from https://github.com/germannp/yalla/blob/main/examples/intercalationw_gradient.cu
// Simulate mesenchymal intercalation orchestrated by epithelial signals

#include "../include/tecs.hpp"

enum CellType {
  Mesenchyme,
  Epithelium
};

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_COM_FIXER(com_fixers::GlobalComFixer)
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity),
    (Scalar, w),
    (Scalar, f)
  )
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_max = 15'000;
const int steps = 500;
const double dt = 0.1;
const Scalar r_max = 1.0;
const Scalar r_min = 0.8;
const int protrusions_per_cell = 1;
const Scalar protrusion_radius = 2.0;
const Scalar proliferation_rate = 0.015;
const Scalar link_strength = 0.2;

int main() {
  // load initial conditions
  io::HDF5_Reader input_reader("./examples/data/sphere_ic.h5");

  int n_cells = input_reader.get_dataset_dimensions("POINTS")[0];
  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/intercalation_w_gradient");
  settings.capacity = n_max;
  settings.cutoff_distance = r_max;
  settings.link_capacity = n_max * protrusions_per_cell;
  settings.link_active_count = n_cells * protrusions_per_cell;

  Simulation<SimulationConfig> sim(settings);
  auto& positions = sim.get_view<FIELD(Vector, position)>();
  auto& polarities = sim.get_view<FIELD(Polarity, polarity)>();
  auto& ws = sim.get_view<FIELD(Scalar, w)>();
  auto& fs = sim.get_view<FIELD(Scalar, f)>();
  auto& links = sim.get_links();
  View<int> types("types", n_cells, n_max);
  View<int> mesenchyme_neighbours("mesenchyme_neighbours", n_cells, n_max);
  View<int> epithelium_neighbours("epithelium_neighbours", n_cells, n_max);

  input_reader.read_dataset("POINTS", positions);
  input_reader.read_dataset("polarity", polarities);
  input_reader.read_dataset("cell_type", types);

  auto init = INIT_FUNC(
    ws(i) = 0.0;
    if (types(i) == CellType::Mesenchyme)
      return;
    
    Vector& position_i = positions(i);
    if (position_i.z() <= 0.0)
      return;
    
    ws(i) = 1.0;
    if (position_i.x() > 0.0 && Kokkos::abs(position_i.y()) < 2.5 && position_i.z() < 3.0)
      fs(i) = 1.0;
  );
  sim.init(init());

  acceleration::Grid protrusions_grid(positions, 0, Vector(0), Vector(0), 0.0);
  auto update_protrusions = UPDATE_FUNC(
    Link& link = links(i);
    const VectorI bin_extents = protrusions_grid.get_bin_extents();

    // create new link candidate
    int new_a = (i + 0.5) / protrusions_per_cell;
    if (types(new_a) != CellType::Mesenchyme)
      return;

    auto filter = [&](int j) -> bool {
      return j != new_a && types(j) == CellType::Mesenchyme;
    };
    int new_b = protrusions_grid.get_random_point_index_in_neighbourhood(
      positions(new_a), protrusion_radius, rng, filter
    );
    if (new_b < 0)
      return;

    // compare old link vs. candidate
    Scalar new_distance_squared = positions(new_a).distance_to_squared(positions(new_b));
    if (new_distance_squared > protrusion_radius * protrusion_radius)
      return;
    if (link.a == link.b) {
      link.a = new_a;
      link.b = new_b;
      return;
    }

    Scalar noise = rng.drand();
    Scalar new_distance = Kokkos::sqrt(new_distance_squared);
    Scalar old_distance = positions(link.a).distance_to(positions(link.b));

    // cells close to w source align normal to f gradient
    if (ws(new_a) + ws(new_b) > 0.3) {
      if (Kokkos::abs(fs(new_a) - fs(new_b)) / new_distance < Kokkos::abs(fs(link.a) - fs(link.b)) / old_distance * (1.0 - noise)) {
        link.a = new_a;
        link.b = new_b;
      }
    }
    // cells far from w source align along w gradient
    else {
      if (Kokkos::abs(ws(new_a) - ws(new_b)) / new_distance > Kokkos::abs(ws(link.a) - ws(link.b)) / old_distance * (1.0 - noise)) {
        link.a = new_a;
        link.b = new_b;
      }
    }
  );

  auto intercalation = LINK_FORCE(
    Vector displacement = ctx.position.b - ctx.position.a;
    Scalar distance = displacement.length();

    ctx.position.delta_a +=  link_strength * displacement / distance;
    ctx.position.delta_b += -link_strength * displacement / distance;
  );

  auto generic_force = GENERIC_FORCE(
    int type_i = types(i);
    ctx.w.delta += -0.01 * (type_i == CellType::Mesenchyme) * ctx.w.self;
    ctx.f.delta += -0.01 * (type_i == CellType::Mesenchyme) * ctx.f.self;
  );

  auto pairwise_force = PAIRWISE_FORCE(
    int type_i = types(i);
    int type_j = types(j);

    if (type_i == type_j) {
      if (type_i == CellType::Mesenchyme)
        ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.8f, 2.0f, 1.0f);
      else
        ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.8f, 2.0f, 2.0f);
    }
    else {
      ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.9f, 2.0f, 2.0f);
    }

    ctx.w.delta += -(ctx.w.self - ctx.w.other) * (type_i == CellType::Mesenchyme) * 0.1;
    ctx.f.delta += -(ctx.f.self - ctx.f.other) * (type_i == CellType::Mesenchyme) * 0.1;
    
    if (type_j == CellType::Mesenchyme)
      Kokkos::atomic_add(&mesenchyme_neighbours(i), 1);
    else
      Kokkos::atomic_add(&epithelium_neighbours(i), 1);

    if (type_i == CellType::Epithelium && type_j == CellType::Epithelium) {
      auto result = ctx.polarity.self.bending_force(displacement, ctx.polarity.other, distance);
      ctx.position.delta += result.vector * 0.15;
      ctx.polarity.delta += result.polarity * 0.15;
    }
    drag += 1.0;
  );

  DeviceVar<int> counter = sim.get_agent_count();
  auto proliferate = UPDATE_FUNC(
    // check conditions
    if (types(i) == CellType::Mesenchyme)
      return;
    if (mesenchyme_neighbours(i) < 1 || epithelium_neighbours(i) > 7)
      return;
    if (rng.drand(1.0) > proliferation_rate)
      return;
    
    int n = Kokkos::atomic_fetch_add(counter.data(), 1);

    Polarity temp_polarity = Polarity(
      Kokkos::acos(2.0 * rng.drand(0.0, 1.0) - 1),
      rng.drand(0.0, 1.0) * 2.0 * Kokkos::numbers::pi_v<Scalar>
    );
    positions(n) = positions(i) + r_min / 4 * temp_polarity.to_vector3();
    polarities(n) = polarities(i);
    types(n) = types(i);
    ws(n) = ws(i);
    fs(n) = fs(i);
  );

  for (int i = 0; i < steps + 1; ++i) {
    protrusions_grid = acceleration::Grid(
      positions,
      sim.get_agent_count(),
      utils::get_bounds<Vector>(positions),
      protrusion_radius
    );
    protrusions_grid.rebuild();

    mesenchyme_neighbours.deep_copy(0);
    epithelium_neighbours.deep_copy(0);

    sim.run(update_protrusions());
    sim.take_step(dt, intercalation(), generic_force(), pairwise_force());
    sim.run(proliferate());

    sim.set_agent_count(counter);
    links.set_active_count(sim.get_agent_count() * protrusions_per_cell);

    sim.write(i * dt, types);
  }

  return 0;
}
