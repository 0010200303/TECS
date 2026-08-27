// example translated from https://github.com/germannp/yalla/blob/main/examples/turing.cu
// Simulate Meinhardt equations within an epithelium

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_COM_FIXER(com_fixers::GlobalComFixer)
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity),
    (Scalar, u),
    (Scalar, v)
  )
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 500;
const int steps = 10000;
const int save_every_nth = 100; // skip_steps
const Scalar r_max = 1.0;
const Scalar r_min = 0.6;

const Scalar lambda = 1.0;
const Scalar D_v = 4.0;
const Scalar f_v = 1.0;
const Scalar f_u = 80.0;
const Scalar g_u = 40.0;
const Scalar m_u = 0.25;
const Scalar m_v = 0.5;
const Scalar s_u = 0.05;
const Scalar D_u = 0.1;

const double dt = 0.05 * r_min * r_min / D_v;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/turing", r_max);
  auto& polarities = sim.get_view<FIELD(Polarity, polarity)>();
  auto& u_view = sim.get_view<FIELD(Scalar, u)>();
  auto& v_view = sim.get_view<FIELD(Scalar, v)>();
  auto init = INIT_FUNC(
    polarities(i).theta() = Kokkos::numbers::pi_v<Scalar> / 2.0f;
    u_view(i) = rng.drand(-0.1, 0.1);
    v_view(i) = rng.drand(-0.1, 0.1);
  );
  sim.init_random_disk(0.5, init());
  sim.write(0.0);

  auto meinhardt_equations = GENERIC_FORCE(
    ctx.u.delta += lambda * ((f_u * ctx.u.self * ctx.u.self) / (1.0f + f_v * ctx.v.self) - m_u * ctx.u.self + s_u);
    ctx.v.delta += lambda * (g_u * ctx.u.self * ctx.u.self - m_v * ctx.v.self);
  );

  auto epithelium_w_turing = PAIRWISE_FORCE(
    ctx.u.delta -= D_u * (ctx.u.self - ctx.u.other);
    ctx.v.delta -= D_v * (ctx.v.self - ctx.v.other);

    ctx.position.delta += (2.0f * (r_min - distance) * (r_max - distance) + Kokkos::pow(r_max - distance, 2.0f))
      * displacement / distance;
    
    auto bending_force = ctx.polarity.self.bending_force(displacement, ctx.polarity.other, distance);
    ctx.position.delta += bending_force.vector * 3.0f;
    ctx.polarity.delta += bending_force.polarity * 3.0f;

    Vector f = (2.0f * (r_min - distance) * (r_max - distance) + Kokkos::pow(r_max - distance, 2.0f))
      * displacement / distance;

    drag += 1.0f;
  );

  for (int i = 1; i < steps + 1; ++i) {
    sim.take_step(dt, meinhardt_equations(), epithelium_w_turing());
    if (i % save_every_nth == 0)
      sim.write(i * dt);
  }

  return 0;
}
