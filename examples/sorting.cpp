// Simulate cell sorting by forces strength

#include "../include/tecs.hpp"

using namespace tecs;
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(DefaultSimulationConfig)

const int n_cells = 100;
const int steps = 300;
const double dt = 0.05;
const Scalar r_max = 1.0;

int main(int argc, char* argv[]) {
  std::string help_str = 
    "mixed: tau_{1,1} = tau_{2,2} = tau_{1,2}"
    "separated: tau_{1,1} = tau_{2,2} > tau_{1,2}"
    "engulfed: tau_{1,1} > tau_{2,2} = tau_{1,2}";

  std::string mode;
  bool ok = Arguments("sorting")
    .add_argument("-m", "--mode", mode, "separated", help_str, "mixed", "separated", "engulfed")
    .parse(argc, argv);
  if (ok == false)
    return 1;

  // Resolve mode to an integer on the host before entering GPU code
  // 0 = mixed, 1 = separated, 2 = engulfed
  int mode_val = 0;
  if (mode == "separated")
    mode_val = 1;
  else if (mode == "engulfed")
    mode_val = 2;

  Simulation<DefaultSimulationConfig> sim(n_cells, "./output/sorting", r_max);
  View<bool> types("types", n_cells);
  auto types_init = INIT_FUNC(
    types(i) = (i < n_cells / 2) ? 0 : 1;
  );
  sim.init_random_filled_sphere(1.0, types_init());
  sim.write_static(types);
  sim.write(0.0);

  auto differential_adhesion = PAIRWISE_FORCE(
    Scalar tau;
    if (mode_val == 1) {
      // separated: tau_{1,1} = tau_{2,2} > tau_{1,2}
      tau = (types(i) == types(j)) ? Scalar(2.0) : Scalar(1.0);
    }
    else if (mode_val == 2) {
      // engulfed: tau_{1,1} > tau_{2,2} = tau_{1,2}
      tau = (types(i) == true || types(j) == true) ? Scalar(2.0) : Scalar(1.0);
    }
    else {
      // mixed: tau_{1,1} = tau_{2,2} = tau_{1,2}
      tau = Scalar(2.0);
    }
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.7f, 1.0f, tau);
  );

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, differential_adhesion());
    sim.write(i * dt);
  }

  return 0;
}
