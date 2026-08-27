#include <Kokkos_Core.hpp>
#include "../../include/tecs.hpp"

// testing the performance impact of splitting forces across multiple kernels and 
// auto fusing them vs. user defined fusion vs. just evaluating split kernels

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_WRITER(io::Dummy)
  CONFIG_INTEGRATOR(integrators::Euler)
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

enum class BenchmarkType { ControlKernel, SplitKernel, UserFusedKernel, AutoFusedKernel };

inline const char* benchmark_name(BenchmarkType b) {
  switch (b) {
    case BenchmarkType::ControlKernel:
      return "ControlKernel";
    case BenchmarkType::SplitKernel:
      return "SplitKernel";
    case BenchmarkType::UserFusedKernel:
      return "UserFusedKernel";
    case BenchmarkType::AutoFusedKernel:
      return "AutoFusedKernel";
  }
  return "UNKNOWN";
}

template<typename Force>
double benchmark_kernel(
  Force kernel,
  const int steps,
  const float dt,
  int n_agents,
  double& checksum_out
) {
  auto checksum = [&](const VectorView& positions) {
    auto host_pos = Kokkos::create_mirror_view(positions.view_device());
    Kokkos::deep_copy(host_pos, positions.view_device());
    Kokkos::fence();
    double sum = 0.0;
    for (unsigned int i = 0; i < host_pos.extent(0); ++i)
      sum += host_pos(i).x() + host_pos(i).y() + host_pos(i).z();
    return sum;
  };

  Simulation<SimulationConfig> sim(n_agents, "");
  auto& positions = sim.get_view<FIELD(Vector, position)>();

  initializers::RandomHollowSphere<SimulationConfig> init(positions, 32.0);
  sim.init(init);

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int i = 0; i < steps; ++i) {
    sim.take_step(dt, kernel());
  }

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

template<typename ForceX, typename ForceY, typename ForceZ>
double benchmark_split_kernel(
  ForceX kernel_x,
  ForceY kernel_y,
  ForceZ kernel_z,
  const int steps,
  const float dt,
  int n_agents,
  double& checksum_out
) {
  auto checksum = [&](const VectorView& positions) {
    auto host_pos = Kokkos::create_mirror_view(positions.view_device());
    Kokkos::deep_copy(host_pos, positions.view_device());
    Kokkos::fence();
    double sum = 0.0;
    for (unsigned int i = 0; i < host_pos.extent(0); ++i)
      sum += host_pos(i).x() + host_pos(i).y() + host_pos(i).z();
    return sum;
  };

  Simulation<SimulationConfig> sim(n_agents, "");
  auto& positions = sim.get_view<FIELD(Vector, position)>();

  initializers::RandomHollowSphere<SimulationConfig> init(positions, 16.0);
  sim.init(init);

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int i = 0; i < steps; ++i) {
    sim.integrator.evaluate_forces(
      true, sim.random_pool,
      sim.integrator.stage_pack[0],
      sim.integrator.stage_pack[1],
      kernel_x());
    sim.integrator.evaluate_forces(
      true, sim.random_pool,
      sim.integrator.stage_pack[0],
      sim.integrator.stage_pack[1],
      kernel_y());
    sim.integrator.evaluate_forces(
      true, sim.random_pool,
      sim.integrator.stage_pack[0],
      sim.integrator.stage_pack[1],
      kernel_z());
    sim.integrator.apply_euler(dt);
  }

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

template<typename ForceX, typename ForceY, typename ForceZ>
double benchmark_fusion_kernel(
  ForceX kernel_x,
  ForceY kernel_y,
  ForceZ kernel_z,
  const int steps,
  const float dt,
  int n_agents,
  double& checksum_out
) {
  auto checksum = [&](const VectorView& positions) {
    auto host_pos = Kokkos::create_mirror_view(positions.view_device());
    Kokkos::deep_copy(host_pos, positions.view_device());
    Kokkos::fence();
    double sum = 0.0;
    for (unsigned int i = 0; i < host_pos.extent(0); ++i)
      sum += host_pos(i).x() + host_pos(i).y() + host_pos(i).z();
    return sum;
  };

  Simulation<SimulationConfig> sim(n_agents, "");
  auto& positions = sim.get_view<FIELD(Vector, position)>();

  initializers::RandomHollowSphere<SimulationConfig> init(positions, 16.0);
  sim.init(init);

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int i = 0; i < steps; ++i) {
    sim.take_step(dt, kernel_x(), kernel_y(), kernel_z());
  }

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

void run_benchmark_case(int n_agents, int n_steps, int n_reps, float dt_in, BenchmarkType bench) {
  const float stiffness = 0.1f;
  auto control_kernel = GENERIC_FORCE(
    ctx.position.delta += -stiffness * ctx.position.self;
  );

  auto split_kernel_x = GENERIC_FORCE(
    ctx.position.delta.x() += -stiffness * ctx.position.self.x();
  );

  auto split_kernel_y = GENERIC_FORCE(
    ctx.position.delta.y() += -stiffness * ctx.position.self.y();
  );

  auto split_kernel_z = GENERIC_FORCE(
    ctx.position.delta.z() += -stiffness * ctx.position.self.z();
  );

  auto user_fused_kernel = GENERIC_FORCE(
    ctx.position.delta.x() += -stiffness * ctx.position.self.x();
    ctx.position.delta.y() += -stiffness * ctx.position.self.y();
    ctx.position.delta.z() += -stiffness * ctx.position.self.z();
  );

  double checksum = 0.0;
  double total_time = 0.0;
  for (int i = 0; i < n_reps; ++i) {
    if (bench == BenchmarkType::ControlKernel) {
      total_time += benchmark_kernel(control_kernel, n_steps, dt_in, n_agents, checksum);
    } else if (bench == BenchmarkType::SplitKernel) {
      total_time += benchmark_split_kernel(split_kernel_x, split_kernel_y, split_kernel_z, n_steps, dt_in, n_agents, checksum);
    } else if (bench == BenchmarkType::UserFusedKernel) {
      total_time += benchmark_kernel(user_fused_kernel, n_steps, dt_in, n_agents, checksum);
    } else if (bench == BenchmarkType::AutoFusedKernel) {
      total_time += benchmark_fusion_kernel(split_kernel_x, split_kernel_y, split_kernel_z, n_steps, dt_in, n_agents, checksum);
    }
  }
  double avg = total_time / static_cast<double>(n_reps);
  double time_per_step_ms = (avg / static_cast<double>(n_steps)) * 1e3;

  std::cout << benchmark_name(bench) << "," << n_agents << ","
            << n_steps << "," << n_reps << ","
            << std::setprecision(10) << dt_in << ","
            << std::setprecision(10) << time_per_step_ms << ","
            << std::setprecision(10) << checksum << "\n";
}

int main() {
  const std::vector<int> agent_counts = {362144};
  const int steps = 100;
  const int repetitions = 10;
  const float dt = 0.000001;

  Kokkos::initialize();
  {
    Kokkos::print_configuration(std::cout);

    std::cout << "benchmark,agents,steps,repetitions,dt,time_per_step_ms,checksum\n";

    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::ControlKernel);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::SplitKernel);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::UserFusedKernel);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::AutoFusedKernel);
  }
  Kokkos::finalize();

  return 0;
}
