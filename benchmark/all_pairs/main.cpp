// benchmark using a simple pairwise force to test the 
// raw performance of various all pairs pair finders
// for matching checksums use a serial backend

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <csignal>
#include <sys/wait.h>
#include <fcntl.h>

#include "../../include/tecs.hpp"
#include "naive_all_pairs_parallel_reduce.hpp"
#include "naive_all_pairs_parallel_reduce_double.hpp"
#include "naive_all_pairs_parallel_reduce_symmetric.hpp"
#include "naive_all_pairs_for.hpp"
#include "naive_all_pairs_for_double.hpp"
#include "naive_all_pairs_for_symmetric.hpp"
#include "naive_all_pairs_spread.hpp"
#include "binned_all_pairs_reduce_for.hpp"
#include "binned_all_pairs_reduce_for_double.hpp"
#include "binned_all_pairs_reduce_for_symmetric.hpp"
#include "binned_all_pairs_reduce_parallel.hpp"
#include "binned_all_pairs_reduce_parallel_double.hpp"
#include "binned_all_pairs_reduce_parallel_symmetric.hpp"

using namespace tecs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_WRITER(io::Dummy)
  CONFIG_INTEGRATOR(integrators::Euler)
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

#define BENCHMARK_VARIANTS(M)                                                   \
  M(NaiveParallelReduce,                  NaiveAllPairsParallelReduce)           \
  M(NaiveFor,                             NaiveAllPairsFor)                      \
  M(NaiveSpread,                          NaiveAllPairsSpread)                   \
  M(NaiveForDouble,                       NaiveAllPairsForDouble)                \
  M(NaiveParallelReduceDouble,            NaiveAllPairsParallelReduceDouble)     \
  M(NaiveForSymmetric,                    NaiveAllPairsForSymmetric)             \
  M(NaiveParallelReduceSymmetric,         NaiveAllPairsParallelReduceSymmetric)  \
  M(BinnedReduceFor,                      BinnedAllPairsReduceFor)               \
  M(BinnedReduceForDouble,                BinnedAllPairsReduceForDouble)         \
  M(BinnedReduceForSymmetric,             BinnedAllPairsReduceForSymmetric)      \
  M(BinnedReduceParallel,                 BinnedAllPairsReduceParallel)          \
  M(BinnedReduceParallelDouble,           BinnedAllPairsReduceParallelDouble)    \
  M(BinnedReduceParallelSymmetric,        BinnedAllPairsReduceParallelSymmetric)

#define DEFINE_CONFIG(EnumName, PairFinder)                                     \
  struct EnumName##Config : public SimulationConfig {                           \
    CONFIG_PAIR_FINDER(benchmark::PairFinder)                                   \
  };
BENCHMARK_VARIANTS(DEFINE_CONFIG)
#undef DEFINE_CONFIG

static constexpr int CHILD_TIMEOUT_S = 120;

static bool run_benchmark_child(
  int n_agents, int n_steps, int n_reps,
  const char* bench_name
) {
  std::string exe = []() {
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) buf[len] = '\0';
    return std::string(buf);
  }();

  std::string agents_s = std::to_string(n_agents);
  std::string steps_s  = std::to_string(n_steps);
  std::string reps_s   = std::to_string(n_reps);

  const char* argv[] = {
    exe.c_str(), "--child-bench", bench_name,
    agents_s.c_str(), steps_s.c_str(), reps_s.c_str(), nullptr
  };

  int pipefd[2];
  if (pipe(pipefd) != 0) {
    std::perror("pipe");
    return false;
  }

  pid_t pid = fork();
  if (pid == -1) { std::perror("fork"); close(pipefd[0]); close(pipefd[1]); return false; }

  if (pid == 0) {
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);
    execvp(argv[0], const_cast<char* const*>(argv));
    std::perror("execvp");
    _exit(127);
  }

  close(pipefd[1]);

  std::string child_out;
  child_out.reserve(256);
  {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(CHILD_TIMEOUT_S);
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);
    char buf[128];
    while (true) {
      if (std::chrono::steady_clock::now() >= deadline) {
        kill(pid, SIGKILL);
        close(pipefd[0]);
        int ws;
        waitpid(pid, &ws, 0);
        // std::cerr << "TIMEOUT: " << bench_name << " n=" << n_agents << "\n";
        return false;
      }
      ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
      if (n > 0) { buf[n] = '\0'; child_out += buf; }
      else if (n == 0) break;
      else if (errno != EAGAIN && errno != EWOULDBLOCK) break;
      else std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  close(pipefd[0]);

  int wstatus;
  waitpid(pid, &wstatus, 0);
  if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0)
    return false;

  std::cout << child_out;
  std::cout.flush();
  return true;
}

#define ENUM_VALUE(EnumName, _) EnumName,
enum class BenchmarkType {
  BENCHMARK_VARIANTS(ENUM_VALUE)
  ENUM_COUNT
};
#undef ENUM_VALUE

inline const char* benchmark_name(BenchmarkType b) {
  switch (b) {
    #define NAME_CASE(EnumName, _) case BenchmarkType::EnumName: return #EnumName;
    BENCHMARK_VARIANTS(NAME_CASE)
    #undef NAME_CASE
  }
  return "UNKNOWN";
}

template<typename Config, typename Force>
static double run_one_benchmark(
  Force kernel, int n_agents, int n_steps, double& checksum_out
) {
  auto checksum_fn = [&](const auto& positions) {
    auto host_pos = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), positions.view_device());
    double s = 0.0;
    for (int i = 0; i < host_pos.extent(0); ++i)
      s += host_pos(i).x() + host_pos(i).y() + host_pos(i).z();
    return s;
  };

  const Scalar r_max = 1.0f;
  Simulation<Config> sim(n_agents, "", r_max);
  sim.init_random_filled_sphere(16.0f);
  auto& positions = sim.template get_view<FIELD(Vector, position)>();

  Kokkos::fence();
  Kokkos::Timer timer;
  for (int s = 0; s < n_steps; ++s)
    sim.take_step(0.0f, kernel);
  Kokkos::fence();

  double t = timer.seconds();
  checksum_out = checksum_fn(positions);
  return t;
}

template<typename Force>
static double dispatch_benchmark(
  BenchmarkType bench, Force kernel,
  int n_agents, int n_steps, double& checksum
) {
  switch (bench) {
    #define DISPATCH_CASE(EnumName, _) \
      case BenchmarkType::EnumName: \
        return run_one_benchmark<EnumName##Config>(kernel, n_agents, n_steps, checksum);
    BENCHMARK_VARIANTS(DISPATCH_CASE)
    #undef DISPATCH_CASE
  }
  return -1.0;
}

static int child_main(int argc, char** argv) {
  // argv: --child-bench <name> <agents> <steps> <reps>
  if (argc < 6)
    return 1;

  const char* bench_name = argv[2];
  int  n_agents          = std::atoi(argv[3]);
  int  n_steps           = std::atoi(argv[4]);
  int  n_reps            = std::atoi(argv[5]);

  auto find_type = [&]() -> BenchmarkType {
    for (int b = 0; b < static_cast<int>(BenchmarkType::ENUM_COUNT); ++b) {
      auto bt = static_cast<BenchmarkType>(b);
      if (std::strcmp(benchmark_name(bt), bench_name) == 0)
        return bt;
    }
    std::cerr << "Unknown bench: " << bench_name << "\n";
    std::exit(1);
  };
  BenchmarkType bench = find_type();

  Kokkos::initialize();
  {
    auto kernel = PAIRWISE_FORCE(
      ctx.position.delta += forces::Spring(displacement, distance, 0.5f);
    );

    double checksum = 0.0;
    double total_time = 0.0;
    int completed = 0;

    for (int i = 0; i < n_reps; ++i) {
      double t = dispatch_benchmark(bench, kernel, n_agents, n_steps, checksum);
      if (t < 0.0) break;
      total_time += t;
      ++completed;
    }

    if (completed > 0) {
      double avg = total_time / completed;
      double tps_ms = (avg / n_steps) * 1e3;
      std::cout << bench_name << "," << n_agents << ","
                << n_steps << "," << completed << ","
                << std::setprecision(10) << tps_ms << ","
                << std::setprecision(10) << checksum << "\n";
    }
  }
  Kokkos::finalize();
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 3 && std::strcmp(argv[1], "--child-bench") == 0)
    return child_main(argc, argv);

  const int steps = 100;
  const int repetitions = 10;

  std::cout << "benchmark,agents,steps,repetitions,time_per_step_ms,checksum\n";

  constexpr int max_agents = 67108864;
  constexpr int start_agents = 32;
  constexpr double scale_factor = 2.0;

  constexpr const char* bench_names[] = {
    #define NAME_ENTRY(EnumName, _) #EnumName,
    BENCHMARK_VARIANTS(NAME_ENTRY)
    #undef NAME_ENTRY
  };
  constexpr int n_benches = sizeof(bench_names) / sizeof(bench_names[0]);
  bool active[n_benches];
  for (int b = 0; b < n_benches; ++b) active[b] = true;

  int n_agents = start_agents;
  while (n_agents <= max_agents) {
    bool any_active = false;
    for (int b = 0; b < n_benches; ++b) {
      if (!active[b])
        continue;

      any_active = true;
      if (!run_benchmark_child(n_agents, steps, repetitions, bench_names[b]))
        active[b] = false;
    }
    if (!any_active)
      break;
    n_agents = static_cast<int>(static_cast<double>(n_agents) * scale_factor);
  }

  return 0;
}
