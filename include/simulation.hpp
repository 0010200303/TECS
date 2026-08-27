#ifndef TECS_SIMULATION_HPP
#define TECS_SIMULATION_HPP

#include <tuple>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "simulation_config.hpp"
#include "utils/runtime_guard.hpp"
#include "utils/utils.hpp"
#include "forces/kernel_fuser.hpp"

#include "types/vector.hpp"
#include "types/link.hpp"

#include "initializers/line.hpp"
#include "initializers/spheres.hpp"
#include "initializers/hexagon.hpp"
#include "initializers/cuboid.hpp"
#include "initializers/rectangle.hpp"
#include "initializers/disk.hpp"
#include "initializers/heart.hpp"

namespace tecs {

  /**
   * @brief The main simulation class that orchestrates agents, forces,
   *        integrators, pair finders and output.
   *
   * This is the central class of the TECS framework. It holds all simulation
   * state (agent positions, velocities, forces, links), manages the integration
   * loop, and provides convenient methods for initialization, stepping, and
   * output.
   *
   * @tparam SimulationConfig A configuration struct that defines the scalar
   *         type, dimensions, fields, integrator, pair finder, etc.
   *
   * **Basic usage:**
   * @code
   *   struct MyConfig : public DefaultSimulationConfig {
   *     CONFIG_FIELDS(
   *       (Vector, position),
   *       (Vector, velocity)
   *     )
   *   };
   *
   *   Simulation<MyConfig> sim(128, "output/run");
   *   sim.init_random_disk(0.5f);
   *
   *   auto my_force = PAIRWISE_FORCE(
   *     ctx.position.delta += forces::Spring(displacement, distance, 0.5f);
   *   );
   * 
   *   for (int step = 0; step < 1000; ++step) {
   *     sim.take_step(0.1, my_force);
   *     sim.write(step * 0.1);
   *   }
   * @endcode
   */
  template<typename SimulationConfig>
  class Simulation {
    EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)
    using Storage = typename detail::FieldStorageFromList<Fields>::type;

    public:
      /**
       * @brief Aggregated settings struct for constructing a Simulation.
       *
       * An alternative to the positional constructor that makes it easier
       * to set advanced options like capacity, link support, and seed.
       */
      struct Settings {
        Settings(const unsigned int agent_count_, const std::string& output_path_)
          : agent_count(agent_count_)
          , output_path(output_path_)
          , capacity(agent_count_) { }

        unsigned int agent_count;
        std::string output_path;

        /// Allocated capacity (can be larger than agent_count).
        unsigned int capacity;

        Scalar cutoff_distance = Scalar(1'000'000);
        uint64_t seed = 2807;

        PairFinder::Settings pair_finder_settings = {};
        Writer::Settings writer_settings = {};

        /// Allocate space for links between agents. 0 = no links.
        unsigned int link_capacity = 0;
        /// Initial active link count (must be <= link_capacity).
        unsigned int link_active_count = 0;
      };

      /**
       * @brief Construct a simulation with positional arguments (legacy constructor).
       *
       * @param agent_count_         Number of agents.
       * @param output_path          Path for output files (no extension).
       * @param cutoff_distance      Neighbour cutoff for pair finding.
       * @param pair_finder_settings Settings for the pair finder.
       * @param writer_settings      Settings for the output writer.
       * @param seed                 Random seed.
       */
      Simulation(
        const unsigned int agent_count_,
        const std::string& output_path,
        const Scalar cutoff_distance = Scalar(1'000'000),
        const PairFinder::Settings& pair_finder_settings = {},
        const Writer::Settings& writer_settings = {},
        const uint64_t seed = 2807
      )
        : agent_count(agent_count_)
        , capacity(agent_count_)
        , storage((get_runtime_guard(), Storage(agent_count_)))
        , random_pool(seed)
        , pair_finder(agent_count_, cutoff_distance, pair_finder_settings)
        , com_fixer()
        , integrator(make_integrator(agent_count_, agent_count_, pair_finder, com_fixer, storage, links))
        , writer(output_path, writer_settings)
        , current_step(0) { }

      /**
       * @brief Construct a simulation from a Settings struct.
       *
       * This variant supports additional options like link capacity.
       */
      Simulation(const Settings settings)
        : agent_count(settings.agent_count)
        , capacity(settings.capacity)
        , storage((get_runtime_guard(), Storage(settings.capacity)))
        , random_pool(settings.seed)
        , pair_finder(settings.agent_count, settings.cutoff_distance, settings.pair_finder_settings)
        , com_fixer()
        , links(settings.link_capacity != 0 ? View<Link>("links", settings.link_capacity) : View<Link>())
        , integrator(make_integrator(
            settings.agent_count,
            settings.capacity,
            pair_finder,
            com_fixer,
            storage,
            links
          ))
        , writer(settings.output_path, settings.writer_settings)
        , current_step(0) 
        {
          links.set_active_count(settings.link_active_count);
        }

    public:
      /// The allocated capacity (may be larger than agent count).
      unsigned int capacity;
      /// Number of active agents.
      unsigned int agent_count;
      /// Storage for all simulation field views.
      Storage storage;

      /// Links between agents.
      View<Link> links;

      /// Kokkos random number pool.
      RandomPool random_pool;

      /// The pair-finding algorithm.
      PairFinder pair_finder;
      /// The centre-of-mass drift fixer.
      ComFixer com_fixer;
      /// The numerical integrator (Euler, Heun, etc.).
      Integrator integrator;

      /// The output writer (HDF5, Dummy, etc.).
      Writer writer;
      /// Step counter (incremented on each write).
      unsigned int current_step;

      /// last time steps delta time.
      double last_dt;

      /// @brief Access the singleton RuntimeGuard (ensures Kokkos is initialized).
      static detail::RuntimeGuard& get_runtime_guard() {
        static detail::RuntimeGuard guard;
        return guard;
      }

      /// @brief Factory: create the integrator from the storage views.
      static Integrator make_integrator(
        unsigned int agent_count_,
        unsigned int capacity_,
        PairFinder& pair_finder_,
        ComFixer& com_fixer_,
        Storage& storage_,
        View<Link>& links_
      ) {
        return std::apply(
          [&](auto&... views) {
            return Integrator(
              agent_count_,
              capacity_,
              pair_finder_,
              com_fixer_,
              links_.view_device(),
              views.view_device()...
            );
          },
          detail::ViewsFromStorage<Fields, Storage>::get(storage_)
        );
      }

      /// @brief Access the integrator.
      Integrator& get_integrator() const {
        return integrator;
      }

      /// @brief Access the pair finder.
      PairFinder& get_pair_finder() const {
        return pair_finder;
      }
    
    public:
      /**
       * @brief Get a specific field view by field type.
       * @code
       *   auto pos = sim.get_view<Field<Vector, position>>();
       * @endcode
       */
      template <typename Field>
      inline auto& get_view() {
        return detail::get<Field>(storage);
      }
      
      /// @copydoc get_view
      template <typename Field>
      inline const auto& get_view() const {
        return detail::get<Field>(storage);
      }

      /// @brief Get a tuple of all field views.
      inline auto get_views() {
        return detail::ViewsFromStorage<Fields, Storage>::get(storage);
      }

      /// @copydoc get_views
      inline auto get_views() const {
        return detail::ViewsFromStorage<Fields, Storage>::get(storage);
      }

      /// @brief Convenience: get the first field view (usually position).
      inline auto get_positions_view() const {
        return std::get<0>(get_views());
      }

      /// @brief Get the allocated capacity.
      inline unsigned int get_capacity() const {
        return capacity;
      }

      /**
       * @brief Resize all field views and integrator buffers.
       * @param value New capacity.
       */
      inline void set_capacity(const unsigned int value) {
        capacity = value;
        integrator.set_capacity(value);

        std::apply([&](auto&... views) {
          ((views.resize(value)), ...);
          integrator.reattach_stage_0(views.view_device()...);
        }, get_views());
      }

      /// @brief Resize simulation views plus additional custom views.
      template<typename... AdditionalViews>
      inline void set_capacity(const unsigned int value, AdditionalViews&... additional_views) {
        set_capacity(value);
        ((additional_views.resize(value)), ...);
      }

      /// @brief Get the number of active agents.
      inline unsigned int get_agent_count() const {
        return agent_count;
      }

      /**
       * @brief Set the number of active agents (without resizing).
       *
       * Updates the integrator, pair finder, and all field view active counts.
       * 
       * @param value New active agent count.
       */
      inline void set_agent_count(const unsigned int value) {
        agent_count = value;
        integrator.set_agent_count(value);
        pair_finder.set_agent_count(value);
        std::apply([&](auto&... views) {
          ((views.set_active_count(value)), ...);
        }, get_views());
      }

      /// @brief Set agent count plus additional custom views.
      template<typename... AdditionalViews>
      inline void set_agent_count(const unsigned int value, AdditionalViews&... additional_views) {
        set_agent_count(value);
        ((additional_views.set_active_count(value)), ...);
      }

      /// @brief Get the old-velocities buffer from the integrator.
      inline auto get_old_velocities_view_from_integrator() const {
        return integrator.old_velocities;
      }

      /// @brief Access the links view.
      inline View<Link>& get_links() {
        return links;
      }

      /// @copydoc get_links
      inline const View<Link>& get_links() const {
        return links;
      }

      /// @brief Get the total link capacity.
      inline unsigned int get_link_count() const {
        return links.extent(0);
      }

      /// @brief Resize the links array.
      inline void resize_links(const unsigned int value) {
        links.resize(value);
      }

    private:
      template<typename... Forces>
      void take_step_impl(double dt, Forces&&... forces) {
        integrator.integrate(dt, random_pool, static_cast<Forces&&>(forces)...);

        std::apply([&](auto&... views) {
          ((views.modify_device()), ...);
        }, get_views());
      }

    public:
      /// @brief Place agents in a line with the given spacing.
      template<typename... InitFuncs>
      inline void init_line(const Scalar distance, InitFuncs&&... init_functions) {
        initializers::Line<SimulationConfig> initializer(get_positions_view(), distance);
        init(initializer, init_functions...);
      }

      /// @brief Place agents randomly on the surface of a sphere.
      template<typename... InitFuncs>
      inline void init_random_hollow_sphere(const Scalar radius, InitFuncs&&... init_functions) {
        initializers::RandomHollowSphere<SimulationConfig> initializer(get_positions_view(), radius);
        init(initializer, init_functions...);
      }

      /// @brief Place agents randomly inside a solid sphere.
      template<typename... InitFuncs>
      inline void init_random_filled_sphere(const Scalar radius, InitFuncs&&... init_functions) {
        initializers::RandomFilledSphere<SimulationConfig> initializer(get_positions_view(), radius);
        init(initializer, init_functions...);
      }

      /// @brief Place agents on a hexagonal lattice.
      template<typename... InitFuncs>
      inline void init_regular_hexagon(const Scalar distance_to_neighbour, InitFuncs&&... init_functions) {
        initializers::RegularHexagon<SimulationConfig> initializer(get_positions_view(), distance_to_neighbour);
        init(initializer, init_functions...);
      }

      /// @brief Place agents randomly inside a cuboid.
      template<typename... InitFuncs>
      inline void init_random_cuboid(const Vector& min, const Vector& max, InitFuncs&&... init_functions) {
        initializers::RandomCuboid<SimulationConfig> initializer(get_positions_view(), min, max);
        init(initializer, init_functions...);
      }

      /// @brief Place agents on a regular rectangular grid.
      template<typename... InitFuncs>
      inline void init_regular_rectangle(
        const Scalar distance_to_neighbour,
        const unsigned int nx,
        InitFuncs&&... init_functions
      ) {
        initializers::RegularRectangle<SimulationConfig> initializer(get_positions_view(), distance_to_neighbour, nx);
        init(initializer, init_functions...);
      }

      /// @brief Place agents randomly inside a 2D disk.
      template<typename... InitFuncs>
      inline void init_random_disk(const Scalar distance_to_neighbour, InitFuncs&&... init_functions) {
        initializers::RandomDisk<SimulationConfig> initializer(get_positions_view(), distance_to_neighbour);
        init(initializer, init_functions...);
      }

      /// @brief Place inside a sphere and relax (spread apart).
      inline void init_relaxed_sphere(
        const Scalar initial_radius,
        const unsigned int relaxation_steps = 2000
      ) {
        initializers::RelaxedSphere<SimulationConfig> initializer(get_positions_view(), initial_radius, relaxation_steps);
        init(initializer);
        initializer.relax(*this);
      }

      inline void init_relaxed_sphere(
        const Scalar initial_radius,
        const int relaxation_steps
      ) {
        initializers::RelaxedSphere<SimulationConfig> initializer(get_positions_view(), initial_radius, static_cast<unsigned int>(relaxation_steps));
        init(initializer);
        initializer.relax(*this);
      }

      /// @copydoc init_relaxed_sphere(Scalar, unsigned int)
      template<typename... InitFuncs>
      inline void init_relaxed_sphere(
        const Scalar initial_radius,
        const unsigned int relaxation_steps = 2000,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_sphere(initial_radius, relaxation_steps);
        init(init_functions...);
      }

      /// @copydoc init_relaxed_sphere(Scalar, unsigned int)
      template<typename... InitFuncs>
      inline void init_relaxed_sphere(
        const Scalar initial_radius,
        const int relaxation_steps,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_sphere(initial_radius, static_cast<unsigned int>(relaxation_steps));
        init(init_functions...);
      }

      /// @copydoc init_relaxed_sphere(Scalar, unsigned int)
      template<typename... InitFuncs>
      inline void init_relaxed_sphere(
        const Scalar initial_radius,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_sphere(initial_radius, 2000u);
        init(init_functions...);
      }

      /// @brief Place inside a cuboid and relax (spread apart).
      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        const unsigned int relaxation_steps = 2000
      ) {
        initializers::RelaxedCuboid<SimulationConfig> initializer(get_positions_view(), min, max, relaxation_steps);
        init(initializer);
        initializer.relax(*this);
      }

      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        const int relaxation_steps
      ) {
        initializers::RelaxedCuboid<SimulationConfig> initializer(get_positions_view(), min, max, static_cast<unsigned int>(relaxation_steps));
        init(initializer);
        initializer.relax(*this);
      }

      /// @copydoc init_relaxed_cuboid(Vector, Vector, unsigned int)
      template<typename... InitFuncs>
      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        const unsigned int relaxation_steps = 2000,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_cuboid(min, max, relaxation_steps);
        init(init_functions...);
      }

      /// @copydoc init_relaxed_cuboid(Vector, Vector, unsigned int)
      template<typename... InitFuncs>
      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        const int relaxation_steps,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_cuboid(min, max, static_cast<unsigned int>(relaxation_steps));
        init(init_functions...);
      }

      /// @copydoc init_relaxed_cuboid(Vector, Vector, unsigned int)
      template<typename... InitFuncs>
      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_cuboid(min, max, 2000u);
        init(init_functions...);
      }

      /// @brief Place agents randomly inside a 3D heart.
      template<typename... InitFuncs>
      inline void init_random_heart(const Scalar scale, InitFuncs&&... init_functions) {
        initializers::RandomHeart<SimulationConfig> initializer(get_positions_view(), scale);
        init(initializer, init_functions...);
      }

      /**
       * @brief Run an initializer functor in parallel over all agents.
       *
       * The functor is called as `initializer(i, generator)` for each agent.
       */
      template<typename Initializer>
      inline void init(Initializer initializer) {
        auto& random_pool_ = random_pool;
        Kokkos::parallel_for("Simulation::init", agent_count, KOKKOS_LAMBDA(const unsigned int i) {
          Random generator = random_pool_.get_state();

          initializer(i, generator);

          random_pool_.free_state(generator);
        });
      }

      /// @brief Run multiple initializers in sequence.
      template<typename Initializer, typename... InitFuncs>
      inline void init(Initializer initializer, InitFuncs&&... init_functions) {
        init(initializer);
        if constexpr (sizeof...(init_functions) > 0)
          init(static_cast<InitFuncs&&>(init_functions)...);
      }

      /**
       * @brief Advance the simulation by one time step.
       *
       * Evaluates all provided forces, fuses them by tag so that forces of the
       * same type share a single kernel launch, then passes them to the integrator.
       *
       * @param dt     Time step size.
       * @param forces Force functors (GENERIC_FORCE, PAIRWISE_FORCE, LINK_FORCE, etc.).
       */
      template<typename... Forces>
      inline void take_step(double dt, Forces&&... forces) {
        auto fused_forces = detail::fuse_forces(static_cast<Forces&&>(forces)...);

        std::apply([&](auto&&... args) {
          take_step_impl(dt, static_cast<decltype(args)&&>(args)...);
        }, fused_forces);
        last_dt = dt;
      }

      /**
       * @brief Run a custom kernel over a given number of iterations.
       *
       * Each call gets a random generator state.
       *
       * @param count    Number of iterations.
       * @param function Functor called as `function(i, generator)`.
       */
      template<typename Func>
      inline void run_custom(unsigned int count, Func function) {
        auto& random_pool_ = random_pool;
        Kokkos::parallel_for("Simulation::run", count, KOKKOS_LAMBDA(const unsigned int i) {
          Random generator = random_pool_.get_state();

          function(i, generator);

          random_pool_.free_state(generator);
        });
      }

      /// @brief Run multiple custom kernels, fused togther over a user defined count.
      template<typename... Funcs>
      inline void run_custom(unsigned int count, Funcs&&... functions) {
        auto fused_funcs = detail::fuse_forces(static_cast<Funcs&&>(functions)...);

        std::apply([&](auto&&... args) {
          run_custom(count, static_cast<decltype(args)&&>(args)...);
        }, fused_funcs);
      }

      /// @brief Run a kernel over all agents (same as `run_custom(agent_count, ...)`).
      template<typename... Funcs>
      inline void run(Funcs&&... functions) {
        run_custom(agent_count, std::forward<Funcs>(functions)...);
      }

      /// @brief Run a kernel over all active links.
      template<typename... Funcs>
      inline void run_links(Funcs&&... functions) {
        run_custom(links.get_active_count(), std::forward<Funcs>(functions)...);
      }

      /**
       * @brief Write the current simulation state to disk.
       *
       * Increments the internal step counter.
       *
       * @param time Simulation time for this output frame.
       */
      inline void write(const double time) {
        std::apply([&](auto&&... args) {
          writer.write(time, current_step++, static_cast<decltype(args)&&>(args)..., links);
        }, get_views());
      }

      /**
       * @brief Write the current state plus additional custom views to disk.
       *
       * Increments the internal step counter.
       * 
       * @param time              Simulation time.
       * @param additional_views  Extra views to include in the output.
       */
      template<typename... Views>
      inline void write(const double time, Views&&... additional_views) {
        std::apply([&](auto&&... args) {
          writer.write(
            time,
            current_step++,
            static_cast<decltype(args)&&>(args)...,
            std::forward<Views>(additional_views)...,
            links
          );
        }, get_views());
      }

      /**
       * @brief Write the current simulation state to disk.
       *
       * Uses the internally tracked time (`last_dt * current_step`).
       * Increments the internal step counter.
       */
      inline void write() {
        write(last_dt * current_step);
      }

      /**
       * @brief Write the current simulation state plus additional custom views to disk.
       *
       * Uses the internally tracked time (`last_dt * current_step`).
       * Increments the internal step counter.
       * 
       * @param additional_views  Extra views to include in the output.
       */
      template<typename... Views>
      inline void write(Views&&... additional_views) {
        write(last_dt * current_step, std::forward<Views>(additional_views)...);
      }

      /**
       * @brief Write static (time-independent) data once.
       *
       * Should be called before the first `write()` call.
       *
       * @param static_views Views with data that does not change over time.
       */
      template<typename... Views>
      inline void write_static(Views&&... static_views) {
        writer.write_static(static_views...);
      } 
  };
} // namespace tecs

#endif // TECS_SIMULATION_HPP
