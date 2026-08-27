#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <Kokkos_Core.hpp>
#include "../../include/vector.hpp"

// benchmark settings
using type = float;
const type dt = 0.0078125;

using KocsVector = tecs::Vector3<type>;

#pragma region Vector
template<typename Scalar>
struct Vector3 {
  Scalar x, y, z;

  KOKKOS_INLINE_FUNCTION
  Vector3() : x(Scalar(0)), y(Scalar(0)), z(Scalar(0)) { }

  KOKKOS_INLINE_FUNCTION
  Vector3(Scalar x_, Scalar y_, Scalar z_) : x(x_), y(y_), z(z_) { }

  // arithmetic operators
  KOKKOS_INLINE_FUNCTION
  Vector3 operator+(const Vector3& rhs) const {
    return Vector3(x + rhs.x, y + rhs.y, z + rhs.z);
  }

  KOKKOS_INLINE_FUNCTION
  Vector3 operator-(const Vector3& rhs) const {
    return Vector3(x - rhs.x, y - rhs.y, z - rhs.z);
  }

  KOKKOS_INLINE_FUNCTION
  Vector3 operator*(Scalar s) const {
    return Vector3(x * s, y * s, z * s);
  }

  KOKKOS_INLINE_FUNCTION
  Vector3 operator/(Scalar s) const {
    return Vector3(x / s, y / s, z / s);
  }

  KOKKOS_INLINE_FUNCTION
  Vector3& operator+=(const Vector3& rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Vector3& operator-=(const Vector3& rhs) {
    x -= rhs.x; y -= rhs.y; z -= rhs.z;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Vector3& operator*=(Scalar s) {
    x *= s; y *= s; z *= s;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Vector3& operator/=(Scalar s) {
    x /= s; y /= s; z /= s;
    return *this;
  }

  // dot product
  KOKKOS_INLINE_FUNCTION
  Scalar dot(const Vector3& rhs) const {
    return x * rhs.x + y * rhs.y + z * rhs.z;
  }

  // cross product
  KOKKOS_INLINE_FUNCTION
  Vector3 cross(const Vector3& rhs) const {
    return Vector3(
      y * rhs.z - z * rhs.y,
      z * rhs.x - x * rhs.z,
      x * rhs.y - y * rhs.x
    );
  }

  // norms
  KOKKOS_INLINE_FUNCTION
  Scalar norm_squared() const {
    return dot(*this);
  }

  KOKKOS_INLINE_FUNCTION
  Scalar norm() const {
    return std::sqrt(norm_squared());
  }

  KOKKOS_INLINE_FUNCTION
  Vector3 normalized() const {
    Scalar n = norm();
    return (n > Scalar(0)) ? (*this) / n : *this;
  }
};

// scalar x vector operations
template<typename Scalar>
KOKKOS_INLINE_FUNCTION
Vector3<Scalar> operator*(Scalar s, const Vector3<Scalar>& v) {
  return v * s;
}

template<typename Scalar>
KOKKOS_INLINE_FUNCTION
Vector3<Scalar> operator/(Scalar s, const Vector3<Scalar>& v) {
  return v / s;
}
#pragma endregion

using Vector = Vector3<type>;
using ViewOfVectors = Kokkos::View<Vector*>;
using ViewOfArrays = Kokkos::View<type*[3]>;
using ViewOfScalars = Kokkos::View<type*>;
using ViewOfKocsVectors = Kokkos::View<KocsVector*>;

enum class BenchmarkType { ViewOfVectors, ViewOfArrays, ViewOfArraysRaw, ViewOfScalars, KocsVector };

inline const char* benchmark_name(BenchmarkType b) {
  switch (b) {
    case BenchmarkType::ViewOfVectors:
      return "ViewOfVectors";
    case BenchmarkType::ViewOfArrays:
      return "ViewOfArrays";
    case BenchmarkType::ViewOfArraysRaw:
      return "ViewOfArraysRaw";
    case BenchmarkType::ViewOfScalars:
      return "ViewOfScalars";
    case BenchmarkType::KocsVector:
      return "KocsVector";
  }
  return "UNKNOWN";
}

template<typename ForceFn>
double benchmark_view_of_vectors(
  const std::vector<Vector>& host_pos,
  const std::vector<Vector>& host_vel,
  const int steps,
  const type dt,
  double& checksum_out,
  int n_agents,
  ForceFn&& pairwise_force
) {
  auto checksum = [&](const ViewOfVectors& positions) {
    auto host_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), positions);
    double sum = 0.0f;
    for (int i = 0; i < host_pos.extent(0); ++i)
      sum += host_pos(i).x + host_pos(i).y + host_pos(i).z;
    return sum;
  };

  auto positions = ViewOfVectors("pos", n_agents);
  auto velocities = ViewOfVectors("vel", n_agents);
  auto forces = ViewOfVectors("for", n_agents);

  auto host_pos_mirror = Kokkos::create_mirror_view(positions);
  auto host_vel_mirror = Kokkos::create_mirror_view(velocities);

  for (int i = 0; i < host_pos.size(); ++i) {
    host_pos_mirror(i) = host_pos[i];
    host_vel_mirror(i) = host_vel[i];
  }

  Kokkos::deep_copy(positions, host_pos_mirror);
  Kokkos::deep_copy(velocities, host_vel_mirror);

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int step = 0; step < steps; ++step) {
    // zero forces
    Kokkos::parallel_for("zero_forces", n_agents, KOKKOS_LAMBDA(const int i) {
      forces(i) = Vector(0,0,0);
    });

    // compute pairwise forces
    Kokkos::parallel_for(
      "compute_forces",
      Kokkos::TeamPolicy<>(n_agents, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int i = team.league_rank();

        const Vector pi = positions(i);
        Vector fi = Vector(type(0), type(0), type(0));

        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, n_agents),
          [&](const int j, Vector& local) {
            if (j == i)
              return;

            const Vector pj = positions(j);
            pairwise_force(pi, pj, local);
          },
          fi
        );

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          forces(i) = fi;
        });
      }
    );

    // integrate (explicit Euler)
    Kokkos::parallel_for("euler_update", n_agents, KOKKOS_LAMBDA(const int i) {
      Vector v = velocities(i);
      Vector p = positions(i);
      const Vector f = forces(i);

      v.x += f.x * dt;
      v.y += f.y * dt;
      v.z += f.z * dt;

      p.x += v.x * dt;
      p.y += v.y * dt;
      p.z += v.z * dt;

      velocities(i) = v;
      positions(i) = p;
    });
  }

  Kokkos::fence();
  double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

template<typename ForceFn>
double benchmark_view_of_arrays(
  const std::vector<Vector>& host_pos,
  const std::vector<Vector>& host_vel,
  const int steps,
  const type dt,
  double& checksum_out,
  int n_agents,
  ForceFn&& pairwise_force
) {
  auto checksum = [&](const ViewOfArrays& positions) {
    auto host_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), positions);
    double sum = 0.0f;
    for (int i = 0; i < host_pos.extent(0); ++i)
      sum += host_pos(i, 0) + host_pos(i, 1) + host_pos(i, 2);
    return sum;
  };

  auto positions = ViewOfArrays("pos", n_agents);
  auto velocities = ViewOfArrays("vel", n_agents);
  auto forces = ViewOfArrays("for", n_agents);

  auto host_pos_mirror = Kokkos::create_mirror_view(positions);
  auto host_vel_mirror = Kokkos::create_mirror_view(velocities);

  for (int i = 0; i < host_pos.size(); ++i) {
    host_pos_mirror(i, 0) = host_pos[i].x;
    host_pos_mirror(i, 1) = host_pos[i].y;
    host_pos_mirror(i, 2) = host_pos[i].z;
    host_vel_mirror(i, 0) = host_vel[i].x;
    host_vel_mirror(i, 1) = host_vel[i].y;
    host_vel_mirror(i, 2) = host_vel[i].z;
  }

  Kokkos::deep_copy(positions, host_pos_mirror);
  Kokkos::deep_copy(velocities, host_vel_mirror);

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int step = 0; step < steps; ++step) {
    // zero forces
    Kokkos::parallel_for("zero_forces", n_agents, KOKKOS_LAMBDA(const int i) {
      forces(i, 0) = type(0);
      forces(i, 1) = type(0);
      forces(i, 2) = type(0);
    });

    // compute pairwise forces
    Kokkos::parallel_for(
      "compute_forces",
      Kokkos::TeamPolicy<>(n_agents, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int i = team.league_rank();

        const Vector pi(
          positions(i, 0),
          positions(i, 1),
          positions(i, 2)
        );
        Vector fi(type(0), type(0), type(0));

        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, n_agents),
          [&](const int j, Vector& local) {
            if (j == i)
              return;

            const Vector pj(
              positions(j, 0),
              positions(j, 1),
              positions(j, 2)
            );
            pairwise_force(pi, pj, local);
          },
          fi
        );

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          forces(i, 0) = fi.x;
          forces(i, 1) = fi.y;
          forces(i, 2) = fi.z;
        });
      }
    );

    // integrate (explicit Euler)
    Kokkos::parallel_for("euler_update", n_agents, KOKKOS_LAMBDA(const int i) {
      velocities(i, 0) += forces(i, 0) * dt;
      velocities(i, 1) += forces(i, 1) * dt;
      velocities(i, 2) += forces(i, 2) * dt;
      positions(i, 0) += velocities(i, 0) * dt;
      positions(i, 1) += velocities(i, 1) * dt;
      positions(i, 2) += velocities(i, 2) * dt;
    });
  }

  Kokkos::fence();
  double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

template<typename ForceFn>
double benchmark_view_of_arrays_raw(
  const std::vector<Vector>& host_pos,
  const std::vector<Vector>& host_vel,
  const int steps,
  const type dt,
  double& checksum_out,
  int n_agents,
  ForceFn&& pairwise_force
) {
  auto checksum = [&](const ViewOfArrays& positions) {
    auto host_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), positions);
    double sum = 0.0f;
    for (int i = 0; i < host_pos.extent(0); ++i)
      sum += host_pos(i, 0) + host_pos(i, 1) + host_pos(i, 2);
    return sum;
  };

  auto positions = ViewOfArrays("pos", n_agents);
  auto velocities = ViewOfArrays("vel", n_agents);
  auto forces = ViewOfArrays("for", n_agents);

  auto host_pos_mirror = Kokkos::create_mirror_view(positions);
  auto host_vel_mirror = Kokkos::create_mirror_view(velocities);

  for (int i = 0; i < host_pos.size(); ++i) {
    host_pos_mirror(i, 0) = host_pos[i].x;
    host_pos_mirror(i, 1) = host_pos[i].y;
    host_pos_mirror(i, 2) = host_pos[i].z;
    host_vel_mirror(i, 0) = host_vel[i].x;
    host_vel_mirror(i, 1) = host_vel[i].y;
    host_vel_mirror(i, 2) = host_vel[i].z;
  }

  Kokkos::deep_copy(positions, host_pos_mirror);
  Kokkos::deep_copy(velocities, host_vel_mirror);

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int step = 0; step < steps; ++step) {
    // zero forces
    Kokkos::parallel_for("zero_forces", n_agents, KOKKOS_LAMBDA(const int i) {
      forces(i, 0) = type(0);
      forces(i, 1) = type(0);
      forces(i, 2) = type(0);
    });

    // compute pairwise forces
    Kokkos::parallel_for(
      "compute_forces",
      Kokkos::TeamPolicy<>(n_agents, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int i = team.league_rank();

        const type pix = positions(i, 0);
        const type piy = positions(i, 1);
        const type piz = positions(i, 2);

        type fix = type(0);
        type fiy = type(0);
        type fiz = type(0);

        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, n_agents),
          [&](const int j, type& local_x, type& local_y, type& local_z) {
            if (j == i)
              return;

            pairwise_force(
              pix, piy, piz,
              positions(j, 0), positions(j, 1), positions(j, 2),
              local_x, local_y, local_z
            );
          },
          fix, fiy, fiz
        );

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          forces(i, 0) = fix;
          forces(i, 1) = fiy;
          forces(i, 2) = fiz;
        });
      }
    );

    // integrate (explicit Euler)
    Kokkos::parallel_for("euler_update", n_agents, KOKKOS_LAMBDA(const int i) {
      velocities(i, 0) += forces(i, 0) * dt;
      velocities(i, 1) += forces(i, 1) * dt;
      velocities(i, 2) += forces(i, 2) * dt;
      positions(i, 0) += velocities(i, 0) * dt;
      positions(i, 1) += velocities(i, 1) * dt;
      positions(i, 2) += velocities(i, 2) * dt;
    });
  }

  Kokkos::fence();
  double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

template<typename ForceFn>
double benchmark_view_of_scalars(
  const std::vector<Vector>& host_pos,
  const std::vector<Vector>& host_vel,
  const int steps,
  const type dt,
  double& checksum_out,
  int n_agents,
  ForceFn&& pairwise_force
) {
  auto checksum = [&](const ViewOfScalars& pos_x,
                      const ViewOfScalars& pos_y,
                      const ViewOfScalars& pos_z) {
    auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pos_x);
    auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pos_y);
    auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pos_z);
    double sum = 0.0;
    for (int i = 0; i < hx.extent(0); ++i)
      sum += hx(i) + hy(i) + hz(i);
    return sum;
  };

  ViewOfScalars pos_x("pos_x", n_agents), pos_y("pos_y", n_agents), pos_z("pos_z", n_agents);
  ViewOfScalars vel_x("vel_x", n_agents), vel_y("vel_y", n_agents), vel_z("vel_z", n_agents);
  ViewOfScalars force_x("force_x", n_agents), force_y("force_y", n_agents), force_z("force_z", n_agents);

  auto host_pos_x = Kokkos::create_mirror_view(pos_x);
  auto host_pos_y = Kokkos::create_mirror_view(pos_y);
  auto host_pos_z = Kokkos::create_mirror_view(pos_z);
  auto host_vel_x = Kokkos::create_mirror_view(vel_x);
  auto host_vel_y = Kokkos::create_mirror_view(vel_y);
  auto host_vel_z = Kokkos::create_mirror_view(vel_z);

  for (int i = 0; i < host_pos.size(); ++i) {
    host_pos_x(i) = host_pos[i].x;
    host_pos_y(i) = host_pos[i].y;
    host_pos_z(i) = host_pos[i].z;
    host_vel_x(i) = host_vel[i].x;
    host_vel_y(i) = host_vel[i].y;
    host_vel_z(i) = host_vel[i].z;
  }

  Kokkos::deep_copy(pos_x, host_pos_x);
  Kokkos::deep_copy(pos_y, host_pos_y);
  Kokkos::deep_copy(pos_z, host_pos_z);
  Kokkos::deep_copy(vel_x, host_vel_x);
  Kokkos::deep_copy(vel_y, host_vel_y);
  Kokkos::deep_copy(vel_z, host_vel_z);

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int step = 0; step < steps; ++step) {
    // zero forces
    Kokkos::parallel_for("zero_forces", n_agents, KOKKOS_LAMBDA(const int i) {
      force_x(i) = type(0);
      force_y(i) = type(0);
      force_z(i) = type(0);
    });

    // compute pairwise forces
    Kokkos::parallel_for(
      "compute_forces",
      Kokkos::TeamPolicy<>(n_agents, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int i = team.league_rank();

        const type pix = pos_x(i);
        const type piy = pos_y(i);
        const type piz = pos_z(i);

        type fix = type(0);
        type fiy = type(0);
        type fiz = type(0);

        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, n_agents),
          [&](const int j, type& local_x, type& local_y, type& local_z) {
            if (j == i)
              return;

            pairwise_force(
              pix, piy, piz,
              pos_x(j), pos_y(j), pos_z(j),
              local_x, local_y, local_z
            );
          },
          fix, fiy, fiz
        );

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          force_x(i) = fix;
          force_y(i) = fiy;
          force_z(i) = fiz;
        });
      }
    );

    // integrate (explicit Euler)
    Kokkos::parallel_for("euler_update", n_agents, KOKKOS_LAMBDA(const int i) {
      vel_x(i) += force_x(i) * dt;
      vel_y(i) += force_y(i) * dt;
      vel_z(i) += force_z(i) * dt;
      pos_x(i) += vel_x(i) * dt;
      pos_y(i) += vel_y(i) * dt;
      pos_z(i) += vel_z(i) * dt;
    });
  }

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(pos_x, pos_y, pos_z);
  return time;
}

template<typename ForceFn>
double benchmark_tecs_vector(
  const std::vector<Vector>& host_pos,
  const std::vector<Vector>& host_vel,
  const int steps,
  const type dt,
  double& checksum_out,
  int n_agents,
  ForceFn&& pairwise_force
) {
  auto checksum = [&](const ViewOfKocsVectors& positions) {
    auto host_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), positions);
    double sum = 0.0;
    for (int i = 0; i < host_pos.extent(0); ++i)
      sum += host_pos(i).x() + host_pos(i).y() + host_pos(i).z();
    return sum;
  };

  auto positions = ViewOfKocsVectors("pos", n_agents);
  auto velocities = ViewOfKocsVectors("vel", n_agents);
  auto forces = ViewOfKocsVectors("for", n_agents);

  auto host_pos_mirror = Kokkos::create_mirror_view(positions);
  auto host_vel_mirror = Kokkos::create_mirror_view(velocities);

  for (int i = 0; i < host_pos.size(); ++i) {
    host_pos_mirror(i) = KocsVector(host_pos[i].x, host_pos[i].y, host_pos[i].z);
    host_vel_mirror(i) = KocsVector(host_vel[i].x, host_vel[i].y, host_vel[i].z);
  }

  Kokkos::deep_copy(positions, host_pos_mirror);
  Kokkos::deep_copy(velocities, host_vel_mirror);

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int step = 0; step < steps; ++step) {
    Kokkos::parallel_for("zero_forces", n_agents, KOKKOS_LAMBDA(const int i) {
      forces(i) = KocsVector(type(0));
    });

    Kokkos::parallel_for(
      "compute_forces",
      Kokkos::TeamPolicy<>(n_agents, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int i = team.league_rank();
        const KocsVector pi = positions(i);
        KocsVector fi(type(0));

        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, n_agents),
          [&](const int j, KocsVector& local) {
            if (j == i)
              return;
            const KocsVector pj = positions(j);
            pairwise_force(pi, pj, local);
          },
          fi
        );

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          forces(i) = fi;
        });
      }
    );

    Kokkos::parallel_for("euler_update", n_agents, KOKKOS_LAMBDA(const int i) {
      KocsVector v = velocities(i);
      KocsVector p = positions(i);
      const KocsVector f = forces(i);

      v += f * dt; 
      p += v * dt;

      velocities(i) = v;
      positions(i) = p;
    });
  }

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

template<typename T>
void run_benchmark_case(int n_agents, int n_steps, int n_reps, T dt_in, BenchmarkType bench) {
  std::vector<Vector> host_pos(n_agents);
  std::vector<Vector> host_vel(n_agents);

  // initialize host positions/velocities with small random values
  {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<type> dist_pos(-1.0f, 1.0f);
    std::uniform_real_distribution<type> dist_vel(-0.01f, 0.01f);
    for (int i = 0; i < n_agents; ++i) {
      host_pos[i] = Vector(dist_pos(rng), dist_pos(rng), dist_pos(rng));
      host_vel[i] = Vector(dist_vel(rng), dist_vel(rng), dist_vel(rng));
    }
  }

  double checksum = 0.0;
  double total_time = 0.0;
  for (int i = 0; i < n_reps; ++i) {
    if (bench == BenchmarkType::ViewOfVectors) {
      auto pairwise_force = KOKKOS_LAMBDA(const Vector& pi, const Vector& pj, Vector& local) {
        const Vector r = pj - pi;
        const type dist_squared = r.norm_squared();
        if (dist_squared == type(0))
          return;

        const type inv_dist = type(1) / sqrt(dist_squared);
        const type inv_dist3 = inv_dist * inv_dist * inv_dist;
        local += r * inv_dist3;
      };

      total_time += benchmark_view_of_vectors(
        host_pos, host_vel, n_steps, static_cast<type>(dt_in), checksum, n_agents, pairwise_force
      );
    } else if (bench == BenchmarkType::ViewOfArrays) {
      auto pairwise_force = KOKKOS_LAMBDA(const Vector& pi, const Vector& pj, Vector& local) {
        const Vector r = pj - pi;
        const type dist_squared = r.norm_squared();
        if (dist_squared == type(0))
          return;

        const type inv_dist = type(1) / sqrt(dist_squared);
        const type inv_dist3 = inv_dist * inv_dist * inv_dist;
        local += r * inv_dist3;
      };

      total_time += benchmark_view_of_arrays(
        host_pos, host_vel, n_steps, static_cast<type>(dt_in), checksum, n_agents, pairwise_force
      );
    } else if (bench == BenchmarkType::ViewOfArraysRaw) {
      auto pairwise_force = KOKKOS_LAMBDA(
        const type pix, const type piy, const type piz,
        const type pjx, const type pjy, const type pjz,
        type& local_x, type& local_y, type& local_z
      ) {
        const type rx = pjx - pix;
        const type ry = pjy - piy;
        const type rz = pjz - piz;

        const type dist_squared = rx * rx + ry * ry + rz * rz;
        if (dist_squared == type(0))
          return;

        const type inv_dist = type(1) / sqrt(dist_squared);
        const type inv_dist3 = inv_dist * inv_dist * inv_dist;
        local_x += rx * inv_dist3;
        local_y += ry * inv_dist3;
        local_z += rz * inv_dist3;
      };

      total_time += benchmark_view_of_arrays_raw(
        host_pos, host_vel, n_steps, static_cast<type>(dt_in), checksum, n_agents, pairwise_force
      );
    } else if (bench == BenchmarkType::ViewOfScalars) {
      auto pairwise_force = KOKKOS_LAMBDA(
        const type pix, const type piy, const type piz,
        const type pjx, const type pjy, const type pjz,
        type& local_x, type& local_y, type& local_z
      ) {
        const type rx = pjx - pix;
        const type ry = pjy - piy;
        const type rz = pjz - piz;

        const type dist_squared = rx * rx + ry * ry + rz * rz;
        if (dist_squared == type(0))
          return;

        const type inv_dist = type(1) / sqrt(dist_squared);
        const type inv_dist3 = inv_dist * inv_dist * inv_dist;
        local_x += rx * inv_dist3;
        local_y += ry * inv_dist3;
        local_z += rz * inv_dist3;
      };

      total_time += benchmark_view_of_scalars(
        host_pos, host_vel, n_steps, static_cast<type>(dt_in), checksum, n_agents, pairwise_force
      );
    } else if (bench == BenchmarkType::KocsVector) {
      auto pairwise_force = KOKKOS_LAMBDA(
        const KocsVector& pi, const KocsVector& pj, KocsVector& local
      ) {
        const KocsVector r = pj - pi;
        const type dist_squared = r.length_squared();
        if (dist_squared == type(0))
          return;

        const type inv_dist = type(1) / Kokkos::sqrt(dist_squared);
        const type inv_dist3 = inv_dist * inv_dist * inv_dist;
        local += r * inv_dist3;
      };

      total_time += benchmark_tecs_vector(
        host_pos, host_vel, n_steps, static_cast<type>(dt_in), checksum, n_agents, pairwise_force
      );
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
  const std::vector<int> agent_counts = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
  const int steps = 100;
  const int repetitions = 10;

  Kokkos::initialize();
  {
    Kokkos::print_configuration(std::cout);

    std::cout << "benchmark,agents,steps,repetitions,dt,time_per_step_ms,checksum\n";

    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::ViewOfVectors);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::ViewOfArrays);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::ViewOfArraysRaw);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::ViewOfScalars);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::KocsVector);
  }
  Kokkos::finalize();

  return 0;
}
