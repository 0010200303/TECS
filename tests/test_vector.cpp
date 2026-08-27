#include <Kokkos_Core.hpp>
#include "types/vector.hpp"
#include "tests_common.hpp"

using namespace tecs;

template<typename Scalar, unsigned int dimensions>
int test_addition() {
  VectorN<Scalar, dimensions> v1(Scalar(2800));
  VectorN<Scalar, dimensions> v2(Scalar(7));
  VectorN<Scalar, dimensions> v3 = v1 + v2;

  for (unsigned int i = 0; i < dimensions; ++i)
    ASSERT(v3[i] == Scalar(2807));

  return 0;
}

template<typename Scalar, unsigned int dimensions>
int test_arithmetic() {
  VectorN<Scalar, dimensions> v1(Scalar(10));
  VectorN<Scalar, dimensions> v2(Scalar(2));
  
  auto vadd = v1 + v2;
  auto vsub = v1 - v2;
  auto vmul = v1 * v2;
  auto vdiv = v1 / v2;

  for (unsigned int i = 0; i < dimensions; ++i) {
    ASSERT(vadd[i] == Scalar(12));
    ASSERT(vsub[i] == Scalar(8));
    ASSERT(vmul[i] == Scalar(20));
    ASSERT(vdiv[i] == Scalar(5));
  }

  auto sadd = v1 + Scalar(2);
  auto ssub = v1 - Scalar(2);
  auto smul = v1 * Scalar(2);
  auto sdiv = v1 / Scalar(2);

  for (unsigned int i = 0; i < dimensions; ++i) {
    ASSERT(sadd[i] == Scalar(12));
    ASSERT(ssub[i] == Scalar(8));
    ASSERT(smul[i] == Scalar(20));
    ASSERT(sdiv[i] == Scalar(5));
  }

  auto rsub = Scalar(10) - v2;
  auto rdiv = Scalar(10) / v2;
  for (unsigned int i = 0; i < dimensions; ++i) {
    ASSERT(rsub[i] == Scalar(8));
    ASSERT(rdiv[i] == Scalar(5));
  }

  return 0;
}

template<typename Scalar, unsigned int dimensions>
int test_assignment() {
  VectorN<Scalar, dimensions> v1(Scalar(10));
  VectorN<Scalar, dimensions> v2(Scalar(2));

  v1 += v2;
  for (unsigned int i = 0; i < dimensions; ++i) ASSERT(v1[i] == Scalar(12));

  v1 -= v2;
  for (unsigned int i = 0; i < dimensions; ++i) ASSERT(v1[i] == Scalar(10));

  v1 *= v2;
  for (unsigned int i = 0; i < dimensions; ++i) ASSERT(v1[i] == Scalar(20));

  v1 /= v2;
  for (unsigned int i = 0; i < dimensions; ++i) ASSERT(v1[i] == Scalar(10));

  v1 += Scalar(2);
  for (unsigned int i = 0; i < dimensions; ++i) ASSERT(v1[i] == Scalar(12));

  v1 -= Scalar(2);
  for (unsigned int i = 0; i < dimensions; ++i) ASSERT(v1[i] == Scalar(10));

  v1 *= Scalar(2);
  for (unsigned int i = 0; i < dimensions; ++i) ASSERT(v1[i] == Scalar(20));

  v1 /= Scalar(2);
  for (unsigned int i = 0; i < dimensions; ++i) ASSERT(v1[i] == Scalar(10));

  return 0;
}

template<typename Scalar, unsigned int dimensions>
int test_math() {
  VectorN<Scalar, dimensions> v1(Scalar(2));
  VectorN<Scalar, dimensions> v2(Scalar(3));

  Scalar d = v1.dot(v2);
  ASSERT(d == Scalar(6 * dimensions));

  ASSERT(v1.length_squared() == Scalar(4 * dimensions));
  ASSERT(v1.distance_to_squared(v2) == Scalar(1 * dimensions));

  if constexpr (std::is_floating_point_v<Scalar>) {
    VectorN<Scalar, dimensions> v_norm = v1.normalized();
    ASSERT(Kokkos::abs(v_norm.length_squared() - Scalar(1)) < Scalar(1e-4));
  }
  
  return 0;
}

template<typename Scalar>
int test_cross() {
  Vector3<Scalar> v1(Scalar(1), Scalar(0), Scalar(0));
  Vector3<Scalar> v2(Scalar(0), Scalar(1), Scalar(0));
  auto v3 = v1.cross(v2);
  ASSERT(v3.x() == Scalar(0));
  ASSERT(v3.y() == Scalar(0));
  ASSERT(v3.z() == Scalar(1));
  return 0;
}

template<typename Scalar, unsigned int dimensions>
int test_unary() {
  VectorN<Scalar, dimensions> v(Scalar(5));
  auto v_neg = -v;
  auto v_pos = +v;
  for (unsigned int i = 0; i < dimensions; ++i) {
    ASSERT(v_neg[i] == Scalar(-5));
    ASSERT(v_pos[i] == Scalar(5));
  }
  return 0;
}

template<typename Scalar, unsigned int dimensions>
int test_constructors_and_accessors() {
  VectorN<Scalar, dimensions> v1;
  for (unsigned int i = 0; i < dimensions; ++i) ASSERT(v1[i] == Scalar(0));

  std::array<Scalar, dimensions> arr;
  arr.fill(Scalar(7));
  VectorN<Scalar, dimensions> v2(arr);
  for (unsigned int i = 0; i < dimensions; ++i) ASSERT(v2[i] == Scalar(7));

  ASSERT(v2.size() == dimensions);

  if constexpr (dimensions > 0) ASSERT(v2.x() == Scalar(7));
  if constexpr (dimensions > 1) ASSERT(v2.y() == Scalar(7));
  if constexpr (dimensions > 2) ASSERT(v2.z() == Scalar(7));
  if constexpr (dimensions > 3) ASSERT(v2.w() == Scalar(7));

  return 0;
}

#define RUN_ALL_TESTS(Type, Dim) do { \
  RUN_TEST((test_addition<Type, Dim>())); \
  RUN_TEST((test_arithmetic<Type, Dim>())); \
  RUN_TEST((test_assignment<Type, Dim>())); \
  RUN_TEST((test_math<Type, Dim>())); \
  RUN_TEST((test_unary<Type, Dim>())); \
  RUN_TEST((test_constructors_and_accessors<Type, Dim>())); \
} while(0)

int main(int argc, char* argv[]) {
  RUN_ALL_TESTS(float, 1);
  RUN_ALL_TESTS(float, 2);
  RUN_ALL_TESTS(float, 3);
  RUN_ALL_TESTS(float, 4);
  
  RUN_ALL_TESTS(double, 1);
  RUN_ALL_TESTS(double, 2);
  RUN_ALL_TESTS(double, 3);
  RUN_ALL_TESTS(double, 4);
  
  RUN_ALL_TESTS(int, 1);
  RUN_ALL_TESTS(int, 2);
  RUN_ALL_TESTS(int, 3);
  RUN_ALL_TESTS(int, 4);

  RUN_TEST((test_cross<float>()));
  RUN_TEST((test_cross<double>()));
  RUN_TEST((test_cross<int>()));

  return 0;
}
