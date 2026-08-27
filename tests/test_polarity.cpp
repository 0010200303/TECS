#include <Kokkos_Core.hpp>

#include "types/polarity.hpp"

#include "tests_common.hpp"

using namespace tecs;

template<typename Scalar>
int test_constructors_and_accessors() {
  Polarity_<Scalar> p1;
  ASSERT(p1.theta() == Scalar(0));
  ASSERT(p1.phi() == Scalar(0));

  Polarity_<Scalar> p2(Scalar(1.5), Scalar(2.5));
  ASSERT(p2.theta() == Scalar(1.5));
  ASSERT(p2.phi() == Scalar(2.5));

  Scalar val = Scalar(2.0);
  Polarity_<Scalar> p3(val);
  ASSERT(p3.theta() == val);
  ASSERT(p3.phi() == val);

  return 0;
}

template<typename Scalar>
int test_vector3_conversions() {
  const Scalar pi = Kokkos::numbers::pi_v<Scalar>;
  const Scalar half_pi = pi / Scalar(2);

  // +Z axis
  Vector3<Scalar> v1(0, 0, 1);
  Polarity_<Scalar> p1 = Polarity_<Scalar>::from_vector3(v1, Scalar(1));
  ASSERT(Kokkos::abs(p1.theta()) <= Scalar(1e-5));
  ASSERT(Kokkos::abs(p1.phi()) <= Scalar(1e-5));

  // +X axis
  Vector3<Scalar> v2(1, 0, 0);
  Polarity_<Scalar> p2 = Polarity_<Scalar>::from_vector3(v2);
  ASSERT(Kokkos::abs(p2.theta() - half_pi) <= Scalar(1e-5));
  ASSERT(Kokkos::abs(p2.phi()) <= Scalar(1e-5));

  // Convert back to vector
  Vector3<Scalar> v_out = p2.to_vector3();
  ASSERT(Kokkos::abs(v_out.x() - v2.x()) <= Scalar(1e-4));
  ASSERT(Kokkos::abs(v_out.y() - v2.y()) <= Scalar(1e-4));
  ASSERT(Kokkos::abs(v_out.z() - v2.z()) <= Scalar(1e-4));

  return 0;
}

template<typename Scalar>
int test_dot() {
  const Scalar pi = Kokkos::numbers::pi_v<Scalar>;
  const Scalar half_pi = pi / Scalar(2);

  Polarity_<Scalar> pX(half_pi, 0);       // pointing along +X
  Polarity_<Scalar> pY(half_pi, half_pi); // pointing along +Y
  Polarity_<Scalar> pZ(0, 0);             // pointing along +Z

  // Self dot should be 1
  ASSERT(Kokkos::abs(pX.dot(pX) - Scalar(1)) < Scalar(1e-4));
  ASSERT(Kokkos::abs(pY.dot(pY) - Scalar(1)) < Scalar(1e-4));
  ASSERT(Kokkos::abs(pZ.dot(pZ) - Scalar(1)) < Scalar(1e-4));

  // Orthogonal dots should be 0
  ASSERT(Kokkos::abs(pX.dot(pY)) < Scalar(1e-4));
  ASSERT(Kokkos::abs(pX.dot(pZ)) < Scalar(1e-4));
  ASSERT(Kokkos::abs(pY.dot(pZ)) < Scalar(1e-4));

  return 0;
}

template<typename Scalar>
int test_forces() {
  Polarity_<Scalar> p1(0.5, 0.5);
  Polarity_<Scalar> p2(0.6, 0.6);

  // Sanity check computational output for forces (avoids exact rigorous math proofs,
  // ensures no crashes/NaNs and logic completeness)
  Polarity_<Scalar> uni = p1.unidirectional_polarization_force(p2);
  Polarity_<Scalar> bi = p1.bidirectional_polarization_force(p2);
  
  Vector3<Scalar> disp(1, 0, 0);
  auto bend = p1.bending_force(disp, p2, Scalar(1.0));
  Vector3<Scalar> mig = p1.migration_force(disp, p2, Scalar(1.0));

  ASSERT(uni.size() == 2);
  ASSERT(bi.size() == 2);
  ASSERT(bend.vector.size() == 3);
  ASSERT(bend.polarity.size() == 2);
  ASSERT(mig.size() == 3);

  return 0;
}

#define RUN_ALL_TESTS(Type) do { \
  RUN_TEST((test_constructors_and_accessors<Type>())); \
  RUN_TEST((test_vector3_conversions<Type>())); \
  RUN_TEST((test_dot<Type>())); \
  RUN_TEST((test_forces<Type>())); \
} while(0)

int main(int argc, char* argv[]) {
    RUN_ALL_TESTS(float);
    RUN_ALL_TESTS(double);
    return 0;
}
