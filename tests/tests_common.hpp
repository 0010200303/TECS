#ifndef TECS_TESTS_COMMON_HPP
#define TECS_TESTS_COMMON_HPP

#include <iostream>

#define ASSERT(cond) do { \
  if (!(cond)) { \
    std::cerr << "Assertion failed: " << #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    return 1; \
  } \
} while(0)

#define RUN_TEST(...) \
  do { \
    if (int err = (__VA_ARGS__)) { \
      std::cerr << "Test failed: " << #__VA_ARGS__ << std::endl; \
      return err; \
    } \
  } while(0)

#endif // TECS_TESTS_COMMON_HPP
