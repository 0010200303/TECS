#ifndef TECS_HPP
#define TECS_HPP

// TODO: remove unnecessary includes?

#include "utils/utils.hpp"

#include "types/vector.hpp"
#include "types/polarity.hpp"
#include "types/plane.hpp"
#include "types/link.hpp"
#include "types/view.hpp"
#include "types/device_var.hpp"

#include "integrators/base.hpp"
#include "forces/detail.hpp"
#include "forces/kernel_fuser.hpp"

#include "initializers/line.hpp"
#include "initializers/spheres.hpp"
#include "initializers/hexagon.hpp"
#include "initializers/cuboid.hpp"
#include "initializers/rectangle.hpp"
#include "initializers/disk.hpp"
#include "initializers/heart.hpp"

#include "pair_finders/naive_all_pairs.hpp"
#include "pair_finders/naive_all_pairs_symmetric.hpp"
#include "pair_finders/binned_all_pairs.hpp"
#include "pair_finders/binned_all_pairs_symmetric.hpp"
#include "pair_finders/naive_gabriel.hpp"
#include "pair_finders/naive_gabriel_symmetric.hpp"
#include "pair_finders/binned_gabriel.hpp"
#include "pair_finders/binned_gabriel_symmetric.hpp"
#include "pair_finders/naive_delaunay.hpp"

#include "integrators/base.hpp"
#include "integrators/euler.hpp"
#include "integrators/heun.hpp"

#include "com_fixers/com_fixers.hpp"

#include "forces/predefined.hpp"

#include "simulation_config.hpp"
#include "simulation.hpp"

#include "io/dummy.hpp"
#include "io/hdf5_writer.hpp"
#include "io/hdf5_reader.hpp"

#include "utils/functions.hpp"
#include "utils/grid.hpp"

#include "argparse/arguments.hpp"

#endif // TECS_HPP
