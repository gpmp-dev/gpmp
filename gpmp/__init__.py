# set multithreaded/multicore parallelism
# see https://github.com/google/jax/issues/8345
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=512"

# set double precision for floats
import jax
jax.config.update("jax_enable_x64", True)
eps = jax.numpy.finfo(jax.numpy.float64).eps

from . import core
from . import kernel
from . import misc
