import os
import sys
from importlib.machinery import ExtensionFileLoader, ModuleSpec
from importlib.util import module_from_spec

# Path to liblcvm pybind shared library
LBLCVM_PYBIND_LIB_PATH = "/tmp/lcvm_push/liblcvm/build"
LIBRARY_PATH = os.path.join(LBLCVM_PYBIND_LIB_PATH, "liblcvm_pybind.so")

print(f"LIBLCVM PYBIND LIBRARY_PATH: {LIBRARY_PATH}")

# Verify that the liblcvm pybind shared library exists
if not os.path.exists(LIBRARY_PATH):
    raise FileNotFoundError(
        f"LIBLCVM PYBIND shared library not found at: {LIBRARY_PATH}"
    )

# Load the liblcvm pybind shared library
try:
    loader = ExtensionFileLoader("liblcvm_pybind", LIBRARY_PATH)
    spec = ModuleSpec("liblcvm_pybind", loader, origin=LIBRARY_PATH)
    cpp_lib = module_from_spec(spec)
    loader.exec_module(cpp_lib)
except Exception as e:
    print(f"Failed to load LIBLCVM PYBIND shared library: {e}")
    raise

# Expose liblcvm pybind shared library module's contents
globals().update({name: getattr(cpp_lib, name) for name in dir(cpp_lib)})
