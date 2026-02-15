import os
import glob
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --- CONFIGURATION ---
# Path to the source code (relative to this script)
OPS_REL_PATH = "./torch_ops"
OUTPUT_DIR= "./torch_ops/build"
OPS_ROOT = os.path.abspath(OPS_REL_PATH)

OPS_LIST = ["patch_outerprod", "soft_bin_ops", "sym3eig"]

print(f"Looking for ops in: {OPS_ROOT}")

if not os.path.exists(OPS_ROOT):
    print("Error: Ops source directory not found!")
    sys.exit(1)

class CustomBuildExtension(BuildExtension):
    def initialize_options(self):
        super().initialize_options()
        self.build_lib = OUTPUT_DIR  # Force output directory

    def get_ext_filename(self, ext_name):
        # Force simple filename: "kinematics_cuda.so" instead of long python name
        return ext_name + ".so"

# --- GATHER EXTENSIONS ---
extensions = []
for op in OPS_LIST:
    op_dir = os.path.join(OPS_ROOT, op)
    sources = glob.glob(os.path.join(op_dir, "*.cpp")) + glob.glob(os.path.join(op_dir, "*.cu"))
    
    if not sources:
        print(f"Warning: No sources found for {op}")
        continue
        
    print(f"Configuring {op}...")
    ext = CUDAExtension(
        name=f"{op}_cuda",
        sources=sources,
        include_dirs=[OPS_ROOT, op_dir],
        extra_compile_args={
            'cxx': ['-O3', '-fopenmp', '-w'],  # -w suppresses warnings
            'nvcc': ['-O3', '--expt-extended-lambda', '--expt-relaxed-constexpr', '-w']
        }
    )
    extensions.append(ext)

# --- BUILD ---
setup(
    name='gum_custom_ops',
    ext_modules=extensions,
    cmdclass={'build_ext': CustomBuildExtension},
    script_args=['build_ext'] # Build .so files right here
)
