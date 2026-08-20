from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="int4_down_ops",
    ext_modules=[CUDAExtension(
        name="int4_down_ops",
        sources=["int4_gather_down.cu"],
        extra_compile_args={"nvcc": [
            "-O3", "--use_fast_math", "-arch=sm_75",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF_CONVERSIONS__"]},
    )],
    cmdclass={"build_ext": BuildExtension},
)