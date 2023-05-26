from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fast_sligru',
    ext_modules=[
        CppExtension(
            'fast_sligru_cpp',
            ['fast_sligru/csrc/sligru.cpp', 'fast_sligru/csrc/sligru_kernel.cu']
            # extra_compile_args=['-use_fast_math']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }, 
    version="0.1.0", 
    packages=find_packages()
)