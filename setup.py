from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fast_sligru_cpp',
    ext_modules=[
        CppExtension(
            'fast_sligru_cpp',
            ['sligru.cpp', 'sligru_kernel.cu']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })