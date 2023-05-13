from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fast_ligru',
    ext_modules=[
        CppExtension(
            'fast_ligru',
            ['fast_ligru.cpp', 'fast_ligru.cu']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })