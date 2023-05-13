from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fast_ligru',
    ext_modules=[
        CppExtension(
            'fast_ligru',
            ['ligru.cpp']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })