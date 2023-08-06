from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fast_sligru',
    ext_modules=[
        CppExtension(
            'fast_sligru_cpp',
            [
                'fast_sligru/csrc/sligru_kernel.cu',
                'fast_sligru/csrc/rnns.cpp', 
                'fast_sligru/csrc/ligru_kernel.cu',
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }, 
    version="0.1.0", 
    author="Adel Moumen",
    author_email="adel.moumen@univ-avignon.fr",
    description="A fast CUDA implementation of the SLiGRU model.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    python_requires=">=3.8",
)