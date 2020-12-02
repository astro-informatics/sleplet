import os

from Cython.Build import cythonize
from setuptools import Extension, find_namespace_packages, setup

os.environ["CC"] = "gcc-10"

setup(
    name="pys2sleplet",
    version="0.1.0",
    author="Patrick Roddy",
    author_email="patrickjamesroddy@gmail.com",
    packages=find_namespace_packages(),
    include_package_data=True,
    entry_points=dict(console_scripts=["plotting=pys2sleplet.scripts.plotting:main"]),
    ext_modules=cythonize(
        Extension(
            "slepian_computations",
            ["pys2sleplet/cython/*.pyx"],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"],
            include_dirs=["/usr/local/include"],
        ),
        annotate=True,
        language_level=3,
        compiler_directives=dict(boundscheck=False, embedsignature=True),
    ),
)
