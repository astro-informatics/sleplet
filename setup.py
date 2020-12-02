from setuptools import Extension, find_namespace_packages, setup

try:
    from Cython.Build import cythonize
except ImportError:

    USE_CYTHON = False
else:
    USE_CYTHON = True

ext = ".pyx" if USE_CYTHON else ".c"
extensions = Extension("slepian_computations", ["pys2sleplet/cython/*" + ext])

if USE_CYTHON:
    extensions = cythonize(
        extensions,
        annotate=True,
        language_level=3,
        compiler_directives=dict(boundscheck=False, embedsignature=True),
    )

setup(
    name="pys2sleplet",
    version="0.1.0",
    author="Patrick Roddy",
    author_email="patrickjamesroddy@gmail.com",
    packages=find_namespace_packages(),
    include_package_data=True,
    entry_points=dict(console_scripts=["plotting=pys2sleplet.scripts.plotting:main"]),
    ext_modules=extensions,
)
