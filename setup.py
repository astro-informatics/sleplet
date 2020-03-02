from setuptools import find_namespace_packages, setup

setup(
    name="pys2sleplet",
    version="0.1.0",
    author="Patrick Roddy",
    author_email="patrickjamesroddy@gmail.com",
    packages=find_namespace_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["plotting=pys2sleplet.scripts.plotting:main"]},
)
