from setuptools import find_packages, setup

setup(
    name="pys2sleplet",
    version="0.1.0",
    author="Patrick Roddy",
    author_email="patrickjamesroddy@gmail.com",
    packages=find_packages(exclude=["*test"]),
    include_package_data=True,
    entry_points={"console_scripts": ["plotting=pys2sleplet.scripts.command:main"]},
)
