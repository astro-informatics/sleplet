from setuptools import find_packages, setup

setup(
    name="pys2sleplet",
    version="0.1.0",
    author="Patrick Roddy",
    author_email="patrickjamesroddy@gmail.com",
    packages=find_packages(),
    scripts=["scripts/convert_data.py", "scripts/plotting.py"],
)
