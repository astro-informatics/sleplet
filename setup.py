from setuptools import find_namespace_packages, setup

setup(
    name="sleplet",
    version="0.1.3",
    author="Patrick Roddy",
    author_email="patrickjamesroddy@gmail.com",
    packages=find_namespace_packages(),
    include_package_data=True,
    entry_points=dict(
        console_scripts=[
            "sphere=sleplet.scripts.plotting_on_sphere:main",
            "mesh=sleplet.scripts.plotting_on_mesh:main",
        ],
    ),
)
