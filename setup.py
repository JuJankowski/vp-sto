from setuptools import setup

# packages = ['vpsto'],  # indicates a multi-file module and that we have a vpsto folder and vpsto/__init__.py file

try:
    with open('README.md') as file:
        long_description = file.read()  # now assign long_description=long_description below
except IOError:  # file not found
    pass

setup(name="vpsto",
      long_description=long_description,  # __doc__, # can be used in the vpsto.py file
      long_description_content_type = 'text/markdown',
      version='1.0.0',
      description="Via-point based Stochastic Trajectory Optimization",
      author="Julius Jankowski",
      author_email="authors_firstname.lastname@epfl.ch",
      maintainer="Julius Jankowski",
      maintainer_email="authors_firstname.lastname@epfl.ch",
      url="https://github.com/JuJankowski/vp-sto",
      license="BSD",
      packages=["vpsto"],
      install_requires=["numpy", "cma"],
      extras_require={
            "plotting": ["matplotlib"],
            "collision": ["shapely"],
      },
)
