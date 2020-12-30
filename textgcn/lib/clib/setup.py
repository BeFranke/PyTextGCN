import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("graphbuilder", ["graphbuilder.pyx"],
              include_dirs=[numpy.get_include()],
              define_macros=[
                  ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                  # ("PYREX_WITHOUT_ASSERTIONS", "")
              ])
]
setup(
    name="graphbuilder",
    ext_modules=cythonize(
        extensions,
        annotate=True,
        # gdb_debug=True
    ),
)
