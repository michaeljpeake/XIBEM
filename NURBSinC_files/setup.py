# To compile new NURBSinC executable file,
# run 'python setup.py build_ext --inplace' in terminal window
#
# This will require the Cython package for Python
# Sometimes (often actually) you will have move a folder of fortran files to
# get the compilation to work. It's not well supported on the internet but
# should take too long to work out which folder needs to be copied to where
#

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("NURBSinC", ["NURBSinC.pyx"])]
)


