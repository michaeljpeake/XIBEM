XIBEM
=====

PhD code - eXtended Isogeometric Boundary Element Method

This code is a stripped down version of the code I wrote for my PhD research.

This version has had all visualisation code removed and is purely for obtaining data.

Four different meshes (two 2D and two 3D) have been included as examples of problems that can be sovled for.

---

For the NURBS and Bezier parts of these codes to work, the correct NURBSinC file must be placed in the BEM2D and BEM3D folders.

I have run some of the recursive functions in C because they are faster than running lots of for-loops inside Python. The functions can, of course, be run in Python if prefered.

Inside NURBSinC_files are the following files which may work for the following OS:

NURBSinC_linux32.so -- for 32-bit Linux (e.g. Ubuntu)
NURBSinC_linux64.so -- for 64-bit Linux (e.g. Ubuntu)
NURBSinC_mac64.so   -- this should work for all Mac OSX
NURBSinC_win32.pyd  -- for 32-Windows (compiled in Windows 7)

The right file should be copied into the BEM2D and BEM3D folders and renamed NURBSinC.so or NURBSinC.pyd as appropriate.

If a new file is needed, the NURBSinC.pyx file can be converted into C code and compiled by running the following command in the terminal:

python setup.py build_ext --inplace

This will require the Cython package to be installed for Python. I have found that this rarely works first time and a folder of NumPy Fortran (.h) files will have to be copied into an 'include' folder somewhere. Where the files are and where they have to be copied to varies on OS and Python installation, though much of this can be determined from reading the logfiles.
