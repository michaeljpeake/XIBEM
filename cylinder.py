# Michael Peake
# Durham University

import numpy as np
import BEM2D.Meshes as Meshes
import BEM2D.Basis as Basis
import BEM2D.Collocation as Collocation
import BEM2D.Assembly as Assembly
import BEM2D.PostProcessing as PostProcessing
import BEM2D.ExactSolutions as Exact

##########################
# Set initial parameters #
##########################

incident_wavenumber = 50
incident_wave_direction = [1.0,0.0]
incident_wave_direction /= np.linalg.norm(incident_wave_direction) # Normalises inc vector
incident_wave_amplitude = 1.0

use_CHIEF = True
integration_order = 6

simulation_type = 1
tau = 10
# 1: Conventional BEM with continuous quadratic elements
# 2: PU-BEM (continous quadratic shape functions)
# 3: PU-BEM (continous trigonometric shape functions)
# 4: IGABEM (rational Bezier interpolation)
# 5: XIBEM (rational Bezier interpolation)


########
# Mesh #
########
if simulation_type==1:
    number_of_elements = int(np.ceil(tau*incident_wavenumber/2.0))
    mesh = Meshes.Cylinder_Quadratic(number_of_elements)
if simulation_type==2 or simulation_type==3:
    mesh = Meshes.Cylinder_Exact(num_elements=2)
if simulation_type==4:
    elements_per_quarter_circle = int(np.ceil(tau*incident_wavenumber/8.0))
    mesh = Meshes.Cylinder_Bezier(elements_per_quarter_circle)
if simulation_type==5:
    mesh = Meshes.Cylinder_Bezier(elements_per_quarter_circle=1)


######################################
# Field variable approximation basis #
######################################
if simulation_type==1 or simulation_type==4:
    Basis.Function_Variable_Approximation_Basis(mesh)
if simulation_type==2 or simulation_type==3 or simulation_type==5:
    number_of_in_enrichment_waves = int(np.ceil(0.5*tau*incident_wavenumber/mesh.numElements))
    Basis.Function_Variable_Approximation_Basis(mesh,number_of_in_enrichment_waves,incident_wavenumber,incident_wave_direction)
    if simulation_type==3:
        for e in mesh.eList: e.shape_functions = e.TrigonometricShapeFunctions


########################
#     Collocation      #
########################
Collocation.EquallySpacedInLocalCoordinate(mesh)
#if use_CHIEF: Collocation.CHIEF(mesh,number_of_CHIEF_points=20)
if use_CHIEF: Collocation.CHIEF(mesh,fraction_extra_collocation=0.20)


#####################################
#     Assemble system matrices      #
#####################################
assembler = Assembly.Assembler(mesh,incident_wavenumber,incident_wave_amplitude,incident_wave_direction)
incident_wave_vector = assembler.Incident_Wave_Vector()
system_matrix = assembler.Helmholtz_CBIE_Matrix(integration_order)


#####################################
#       Solve system matrices       #
#####################################

# Use SVD to solve an Ax=b system.
# Faster solvers available for conventional BEM simulations
# but haven't bothered using them.
# The parameter rcond below is the cut-off ratio for small singular values of the system matrix
# Values small than rcond are set to zero.
DoF_Coefficients,residuals,rank,s = np.linalg.lstsq(system_matrix,incident_wave_vector,rcond=1e-10)
Condition_Number = abs(s[0]/s[-1]) # Find the condition number of the system

## The following is an older version of a code I had. It work well in 64-bit Linux but seems
## to cause errors in 32-bit systems and those with little memory.
#A = system_matrix
#from scipy.linalg import pinv,diagsvd
#U,s,Vh = np.linalg.svd(A) # Decompose system_matrix using SVD
#Condition_Number = abs(s[0]/s[-1]) # Find the condition number of the system
#s[np.where(s<1e-10)]=0 # Ignore values less than ... this is called a truncated SVD
#A = np.dot(np.conj(Vh).T,np.dot(pinv(diagsvd(s,A.shape[0],A.shape[1])),np.conj(U).T)) # Pseudo inverse
#DoF_Coefficients = np.dot(A,incident_wave_vector) # Dot product of b and pseudo inverse of A


#####################################
#           Post processing         #
#####################################
BEM_Phi,EvalPoints = PostProcessing.Uniform_Boundary_Evalutation(mesh.dList[0],DoF_Coefficients,1000,return_points=True)
Exact_Phi = Exact.cylinder(incident_wavenumber,EvalPoints[0],EvalPoints[1])
L2_Error = np.sqrt(np.sum(np.abs(Exact_Phi-BEM_Phi)**2) / np.sum(np.abs(Exact_Phi)**2))
print "Tau: %.2f" % (1.0*mesh.ndof / incident_wavenumber)
print "L2 Error: %f" % (L2_Error)

