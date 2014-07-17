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

incident_wavenumber = 10
incident_wave_direction = [1.0,1.0]
incident_wave_direction /= np.linalg.norm(incident_wave_direction) # Normalises inc vector
incident_wave_amplitude = 1.0

use_CHIEF = True
integration_order = 10

simulation_type = 2
tau = 3
# 1: Conventional BEM with continuous quadratic elements
# 2: PU-BEM (continous quadratic shape functions)
# 3: PU-BEM (continous trigonometric shape functions)
# 4: IGABEM (rational Bezier interpolation)
# 5: XIBEM (rational Bezier interpolation)


########
# Mesh #
########
straight_length_factor = 1
if simulation_type==1:
    E_per_segment = int(np.ceil(tau*incident_wavenumber*0.125*(1+straight_length_factor)))
    mesh = Meshes.Capsule_Quadratic(1.0,straight_length_factor,E_per_segment)
if simulation_type==2 or simulation_type==3:
    E_per_segment=1
    mesh = Meshes.Capsule_Exact(1,straight_length_factor,E_per_segment)
if simulation_type==4:
    E = int(np.ceil(tau*incident_wavenumber*0.0625*(1+straight_length_factor)))
    mesh = Meshes.Capsule_Bezier(split_segments_into=E)
if simulation_type==5:
    mesh = Meshes.Capsule_Bezier()


######################################
# Field variable approximation basis #
######################################
if simulation_type==1 or simulation_type==4:
    Basis.Function_Variable_Approximation_Basis(mesh)
if simulation_type==2 or simulation_type==3 or simulation_type==5:
    #number_of_in_enrichment_waves = int(np.ceil(tau*incident_wavenumber/mesh.numElements))
    number_of_in_enrichment_waves = int(np.ceil(0.5*(1+straight_length_factor)*incident_wavenumber*tau/mesh.numElements))
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


#####################################
#           Post processing         #
#####################################
BEM_Phi,EvalPoints = PostProcessing.Uniform_Boundary_Evalutation(mesh.dList[0],DoF_Coefficients,1000,return_points=True)
MFS_Phi = Exact.Alternative_MFS(mesh,EvalPoints[0],EvalPoints[1],incident_wavenumber,incident_wave_direction,tau=7,frac_samp=2,offset=0.15)
L2_Error = np.sqrt(np.sum(np.abs(MFS_Phi-BEM_Phi)**2) / np.sum(np.abs(MFS_Phi)**2))
print "Tau: %.2f" % (1.0*mesh.ndof/incident_wavenumber/(1+straight_length_factor))
print "L2 Error: %f" % (L2_Error)

