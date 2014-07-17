# Michael Peake
# Durham University

import numpy as np
import BEM3D.Meshes as Meshes
import BEM3D.Basis as Basis
import BEM3D.Collocation as Collocation
import BEM3D.Assembly as Assembly
import BEM3D.PostProcessing as PostProcessing
import BEM3D.ExactSolutions as Exact

##########################
# Set initial parameters #
##########################

incident_wavenumber = 5
incident_wave_direction = [1.0,0.0,0.0]
incident_wave_direction /= np.linalg.norm(incident_wave_direction) # Normalises inc vector
incident_wave_amplitude = 1.0

use_CHIEF = True
integration_order = 6

simulation_type = 5
# 1: Conventional BEM with continuous quadratic elements
# 2: Conventional BEM with continuous quadratic serendipity elements
# 3: PU-BEM
# 4: IGABEM (rational Bezier interpolation)
# 5: XIBEM (rational Bezier interpolation)


########
# Mesh #
########
if simulation_type==1:
    N = 6
    mesh = Meshes.Sphere_CubeToStandardLagrangeQuad(N)
if simulation_type==2:
    N = 6
    mesh = Meshes.Sphere_CubeToSerendipityLagrangeQuad(N)
if simulation_type==3:
    mesh = Meshes.Sphere_ExactSerendipityLagrangeQuad()
if simulation_type==4:
    N = 4
    mesh = Meshes.Sphere_Bezier(N)
if simulation_type==5:
    mesh = Meshes.Sphere_Bezier()    


######################################
# Field variable approximation basis #
######################################
if simulation_type==1 or simulation_type==2 or simulation_type==4:
    Basis.Function_Variable_Approximation_Basis(mesh)
if simulation_type==3 or simulation_type==5:
    waves_in_enrichment = 10
    Basis.Function_Variable_Approximation_Basis(mesh,waves_in_enrichment,incident_wavenumber,incident_wave_direction,'CoulombSphere')
    #Basis.Function_Variable_Approximation_Basis(mesh,waves_in_enrichment,incident_wavenumber,incident_wave_direction,'StructuredGrid')


######################################
#            Collocation             #
######################################
if simulation_type==1 or simulation_type==2 or simulation_type==4:
    Collocation.collocate_at_nodes(mesh)
if simulation_type==3 or simulation_type==5:
    Collocation.enriched_collocation(mesh)
if use_CHIEF:
    Collocation.CHIEF_sphere(mesh,number_of_CHIEF_points=10)
    #Collocation.CHIEF_sphere(mesh,fraction_extra_collocation=0.2)

   
#####################################
#     Assemble system matrices      #
#####################################
assembler = Assembly.Assembler(mesh,incident_wavenumber,incident_wave_amplitude,incident_wave_direction)
incident_wave_vector = assembler.Incident_Wave_Vector()
# Can either use CBIE with Telles, or RBIE without
# The latter is much quicker
#system_matrix = assembler.Helmholtz_CBIE_Matrix(integration_order,Telles=True)
system_matrix = assembler.Helmholtz_RBIE_Matrix(integration_order)


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
# Surface potentials
if simulation_type==1 or simulation_type==2 or simulation_type==4:
    BEM_Phi,EvalPoints = PostProcessing.Boundary_Evalutation(mesh.dList[0],DoF_Coefficients,5,return_points=True)
else:
    BEM_Phi,EvalPoints = PostProcessing.Boundary_Evalutation(mesh.dList[0],DoF_Coefficients,10,return_points=True)
Exact_Phi = Exact.SpherePlaneWave(incident_wavenumber,EvalPoints[0])

# Far-field potentials
theta=np.linspace(0,2*np.pi)
r=3
x=r*np.cos(theta)
y=r*np.sin(theta)
z=np.zeros(x.shape)
BEM_Phi2 = PostProcessing.off_scatterer_solve(mesh,DoF_Coefficients,np.vstack([x,y,z]),incident_wavenumber,incident_wave_direction)
Exact_Phi2 = Exact.SpherePlaneWave2(incident_wavenumber,1,r*np.ones(x.shape),theta)

L2_Error = np.sqrt(np.sum(np.abs(Exact_Phi-BEM_Phi)**2) / np.sum(np.abs(Exact_Phi)**2))
L2_Error2 = np.sqrt(np.sum(np.abs(Exact_Phi2-BEM_Phi2)**2) / np.sum(np.abs(Exact_Phi2)**2))

print
print "Number of elements: %s" % mesh.numElements
print "Number of DoFs      %s" % mesh.ndof
print "Tau:                %.1f" % ((2*np.pi/incident_wavenumber)*np.sqrt(mesh.ndof/(4*np.pi)))
print "Surface L2 Error:   %f" % L2_Error
print "Far-field L2 Error: %f" % L2_Error2




