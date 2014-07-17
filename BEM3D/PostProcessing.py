# Michael Peake
# Durham University

import numpy as np
from scipy.special.orthogonal import p_roots
from warnings import simplefilter

def Boundary_Evalutation(Domain,DoF_Coefficients,num_points,return_points=False):
    """Solves points on an N x N square on each element of domain"""
    
    potentials=[]
    
    for e in Domain.eList:
    
        xi1=np.linspace(e.limits[0],e.limits[1],num_points)
        xi2=np.linspace(e.limits[2],e.limits[3],num_points)
        xi1,xi2=np.meshgrid(xi1,xi2)
        xi1=xi1.reshape(-1,)
        xi2=xi2.reshape(-1,)
        
        p = e.vals(xi1,xi2)
        px,py,pz = p        
        if e==Domain.eList[0]:
            XYZ = p
            xi1_temp = xi1
            xi2_temp = xi2
        else:
            px=px.reshape(-1,1) ; py=py.reshape(-1,1) ; pz=pz.reshape(-1,1)
            qx,qy,qz=XYZ
            r = np.sqrt( (qx-px)**2 + (qy-py)**2 + (qz-pz)**2 )
            delete = np.where(np.any(r<1e-10,axis=1))[0]
            xi1_temp = np.delete(xi1,delete)
            xi2_temp = np.delete(xi2,delete)
            p = e.vals(xi1_temp,xi2_temp)
            px,py,pz = p
            XYZ = np.hstack([XYZ,p])

        Basis = e.shape_functions(xi1_temp,xi2_temp)[:,:,0]
        Basis = np.repeat(Basis,e.shapeFunList[0].M,axis=1) # Assumes global M !!!!
        Basis = np.asarray(Basis,np.complex) # Make InterpolationBasis a complex matrix
        Basis *= np.vstack([dof(p[0],p[1],p[2]) for shapefun in e.shapeFunList for dof in shapefun.DegreesOfFreedomList]).T
        
        DoF_mapping = [dof.ID for shapefun in e.shapeFunList for dof in shapefun.DegreesOfFreedomList]
        element_coefficients = DoF_Coefficients[DoF_mapping].reshape(1,-1)
        potentials = np.append(potentials,np.dot(element_coefficients,Basis.T))
                    
    if return_points: return potentials,XYZ
    return potentials
        


def off_scatterer_solve(mesh,DoF_Coefficients,evaluation_points,wavenumber,incDir,num_gauss=6):
    quadrature = GaussQuadrature(num_gauss,num_gauss)
    print "Evaluating scattering analysis matrix"
    system = Assemble_CBIE_Matrix(mesh,quadrature,evaluation_points,wavenumber)
    system = np.dot(system,DoF_Coefficients)
    return np.exp(1j*wavenumber*np.dot(evaluation_points.T,incDir))-system


def Assemble_CBIE_Matrix(mesh,quadrature,evaluation_points,wavenumber):
    """Assemble a system matrix using the supplied kernel"""

    System_Matrix = np.zeros((evaluation_points.shape[1],mesh.ndof),np.complex)

    print "%s percent complete" % 0
    Es = np.array(mesh.eList)
    E = len(mesh.eList)
    comp_old = 25

    for e in mesh.eList:
        ## To see how far through simulation is
        eNum = np.where(e==Es)[0][0]
        if 100*eNum/E > comp_old:
            print "%s percent complete" % comp_old
            comp_old += 25
        ##
            
        # Change of integration interval
        xi1 = (quadrature.xi1*0.5*(e.limits[1]-e.limits[0])+0.5*(e.limits[0]+e.limits[1])).reshape(-1,)
        xi2 = (quadrature.xi2*0.5*(e.limits[3]-e.limits[2])+0.5*(e.limits[2]+e.limits[3])).reshape(-1,)
        w1 = quadrature.w1*0.5*(e.limits[1]-e.limits[0])
        w2 = quadrature.w2*0.5*(e.limits[3]-e.limits[2])
        
        s = np.arange(0,e.num_integration_cells).reshape(-1,1)
        xi1 = ((xi1 - e.limits[0] + (e.limits[1]-e.limits[0])*s)/e.num_integration_cells + e.limits[0]).reshape(-1,)
        xi2 = ((xi2 - e.limits[2] + (e.limits[3]-e.limits[2])*s)/e.num_integration_cells + e.limits[2]).reshape(-1,)
        w1 = np.repeat([w1/e.num_integration_cells],e.num_integration_cells,axis=0).reshape(-1,)
        w2 = np.repeat([w2/e.num_integration_cells],e.num_integration_cells,axis=0).reshape(-1,)
        xi1,xi2 = np.meshgrid(xi1,xi2)
        w1,w2 = np.meshgrid(w1,w2)
        xi1=xi1.reshape(-1,); xi2=xi2.reshape(-1,)
        w1=w1.reshape(-1); w2=w2.reshape(-1)
        

        S = e.num_integration_cells
        # quadmap : quadrature map
        # Loop either in xrange(S) --fast , or xrange(S**2) should never be memory error from integration
        for s in xrange(S):
            quadmap = range(s*S*quadrature.n1*quadrature.n2,(s+1)*S*quadrature.n1*quadrature.n2)
        #for s in xrange(S**2):     
            #quadmap = range(s*quadrature.n1*quadrature.n2,(s+1)*quadrature.n1*quadrature.n2)     
            Element_Matrix = Assemble_CBIE_Element(mesh,e,xi1[quadmap],xi2[quadmap],w1[quadmap],w2[quadmap],evaluation_points,wavenumber)  
            DegreeOfFreedomMapping = [dof.ID for s in e.shapeFunList for dof in s.DegreesOfFreedomList]
            Unique_DoFs=np.unique(DegreeOfFreedomMapping)
            Element_Matrix=np.hstack([np.sum(Element_Matrix[:,np.where(DegreeOfFreedomMapping==dof)[0]],axis=1).reshape(-1,1) for dof in Unique_DoFs])
            System_Matrix[:,Unique_DoFs] += Element_Matrix
        
    print "%s percent complete" % 100
       
    return System_Matrix


def Assemble_CBIE_Element(mesh,e,xi1,xi2,w1,w2,evaluation_points,wavenumber):

    px,py,pz = evaluation_points
    px=px.reshape(-1,1)
    py=py.reshape(-1,1)
    pz=pz.reshape(-1,1)
    
    Jacobian = (w1*w2).reshape(1,-1)   
    
    if e.ExactElement: 
        qx,qy,qz = e.vals(xi1,xi2)
        nq = e.normals(xi1,xi2)
        Jxi = e.J(xi1,xi2) 
        ShapeFunctions = e.shape_functions(xi1,xi2)
    else:
        # Save evaluating shape functions many times
        ShapeFunctions = e.shape_functions(xi1,xi2,1)
        qx,qy,qz = np.dot(e.P,ShapeFunctions[:,:,0].T)
        Jxi = np.zeros((np.size(xi1),2,3))
        Jxi[:,0,:] = np.dot(e.P,ShapeFunctions[:,:,1].T).T
        Jxi[:,1,:] = np.dot(e.P,ShapeFunctions[:,:,2].T).T
        Jxi = np.cross(Jxi[:,0,:],Jxi[:,1,:])
        nq = Jxi.T/np.sqrt(np.sum(Jxi**2,axis=1))
        Jxi = np.sqrt(np.sum(Jxi**2,axis=1))
        
    Jacobian *= Jxi    
    
    r = np.sqrt((qx-px)**2+(qy-py)**2+(qz-pz)**2)
    drdnq = ((qx-px)*nq[0]+(qy-py)*nq[1]+(qz-pz)*nq[2])/r

    # Evaluate kernel
    Kernel = np.exp(1j*wavenumber*r)
    Kernel *= (1j*wavenumber*r-1)
    Kernel *= drdnq
    simplefilter("ignore") # This simple filter stops warnings about divisions by 0
    Kernel /= (4*np.pi*r**2)
    simplefilter("default")
    Kernel[np.where(np.isnan(Kernel))]=0.0 # If an entry is NaN, replace by 0.0
    ######
    
    InterpolationBasis = np.repeat(ShapeFunctions[:,:,0],e.shapeFunList[0].M,axis=1) # Assumes global M !!!!
    InterpolationBasis = np.asarray(InterpolationBasis,np.complex) # Make InterpolationBasis a complex matrix
    InterpolationBasis *= np.vstack([dof(qx,qy,qz) for s in e.shapeFunList for dof in s.DegreesOfFreedomList]).T
        
    return np.dot(Kernel,Jacobian.reshape(-1,1)*InterpolationBasis)


##############################################################################   
#####                                                                    ##### 
#####                     QUADRATURE CLASS                               ##### 
#####                                                                    ##### 
############################################################################## 




class GaussQuadrature:
    """Container class containing 3D quadrature points"""
    def __init__(self,n1,n2):
        
        self.n1=n1
        self.n2=n2
        self.xi1,self.w1 = p_roots(n1)
        self.xi2,self.w2 = p_roots(n2)
            
            

        