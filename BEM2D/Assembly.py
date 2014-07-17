# Michael Peake
# Durham University

import numpy as np
from scipy.special.orthogonal import p_roots
from scipy.special import hankel1

class GaussQuadrature:
    """Container class containing 2D quadrature points"""
    def __init__(self,n):
        self.xi , self.w = p_roots(n)


class Assembler(object):
    
    def __init__(self,mesh,wavenumber,incident_wave_amplitude,direction_of_propagation):
        """Returns an object that can assemble system matrices and vectors"""
        self.mesh = mesh
        self.k = wavenumber
        self.matrixCols = mesh.ndof
        self.matrixRows = mesh.collocation_points.shape[1]
        self.numBoundaryCollocation = mesh.numBoundaryCollocation
        self.incident_wave_amplitude = incident_wave_amplitude
        self.direction_of_propagation = direction_of_propagation
        
    def Incident_Wave_Vector(self):
        return self.incident_wave_amplitude*np.exp(1j*self.k*np.dot(self.mesh.collocation_points.T,self.direction_of_propagation))
        
    def Helmholtz_CBIE_Matrix(self,integration_order=6):
        quadrature=GaussQuadrature(integration_order)
        system = self.Assemble_Helmholtz_CBIE_Matrix(quadrature)
        system[:self.numBoundaryCollocation,:self.matrixCols] += self.Assemble_Jump_Term_Matrix(0.5)
        return system


        
        
        
##############################################################################   
#####                                                                    ##### 
#####                   Helmholtz CBIE MATRIX ASSEMBLY                   ##### 
#####                                                                    ##### 
############################################################################## 
        
    def Assemble_Helmholtz_CBIE_Matrix(self,quadrature):
        
        System_Matrix = np.zeros((self.matrixRows,self.matrixCols),np.complex)
               
        # Integrate of one element at a time
        # This is quicker than doing one collocation points row at a time
        for e in self.mesh.eList:
            Element_Matrix = self.Assemble_Helmholtz_CBIE_Element(e,quadrature)
            DegreeOfFreedomMapping = [dof.ID for s in e.shapeFunList for dof in s.DegreesOfFreedomList]
            System_Matrix[:,DegreeOfFreedomMapping] += Element_Matrix
            
        return System_Matrix
        
    def Assemble_Helmholtz_CBIE_Element(self,e,quadrature):

        px,py = self.mesh.collocation_points
        px=px.reshape(-1,1)
        py=py.reshape(-1,1)
        
        # Change of integration interval to element local coordinate limits
        xi = (quadrature.xi*0.5*(e.limits[1]-e.limits[0])+0.5*(e.limits[0]+e.limits[1])).reshape(-1,)
        Jacobian = quadrature.w*0.5*(e.limits[1]-e.limits[0])
        
        # Create integration subdivisions for large elements ( > lambda/4 )
        length = e.length()
        if length > (2*np.pi/self.k)/4 :
            S = int(np.ceil(2.0*self.k*length/np.pi))
            s = np.arange(0,S).reshape(-1,1)
            xi = ((xi - e.limits[0] + (e.limits[1]-e.limits[0])*s)/S + e.limits[0]).reshape(-1,)
            Jacobian = np.repeat([Jacobian],S,axis=0).reshape(1,-1)
            Jacobian /= 1.0*S
        
        
        # It is possible to use e.vals, e.normals and e.J to find the integration
        # coordinates, their normals and Jacobians.
        # However, it is more efficient to evaluate the shape functions here
        # and use them to find qx,qy,nq,Jxi as the same shape functions are  
        # multiplied by the planewave enrichment in a few lines ...
        # 
        # However, this does not work for my PU-BEM simulations as these values
        # are hard-coded into the elements are nothing to do with the shape
        # functions.
        #
        # Hence, we check for an 'ExactElement' element to determine 
        # the best method
        
        try: e.ExactElement
        except: e.ExactElement=False
               
        if e.ExactElement: 
            qx,qy = e.vals(xi)
            nq = e.normals(xi)
            Jxi = e.J(xi)
            ShapeFunctions = e.shape_functions(xi) 
        else:
            # Shape functions and derivatives at xi coordinates
            ShapeFunctions = e.shape_functions(xi,1) 
            # Integration coordinates
            qx,qy = np.dot(e.P,ShapeFunctions[:,:,0].T)
            Jxi = np.dot(e.P,ShapeFunctions[:,:,1].T)
            nq = np.vstack([Jxi[1,:],-Jxi[0,:]]) / np.sqrt(np.sum(Jxi**2,axis=0))
            Jxi = np.sqrt(np.sum(Jxi**2,axis=0))
            
        Jacobian *= Jxi
    
        r = np.sqrt((qx-px)**2+(qy-py)**2)
        drdnq = ( (qx-px)*nq[0] + (qy-py)*nq[1] ) / r

        # Evaluate kernel
        Kernel = -1j * self.k / 4 * hankel1(1,r*self.k) * drdnq
        InterpolationBasis = np.repeat(ShapeFunctions[:,:,0],e.shapeFunList[0].M,axis=1) # Assumes global M !!!!
        InterpolationBasis = np.asarray(InterpolationBasis,np.complex) # Make InterpolationBasis a complex matrix
        InterpolationBasis *= np.vstack([dof(qx,qy) for s in e.shapeFunList for dof in s.DegreesOfFreedomList]).T
    
        return np.dot(Kernel,Jacobian.reshape(-1,1)*InterpolationBasis)






##############################################################################   
#####                                                                    ##### 
#####                   JUMP TERM MATRIX ASSEMBLY                        ##### 
#####                                                                    ##### 
############################################################################## 
            
        
    def Assemble_Jump_Term_Matrix(self,coefficient):
        Jump_Term_Matrix = np.zeros((self.numBoundaryCollocation,self.matrixCols),np.complex)
        
        for e in self.mesh.eList:
            
            Element_Matrix,CollocationPointMapping = self.Assemble_Jump_Term_Element(e)
            DegreeOfFreedomMapping = [dof.ID for s in e.shapeFunList for dof in s.DegreesOfFreedomList]
            
            ## The two mappings put the entries of Element_Matrix in the right place
            Jump_Term_Matrix[np.meshgrid(CollocationPointMapping,DegreeOfFreedomMapping)] = Element_Matrix.T

        return coefficient * Jump_Term_Matrix
        
        
    def Assemble_Jump_Term_Element(self,e):
        
        qx,qy = e.collocation_points

        # Evaluate basis for each DOF
        InterpolationBasis = e.shape_functions(e.collocationXi)[:,:,0]
        InterpolationBasis = np.repeat(InterpolationBasis,e.shapeFunList[0].M,axis=1) # Assumes global M !!!!
        InterpolationBasis = np.asarray(InterpolationBasis,np.complex) # Make InterpolationBasis a complex matrix
        InterpolationBasis *= np.vstack([dof(qx,qy) for s in e.shapeFunList for dof in s.DegreesOfFreedomList]).T

        # Row finding algorithm
        # It's a bit clunky but it finds out which collocation points (rows) are on the element being evaluated
        # by evaluating the distance between e.collocation_points and mesh.collocation_points
        # It looks for distances of less than 1e-10 (i.e. they are computationally the same
        # It seems a silly way of doing it but looking for values of zero or identical entries in the two vectors 
        # simply failed to work.
        CollocationPointMapping=[]
        for i in xrange(e.collocation_points.shape[1]):
            where=[np.where(np.abs(self.mesh.collocation_points[j]-e.collocation_points[j,i])<1e-10)[0] for j in xrange(2)]
            for j in xrange(where[0].size):
                if where[0][j] in where[1]:
                    CollocationPointMapping.append(where[0][j])
    
        return InterpolationBasis,CollocationPointMapping

       