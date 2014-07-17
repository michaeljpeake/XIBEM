# Michael Peake
# Durham University

import numpy as np
from scipy.integrate import dblquad



class StandardQuadraticElement(object):

    def __init__(self,nodal_points):
        """Contructs a three-dimensional standard quadratic element.
        
        The local coordinates xi1 and xi2 both span the interval [-1,1]."""
        nodal_points = np.asarray(nodal_points,np.float) # Make sure that nodes are stored as floats.
        if nodal_points.shape != (3,9): raise "Expected (3,9) array for nodal points"
        self.P = nodal_points
        self.m=2 # Order element - 1
        self.n=2 # For use with Bezier element structures later
        self.limits=(-1,1,-1,1) # Limits of local coordinate

    def shape_functions(self,xi1,xi2,number_of_derivatives=0):
        """
        Returns basis functions in form [i,j,d]
        where i is the point index
              j is the basis function
              d is the derivative
        d=0 --> N_j
        d=1 --> dN_j/dxi1
        d=2 --> dN_j/dxi2
        """
        if number_of_derivatives > 1: raise "Only coded for first derivative"
        # Make sure values are floating point
        xi1=np.asarray(xi1,np.float).reshape(-1,) 
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        if number_of_derivatives==1:
            N = np.zeros((np.size(xi1),9,3))
            N[:,0,1] = +0.25*xi2*(1-2*xi1)*(1-xi2)
            N[:,1,1] = xi1*xi2*(1-xi2)
            N[:,2,1] = -0.25*xi2*(1+2*xi1)*(1-xi2)
            N[:,3,1] = +0.50*(1+2*xi1)*(1-xi2)*(1+xi2)
            N[:,4,1] = +0.25*xi2*(1+2*xi1)*(1+xi2)
            N[:,5,1] = -xi1*xi2*(1+xi2)
            N[:,6,1] = -0.25*xi2*(1-2*xi1)*(1+xi2)
            N[:,7,1] = -0.50*(1-2*xi1)*(1-xi2)*(1+xi2)
            N[:,8,1] = -2*xi1*(1-xi2)*(1+xi2)
            N[:,0,2] = +0.25*xi1*(1-xi1)*(1-2*xi2)
            N[:,1,2] = -0.50*(1-xi1)*(1+xi1)*(1-2*xi2)
            N[:,2,2] = -0.25*xi1*(1+xi1)*(1-2*xi2)
            N[:,3,2] = -xi1*xi2*(1+xi1)
            N[:,4,2] = +0.25*xi1*(1+xi1)*(1+2*xi2)
            N[:,5,2] = +0.50*(1-xi1)*(1+xi1)*(1+2*xi2)
            N[:,6,2] = -0.25*xi1*(1-xi1)*(1+2*xi2)
            N[:,7,2] = xi1*xi2*(1-xi1)
            N[:,8,2] = -2*xi2*(1-xi1)*(1+xi1)
        else:
            N = np.zeros((np.size(xi1),9,1))
        N[:,0,0] = +0.25*xi1*xi2*(1-xi1)*(1-xi2)
        N[:,1,0] = -0.50*xi2*(1-xi1)*(1+xi1)*(1-xi2)
        N[:,2,0] = -0.25*xi1*xi2*(1+xi1)*(1-xi2)
        N[:,3,0] = +0.50*xi1*(1+xi1)*(1-xi2)*(1+xi2)
        N[:,4,0] = +0.25*xi1*xi2*(1+xi1)*(1+xi2)
        N[:,5,0] = +0.50*xi2*(1-xi1)*(1+xi1)*(1+xi2)
        N[:,6,0] = -0.25*xi1*xi2*(1-xi1)*(1+xi2)
        N[:,7,0] = -0.50*xi1*(1-xi1)*(1-xi2)*(1+xi2)
        N[:,8,0] = (1-xi1)*(1+xi1)*(1-xi2)*(1+xi2)
        return N

    def vals(self,xi1,xi2):
        """Returns (x,y,z) coordinates of the element at
        the point (xi1,xi2)"""
        N = self.shape_functions(xi1,xi2)
        return np.dot(self.P,N[:,:,0].T)

  
    def J(self,xi1,xi2):
        """Returns Jacobian of transformation at xi. For use
        in integration"""
        N = self.shape_functions(xi1,xi2,1)
        dfdxi1 = np.dot(self.P,N[:,:,1].T)
        dfdxi2 = np.dot(self.P,N[:,:,2].T)
        J = np.zeros((np.size(xi1),2,3))
        J[:,0,:] = dfdxi1.T
        J[:,1,:] = dfdxi2.T
        J = np.cross(J[:,0,:],J[:,1,:])
        return np.sqrt(np.sum(J**2,axis=1))

    def normals(self,xi1,xi2):
        """Returns the unit outward point normal to the surface
        of the element at the point (xi1,xi2)"""
        N = self.shape_functions(xi1,xi2,1)
        dfdxi1 = np.dot(self.P,N[:,:,1].T)
        dfdxi2 = np.dot(self.P,N[:,:,2].T)
        J = np.zeros((np.size(xi1),2,3))
        J[:,0,:] = dfdxi1.T
        J[:,1,:] = dfdxi2.T
        J = np.cross(J[:,0,:],J[:,1,:])
        magnitude = np.sqrt(np.sum(J**2,axis=1))
        return J.T/magnitude

    def area(self):
        """Calculates the area of the element"""
        return dblquad(self.J,self.limits[0],self.limits[1],lambda x:self.limits[2],lambda x:self.limits[3])[0]


class SerendipityQuadraticElement(object):

    def __init__(self,nodal_points):
        """Contructs a three-dimensional quadratic serendipity element.
        
        The local coordinates xi1 and xi2 both span the interval [-1,1]."""
        nodal_points = np.asarray(nodal_points,np.float) # Make sure that nodes are stored as floats.
        if nodal_points.shape != (3,8): raise "Expected (3,8) array for nodal points"
        self.P = nodal_points
        self.m=2 # Order element - 1
        self.n=2 # For use with Bezier element structures later
        self.limits=(-1,1,-1,1) # Limits of local coordinate

    def shape_functions(self,xi1,xi2,number_of_derivatives=0):
        """
        Returns basis functions in form [i,j,d]
        where i is the point index
              j is the basis function
              d is the derivative
        d=0 --> N_j
        d=1 --> dN_j/dxi1
        d=2 --> dN_j/dxi2
        """
        if number_of_derivatives > 1: raise "Only coded for first derivative"
        # Make sure values are floating point
        xi1=np.asarray(xi1,np.float).reshape(-1,) 
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        if number_of_derivatives==1:
            N = np.zeros((np.size(xi1),8,3))
            N[:,0,1] = 0.25*(1-xi2)*(2*xi1+xi2)
            N[:,1,1] = -xi1*(1-xi2)
            N[:,2,1] = 0.25*(1-xi2)*(2*xi1-xi2)
            N[:,3,1] = 0.5*(1-xi2**2)
            N[:,4,1] = 0.25*(1+xi2)*(2*xi1+xi2)
            N[:,5,1] = -xi1*(1+xi2)
            N[:,6,1] = 0.25*(1+xi2)*(2*xi1-xi2)
            N[:,7,1] = -0.5*(1-xi2**2)
            N[:,0,2] = 0.25*(1-xi1)*(xi1+2*xi2)
            N[:,1,2] = -0.5*(1-xi1**2)
            N[:,2,2] = 0.25*(1+xi1)*(-xi1+2*xi2)
            N[:,3,2] = -xi2*(1+xi1)
            N[:,4,2] = 0.25*(1+xi1)*(xi1+2*xi2)
            N[:,5,2] = 0.5*(1-xi1**2)
            N[:,6,2] = 0.25*(1-xi1)*(-xi1+2*xi2)
            N[:,7,2] = -xi2*(1-xi1)
        else:
            N = np.zeros((np.size(xi1),8,1))
        N[:,0,0] = 0.25*(1-xi1)*(1-xi2)*(-xi1-xi2-1)
        N[:,1,0] = 0.5*(1-xi1**2)*(1-xi2)
        N[:,2,0] = 0.25*(1+xi1)*(1-xi2)*(xi1-xi2-1)
        N[:,3,0] = 0.5*(1+xi1)*(1-xi2**2)
        N[:,4,0] = 0.25*(1+xi1)*(1+xi2)*(xi1+xi2-1)
        N[:,5,0] = 0.5*(1-xi1**2)*(1+xi2)
        N[:,6,0] = 0.25*(1-xi1)*(1+xi2)*(-xi1+xi2-1)
        N[:,7,0] = 0.5*(1-xi1)*(1-xi2**2)
        return N

    def vals(self,xi1,xi2):
        """Returns (x,y,z) coordinates of the element at
        the point (xi1,xi2)"""
        N = self.shape_functions(xi1,xi2)
        return np.dot(self.P,N[:,:,0].T)

    def J(self,xi1,xi2):
        """Returns Jacobian of transformation at xi. For use
        in integration"""
        N = self.shape_functions(xi1,xi2,1)
        dfdxi1 = np.dot(self.P,N[:,:,1].T)
        dfdxi2 = np.dot(self.P,N[:,:,2].T)
        J = np.zeros((np.size(xi1),2,3))
        J[:,0,:] = dfdxi1.T
        J[:,1,:] = dfdxi2.T
        J = np.cross(J[:,0,:],J[:,1,:])
        return np.sqrt(np.sum(J**2,axis=1))

    def normals(self,xi1,xi2):
        """Returns the unit outward point normal to the surface
        of the element at the point (xi1,xi2)"""
        N = self.shape_functions(xi1,xi2,1)
        dfdxi1 = np.dot(self.P,N[:,:,1].T)
        dfdxi2 = np.dot(self.P,N[:,:,2].T)
        J = np.zeros((np.size(xi1),2,3))
        J[:,0,:] = dfdxi1.T
        J[:,1,:] = dfdxi2.T
        J = np.cross(J[:,0,:],J[:,1,:])
        magnitude = np.sqrt(np.sum(J**2,axis=1))
        return J.T/magnitude

    def area(self):
        """Calculates the area of the element"""
        return dblquad(self.J,self.limits[0],self.limits[1],lambda x:self.limits[2],lambda x:self.limits[3])[0]


class Domain(object):
    
    def __init__(self,list_of_elements):
        """Defines a closed domain consisting of two or more elements. I.e. it
        defines one scatterer."""
        if len(list_of_elements) < 2: raise "A domain requires more than one element"
        self.eList=list_of_elements
        self.numElements = len(list_of_elements)

    def area(self):
        """Returns the area of a domains boundary"""
        return sum([e.area() for e in self.eList])
        

class Mesh(object):
    """Define a mesh of one or more domains"""

    def __init__(self,list_of_domains):
        self.dList = list_of_domains
        self.numDomains = len(self.dList)
        self.eList = [e for d in list_of_domains for e in d.eList]
        self.numElements = len(self.eList)
        






        
#############################################################
###                    EXACT ELEMENTS                     ###
#############################################################

class TorusSegment_Standard(StandardQuadraticElement):
    """Define an analytical arc element"""

    def __init__(self,majorradius,minorradius,theta1limits,theta2limits):
        """Returns an analytical torus segment assuming torus centred at origin"""
        self.R=majorradius
        self.r=minorradius
        self.theta1limits=theta1limits
        self.theta2limits=theta2limits
        P=self.vals(np.array([-1.0,0,1,1,1,0,-1,-1,0]),np.array([-1.0,-1,-1,0,1,1,1,0,0]))
        super(TorusSegment_Standard,self).__init__(P)


    def vals(self,xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xi1 = (xi1*0.5*(self.theta1limits[1]-self.theta1limits[0])+0.5*(self.theta1limits[0]+self.theta1limits[1])).reshape(-1,)
        xi2 = (xi2*0.5*(self.theta2limits[1]-self.theta2limits[0])+0.5*(self.theta2limits[0]+self.theta2limits[1])).reshape(-1,)
        x = np.cos(xi1)*(self.R+self.r*np.cos(xi2))
        y = np.sin(xi1)*(self.R+self.r*np.cos(xi2))
        z = self.r*np.sin(xi2)
        return np.vstack([x,y,z])

    def normals(self,xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xi1 = (xi1*0.5*(self.theta1limits[1]-self.theta1limits[0])+0.5*(self.theta1limits[0]+self.theta1limits[1])).reshape(-1,)
        xi2 = (xi2*0.5*(self.theta2limits[1]-self.theta2limits[0])+0.5*(self.theta2limits[0]+self.theta2limits[1])).reshape(-1,)
        J = np.zeros((np.size(xi1),2,3))
        J[:,0,0] = -np.sin(xi1)*(self.R+self.r*np.cos(xi2))
        J[:,0,1] = np.cos(xi1)*(self.R+self.r*np.cos(xi2))
        J[:,0,2] = np.zeros(xi1.shape)
        J[:,1,0] = -self.r*np.cos(xi1)*np.sin(xi2)
        J[:,1,1] = -self.r*np.sin(xi1)*np.sin(xi2)
        J[:,1,2] = self.r*np.cos(xi2)
        
        J[:,0,:] *= 0.5*(self.theta1limits[1]-self.theta1limits[0])
        J[:,1,:] *= 0.5*(self.theta2limits[1]-self.theta2limits[0])
        
        J = np.cross(J[:,0,:],J[:,1,:])
        magnitude = np.sqrt(np.sum(J**2,axis=1))
        
        return J.T/magnitude
        
    def J(self,xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xi1 = (xi1*0.5*(self.theta1limits[1]-self.theta1limits[0])+0.5*(self.theta1limits[0]+self.theta1limits[1])).reshape(-1,)
        xi2 = (xi2*0.5*(self.theta2limits[1]-self.theta2limits[0])+0.5*(self.theta2limits[0]+self.theta2limits[1])).reshape(-1,)
        J = np.zeros((np.size(xi1),2,3))
        J[:,0,0] = -np.sin(xi1)*(self.R+self.r*np.cos(xi2))
        J[:,0,1] = np.cos(xi1)*(self.R+self.r*np.cos(xi2))
        J[:,0,2] = np.zeros(xi1.shape)
        J[:,1,0] = -self.r*np.cos(xi1)*np.sin(xi2)
        J[:,1,1] = -self.r*np.sin(xi1)*np.sin(xi2)
        J[:,1,2] = self.r*np.cos(xi2)
        
        J[:,0,:] *= 0.5*(self.theta1limits[1]-self.theta1limits[0])
        J[:,1,:] *= 0.5*(self.theta2limits[1]-self.theta2limits[0])
        
        J = np.cross(J[:,0,:],J[:,1,:])
        
        return np.sqrt(np.sum(J**2,axis=1))


class TorusSegment_Serendipity(SerendipityQuadraticElement):
    """Define an analytical arc element"""

    def __init__(self,majorradius,minorradius,theta1limits,theta2limits):
        """Returns an analytical torus segment assuming torus centred at origin"""
        self.R=majorradius
        self.r=minorradius
        self.theta1limits=theta1limits
        self.theta2limits=theta2limits
        P=self.vals(np.array([-1.0,0,1,1,1,0,-1,-1]),np.array([-1.0,-1,-1,0,1,1,1,0]))
        super(TorusSegment_Serendipity,self).__init__(P)

    def vals(self,xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xi1 = (xi1*0.5*(self.theta1limits[1]-self.theta1limits[0])+0.5*(self.theta1limits[0]+self.theta1limits[1])).reshape(-1,)
        xi2 = (xi2*0.5*(self.theta2limits[1]-self.theta2limits[0])+0.5*(self.theta2limits[0]+self.theta2limits[1])).reshape(-1,)
        x = np.cos(xi1)*(self.R+self.r*np.cos(xi2))
        y = np.sin(xi1)*(self.R+self.r*np.cos(xi2))
        z = self.r*np.sin(xi2)
        return np.vstack([x,y,z])

    def normals(self,xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xi1 = (xi1*0.5*(self.theta1limits[1]-self.theta1limits[0])+0.5*(self.theta1limits[0]+self.theta1limits[1])).reshape(-1,)
        xi2 = (xi2*0.5*(self.theta2limits[1]-self.theta2limits[0])+0.5*(self.theta2limits[0]+self.theta2limits[1])).reshape(-1,)
        J = np.zeros((np.size(xi1),2,3))
        J[:,0,0] = -np.sin(xi1)*(self.R+self.r*np.cos(xi2))
        J[:,0,1] = np.cos(xi1)*(self.R+self.r*np.cos(xi2))
        J[:,0,2] = np.zeros(xi1.shape)
        J[:,1,0] = -self.r*np.cos(xi1)*np.sin(xi2)
        J[:,1,1] = -self.r*np.sin(xi1)*np.sin(xi2)
        J[:,1,2] = self.r*np.cos(xi2)
        J[:,0,:] *= 0.5*(self.theta1limits[1]-self.theta1limits[0])
        J[:,1,:] *= 0.5*(self.theta2limits[1]-self.theta2limits[0])
        J = np.cross(J[:,0,:],J[:,1,:])
        magnitude = np.sqrt(np.sum(J**2,axis=1))
        return J.T/magnitude
        
    def J(self,xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xi1 = (xi1*0.5*(self.theta1limits[1]-self.theta1limits[0])+0.5*(self.theta1limits[0]+self.theta1limits[1])).reshape(-1,)
        xi2 = (xi2*0.5*(self.theta2limits[1]-self.theta2limits[0])+0.5*(self.theta2limits[0]+self.theta2limits[1])).reshape(-1,)
        J = np.zeros((np.size(xi1),2,3))
        J[:,0,0] = -np.sin(xi1)*(self.R+self.r*np.cos(xi2))
        J[:,0,1] = np.cos(xi1)*(self.R+self.r*np.cos(xi2))
        J[:,0,2] = np.zeros(xi1.shape)
        J[:,1,0] = -self.r*np.cos(xi1)*np.sin(xi2)
        J[:,1,1] = -self.r*np.sin(xi1)*np.sin(xi2)
        J[:,1,2] = self.r*np.cos(xi2)
        J[:,0,:] *= 0.5*(self.theta1limits[1]-self.theta1limits[0])
        J[:,1,:] *= 0.5*(self.theta2limits[1]-self.theta2limits[0])
        J = np.cross(J[:,0,:],J[:,1,:])
        return np.sqrt(np.sum(J**2,axis=1))





        

                    
if  __name__ == "__main__":

    test = TorusSegment_Standard(1,0.5,[0,0.5*np.pi],[0,-0.5*np.pi])
    
