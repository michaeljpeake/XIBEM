# Michael Peake
# Durham University

import numpy as np
from scipy.integrate import quad




class QuadraticElement(object):
    
    def __init__(self,nodal_points):
        """Constructs a 2D, continuous, Lagrangian quadratic element
        with local coordinate xi in [-1,1]"""
        P = np.asarray(nodal_points,np.float) # Make sure that nodes are stored as floats.
        if P.shape != (2,3): raise "Expected (2,3) array for nodal points"
        self.P = nodal_points
        self.limits=(-1.0,1.0) # Limits of local coordinate

    def shape_functions(self,xi,number_of_derivatives=0):
        """Returns the three basis functions on the element
        in the form [i,j,d] where:
            i: xi coordinate index
            j: shape function index
            d: derivative"""
        if number_of_derivatives > 1: raise "Only coded for first derivative" 
        xi = np.asarray(xi,np.float).reshape(-1,) # Ensure xi variable is float
        if number_of_derivatives == 1:
            N = np.zeros((xi.size,3,2))
            N[:,0,1] = xi - 0.5
            N[:,1,1] = - 2*xi
            N[:,2,1] = xi + 0.5
        else:
            N = np.zeros((xi.size,3,1))
        N[:,0,0] =  0.5 * ( xi**2 - xi )
        N[:,1,0] = - xi**2 + 1
        N[:,2,0] = 0.5 * ( xi**2 + xi )
        return N
        
    def vals(self,xi):
        """Returns (x,y) coordinates of points at xi. Xi can be
        one value or many."""
        N = self.shape_functions(xi)
        return np.dot(self.P,N[:,:,0].T)
 
    def J(self,xi):
        """Returns Jacobian of transformation at xi. For use
        in integration"""
        J = self.shape_functions(xi,1)
        J = np.dot(self.P,J[:,:,1].T)
        return np.sqrt(np.sum(J**2,axis=0))

    def normals(self,xi):
        """Returns unit normal vector at xi"""
        J = self.shape_functions(xi,1)
        J = np.dot(self.P,J[:,:,1].T)
        n = np.vstack([J[1,:],-J[0,:]])
        return n / np.sqrt(np.sum(J**2,axis=0))

    def length(self):
        """Integrates the Jacobian of the limits of the element to
        find the length"""
        return quad(self.J,self.limits[0],self.limits[1])[0]        
        
    def TrigonometricShapeFunctions(self,xi,num_derivatives=0):
        """Returns the three trigonomtric basis functions on 
        the element in the form [i,j,d] where:
            i: xi coordinate index
            j: basis function index
            d: derivative.
            
        Overwrite shape_functions with this for trig elements;
        e.g. e.shape_functions = e.TrigonometricShapeFunctions """
            
        xi = np.asarray(xi,np.float)
        if num_derivatives == 1:
            N = np.zeros((xi.size,3,2))
            N[:,0,1] = 0.25*np.pi*np.sin(np.pi*xi) - 0.25*np.pi*np.cos(np.pi/2*xi)
            N[:,1,1] = -0.5*np.pi*np.sin(np.pi*xi)
            N[:,2,1] = 0.25*np.pi*np.sin(np.pi*xi) + 0.25*np.pi*np.cos(np.pi/2*xi)
        else:
            N = np.zeros((xi.size,3,1))
        N[:,0,0] =  -0.25*np.cos(np.pi*xi) - 0.5*np.sin(np.pi/2*xi) + 0.25
        N[:,1,0] = 0.5*np.cos(np.pi*xi) + 0.5
        N[:,2,0] = -0.25*np.cos(np.pi*xi) + 0.5*np.sin(np.pi/2*xi) + 0.25
        
        return N
        



class Domain(object):
    
    def __init__(self,list_of_elements):
        """Defines a closed domain consisting of two or more elements. I.e. it
        defines one scatterer."""
        if len(list_of_elements) < 2: raise "A domain requires more than one element"
        self.eList=list_of_elements
        self.numElements = len(list_of_elements)

    def length(self):
        """Returns the length of a domains boundary"""
        return sum([e.length() for e in self.eList])
        


     
class Mesh(object):

    def __init__(self,list_of_domains):
        """Defines a mesh, a list of domains. Shape functions are given IDs."""
        self.dList = list_of_domains
        self.numDomains = len(list_of_domains)
        self.eList = [e for d in self.dList for e in d.eList]
        self.numElements = len(self.eList)
        













#############################################################
###                    EXACT ELEMENTS                     ###
#############################################################
    
# These are used for PU-BEM. They register as quadratic elements
# but they overwrite 'vals', 'normals' and 'J' to provide exact 
# geometry and Jacobian. They still contain the same shape
# functions for the field variable interpolation however.
    
class Line(QuadraticElement):

    def __init__(self,a,b):
        """Returns an analytical line element between a and b. Registers
        as a quadratic element."""
        self.ExactElement = True
        self.a = a
        self.b = b
        P=self.vals(np.array([-1.0,0.0,1.0]))
        super(Line,self).__init__(P)

    def vals(self,xi):
        x = self.a[0] + 0.5*(xi+1)*(self.b[0]-self.a[0])
        y = self.a[1] + 0.5*(xi+1)*(self.b[1]-self.a[1])
        return np.vstack([x,y])

    def normals(self,xi):
        dx = 0*xi + 0.5*(self.b[0]-self.a[0])
        dy = 0*xi + 0.5*(self.b[1]-self.a[1])
        J = 0.5*(self.limits[1]-self.limits[0])*np.vstack([dx,dy])
        n = np.vstack([J[1,:],-J[0,:]])
        return n / np.sqrt(np.sum(J**2,axis=0))

    def J(self,xi):
        dx = 0*xi + 0.5*(self.b[0]-self.a[0])
        dy = 0*xi + 0.5*(self.b[1]-self.a[1])
        J = 0.5*(self.limits[1]-self.limits[0])*np.vstack([dx,dy])
        return np.sqrt(np.sum(J**2,axis=0))




class CircularArc(QuadraticElement):

    def __init__(self,X,Y,startAngle,endAngle,radius):
        """Returns an analytical circular arc element centred at (X,Y) and
        that sweeps from startAngle to endAngle. Note: angles are in radians. 
        It registers as a quadratic element."""
        self.ExactElement = True
        self.x0=X
        self.y0=Y
        self.a=startAngle
        self.b=endAngle
        self.r=radius
        P=self.vals(np.array([-1.0,0.0,1.0]))
        super(CircularArc,self).__init__(P)

    def vals(self,xi):
        x = self.x0 + self.r * np.cos( self.a + 0.5*(xi+1)*(self.b-self.a) )
        y = self.y0 + self.r * np.sin( self.a + 0.5*(xi+1)*(self.b-self.a) )
        return np.vstack([x,y])

    def normals(self,xi):
        dx = -0.5*(self.b-self.a) * self.r*np.sin(self.a+0.5*(xi+1)*(self.b-self.a) )
        dy = +0.5*(self.b-self.a) * self.r*np.cos(self.a+0.5*(xi+1)*(self.b-self.a) )
        J = 0.5*(self.limits[1]-self.limits[0])*np.vstack([dx,dy])
        n = np.vstack([J[1,:],-J[0,:]])
        return n / np.sqrt(np.sum(J**2,axis=0))

    def J(self,xi):
        dx = -0.5*(self.b-self.a) * self.r*np.sin(self.a+0.5*(xi+1)*(self.b-self.a) )
        dy = +0.5*(self.b-self.a) * self.r*np.cos(self.a+0.5*(xi+1)*(self.b-self.a) )
        J = 0.5*(self.limits[1]-self.limits[0])*np.vstack([dx,dy])
        return np.sqrt(np.sum(J**2,axis=0))




class Ellipse(QuadraticElement):

    def __init__(self,X,Y,startAngle,endAngle,xRadius,yRadius):
        """Returns an analytical ellipse element"""
        self.ExactElement = True
        self.x0 = X
        self.y0 = Y
        self.a = startAngle
        self.b = endAngle
        self.r1 = xRadius
        self.r2 = yRadius
        P=self.vals(np.array([-1.0,0.0,1.0]))
        super(Ellipse,self).__init__(P)

    def vals(self,xi):
        x = self.x0 + self.r1 * np.cos( self.a + 0.5*(xi+1)*(self.b-self.a) )
        y = self.y0 + self.r2 * np.sin( self.a + 0.5*(xi+1)*(self.b-self.a) )
        return np.vstack([x,y])
        
    def normals(self,xi):
        dx = -0.5*(self.b-self.a) * self.r1*np.sin(self.a+0.5*(xi+1)*(self.b-self.a) )
        dy = +0.5*(self.b-self.a) * self.r2*np.cos(self.a+0.5*(xi+1)*(self.b-self.a) )
        J = 0.5*(self.limits[1]-self.limits[0])*np.vstack([dx,dy])
        n = np.vstack([J[1,:],-J[0,:]])
        return n / np.sqrt(np.sum(J**2,axis=0))

    def J(self,xi):
        dx = -0.5*(self.b-self.a) * self.r1*np.sin(self.a+0.5*(xi+1)*(self.b-self.a) )
        dy = +0.5*(self.b-self.a) * self.r2*np.cos(self.a+0.5*(xi+1)*(self.b-self.a) )
        J = 0.5*(self.limits[1]-self.limits[0])*np.vstack([dx,dy])
        return np.sqrt(np.sum(J**2,axis=0))



