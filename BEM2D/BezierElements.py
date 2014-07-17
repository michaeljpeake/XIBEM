# Michael Peake
# Durham University

try: import NURBSinC
except: print "Could not import NURBSinC. NURBS and Bezier functions will not work."

import numpy as np
from scipy.integrate import quad




class BezierElement(object):
    
    def __init__(self,HomogeneousControlPoints):
        """Defines a 2D rational Bezier element defined by homogeneous
        control points. Control points should be three rows: [ x*w , y*w , w ]"""
        HomogeneousControlPoints = np.asarray(HomogeneousControlPoints,np.float)
        if HomogeneousControlPoints.shape[0] != 3: raise "Control points should have three rows: [x*w,y*w,w]"
        self.Pw = HomogeneousControlPoints
        self.P = self.Pw[:2]/self.Pw[-1]
        self.w = self.Pw[-1]
        self.n = self.Pw.shape[1]-1
        self.limits=(0.0,1.0)
        
    def shape_functions(self,xi,number_of_derivatives=0):
        """Returns the basis functions on the element
        in the form [i,j,d] where:
            i: xi coordinate index
            j: Bezier function index
            d: derivative"""
        if number_of_derivatives > 1: raise "Only coded for first derivative"
        xi=np.asarray(xi,np.float).reshape(-1,)
        if number_of_derivatives==1:
            B = NURBSinC.multiRationalBernsteinDers(self.n,self.Pw.T,xi,1)
        else:
            B = np.zeros((xi.size,self.n+1,1))
            B[:,:,0] = NURBSinC.multiRationalBernstein(self.n,self.Pw.T,xi)
        return B

    def vals(self,xi):
        """Returns (x,y) coordinates of points at xi. Xi can be
        one value or many."""
        B = self.shape_functions(xi)
        return np.dot(self.P,B[:,:,0].T)

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




class Domain(object):
    
    def __init__(self,list_of_elements):
        """Defines a closed domain consisting of two or more Bezier curves."""
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
        



class NURBSCurve(object):
    """Returns a closed NURBS curve. Knot vector will be made to be in [0,1)"""
    def __init__(self,degree,Knot_Vector,Homogeneous_Control_Points):
        self.p = degree
        Knot_Vector= np.asarray(Knot_Vector,np.float)
        Knot_Vector = (Knot_Vector-Knot_Vector[0])/Knot_Vector[-1]
        self.U = Knot_Vector
        self.spans = np.unique(Knot_Vector)
        self.Pw = Homogeneous_Control_Points
        self.P = (self.Pw[:,:2].T/self.Pw[:,2].T).T
        self.w = self.Pw[:,2]
        self.n = Homogeneous_Control_Points.shape[0]-1
        
    def NURBS_basis_functions(self,xi,num_derivatives=0):
        xi = np.asarray(xi,np.float).reshape(-1,)
        spans = NURBSinC.multiFindSpan(self.n,self.p,xi,self.U)
        if num_derivatives==1:
            N = NURBSinC.multidersNURBSbasis(spans,xi,self.p,1,self.U,self.w)
        else:
            N = np.zeros((xi.size,1,self.n+1))
            N[:,0,:] = NURBSinC.multiNURBSbasis(spans,xi,self.p,self.U,self.w)
        return N

    def vals(self,xi):
        R = self.NURBS_basis_functions(xi)
        return np.dot(R[:,0,:],self.P).T
        
    def J(self,xi):
        R = self.NURBS_basis_functions(xi,1)
        J = np.dot(R[:,1,:],self.P).T
        return np.sqrt(np.sum(J(xi)**2,axis=0))

    def normals(self,xi):
        R = self.NURBS_basis_functions(xi,1)
        J = np.dot(R[:,1,:],self.P).T
        n = np.vstack([J[1,:],-J[0,:]])
        return n / np.sqrt(np.sum(J**2,axis=0))

    def length(self,xi0=0,xi1=1):
        """Find length of curve in the interval [xi0,xi1]"""
        return quad(self.J,xi0,xi1)[0]
        
    def refineknotvector(self,XI):
        XI = np.asarray(XI,np.float).reshape(-1,)
        """Refine the curve knot vector"""
        if np.min(XI)<=np.min(self.U):
            raise "Knots cannot be inserted: some values too small"
        if np.max(XI)>=np.max(self.U):
            raise "Knots cannot be inserted: some values too large"
        if np.any([np.equal(xi,self.U).sum()+np.equal(xi,XI).sum()>self.p for xi in np.unique(XI)]):
            raise "Knots cannot be inserted: some multiplicities too high"
        self.n,self.U,self.Pw = NURBSinC.RefineKnotVectCurve(self.n,self.p,self.U,self.Pw,XI)
        self.P,self.w = (self.Pw[:,:2].T/self.Pw[:,2].T).T,self.Pw[:,2]

    def decompose_into_Bezier_segments(self):
        nb,Qw = NURBSinC.DecomposeCurve(self.n,self.p,self.U,self.Pw)
        return nb,Qw



 
    

    
