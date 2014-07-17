# Michael Peake
# Durham University

try: import NURBSinC
except: print "Could not import NURBSinC. NURBS and Bezier functions will not work."
from scipy.linalg import pinv

import numpy as np
from scipy.integrate import dblquad


class BezierElement(object):
    """Defines a 3D rational Bezier element defined by homogeneous
    control points. Control points should be four rows: [ x*w , y*w , z*w, w ]"""

    def __init__(self,HomogeneousControlPoints):
        HomogeneousControlPoints = np.asarray(HomogeneousControlPoints,np.float)
        if HomogeneousControlPoints.shape[0] != 3: raise "Control points should have three rows: [x*w,y*w,w]"
        
        # This takes some jiggery pokery to the the control points in a nice format
        self.Pw = np.zeros((4,HomogeneousControlPoints.shape[0],HomogeneousControlPoints.shape[1]))
        for i in xrange(HomogeneousControlPoints.shape[0]):
            for j in xrange(HomogeneousControlPoints.shape[1]):
                self.Pw[:,i,j] = HomogeneousControlPoints[i,j,:]
        self.Pw = self.Pw.reshape(4,-1)
        #########
        
        self.P = self.Pw[:3]/self.Pw[-1]
        self.w = self.Pw[-1]
        self.n = self.Pw.shape[1]-1
        self.limits=(0.0,1.0,0.0,1.0)       

        self.wmat = HomogeneousControlPoints[:,:,-1]
        self.w = self.wmat.reshape(-1,)
        self.n = HomogeneousControlPoints.shape[0]-1
        self.m = HomogeneousControlPoints.shape[1]-1
        self.limits=(0,1,0,1)

    def shape_functions(self,xi1,xi2,number_of_derivatives=0):
        
        if number_of_derivatives > 1: raise "Only coded for first derivative"
        # Make sure values are floating point
        xi1=np.asarray(xi1,np.float).reshape(-1,) 
        xi2=np.asarray(xi2,np.float).reshape(-1,)        
        
        num = np.size(xi1)
        
        Bi = NURBSinC.multiAllBernsteinDers(self.n,xi1,number_of_derivatives)
        Bj = NURBSinC.multiAllBernsteinDers(self.m,xi2,number_of_derivatives)  

        N = Bi[:,0].reshape(num,-1,1) * Bj[:,0].reshape(num,1,-1) * self.wmat
        W = np.sum(np.sum(N,1),1).reshape(-1,1,1)

        if number_of_derivatives==1:
                            
            R = np.zeros((num,(self.n+1)*(self.m+1),3))
            
            Nu = Bi[:,1].reshape(num,-1,1) * Bj[:,0].reshape(num,1,-1) * self.wmat
            Wu = np.sum(np.sum(Nu,1),1).reshape(-1,1,1)
            Nv = Bi[:,0].reshape(num,-1,1) * Bj[:,1].reshape(num,1,-1) * self.wmat
            Wv = np.sum(np.sum(Nv,1),1).reshape(-1,1,1)

            R[:,:,0] = (N/W).reshape(num,-1)
            # From formula S_alpha(u,v) in NURBS books pg 136
            R[:,:,1] = ((Nu*W-N*Wu)/(W**2)).reshape(num,-1)
            R[:,:,2] = ((Nv*W-N*Wv)/(W**2)).reshape(num,-1)
            
        else:
            R = np.zeros((num,(self.n+1)*(self.m+1),1))
            R[:,:,0] = (N/W).reshape(num,-1)
            
        return R

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
        




class NURBS_Surface(object):
    """ NURBS surface """
    def __init__(self,U_knotvec,U_degree,V_knotvec,V_degree,HomogeneousControlPoints):
        self.p = U_degree
        self.U = U_knotvec
        self.r = self.U.size - 1
        self.n = self.r - self.p - 1
        self.pspans = np.unique(U_knotvec)
        self.q = V_degree
        self.V = V_knotvec
        self.s = self.V.size - 1
        self.m = self.s - self.q - 1
        self.vspans = np.unique(V_knotvec)
        self.Pw = HomogeneousControlPoints
        self.P = (self.Pw[:,:,:3].T/self.Pw[:,:,3].T).T
        self.w = self.Pw[:,:,3]

    def vals(self,u,v):
        """Return coordinates of NURBS surface at (u,v)"""
        u=np.asarray(u,np.float).reshape(-1,)
        v=np.asarray(v,np.float).reshape(-1,)
        return NURBSinC.multiSurfacePoint(self.n,self.p,self.U,self.m,self.q,self.V,self.Pw,u,v)

    def NURBSbasis(self,uv,i=None,j=None):
        """Evaluate one basis function given uv, a (2,-1) matrix"""
        if uv.shape[1]>1:
            if i!=None and j!=None:
                uspan = NURBSinC.multiFindSpan(self.n,self.p,uv[0],self.U)
                Nu = NURBSinC.multiBasisFuns(uspan,uv[0],self.p,self.U)
                vspan = NURBSinC.multiFindSpan(self.m,self.q,uv[1],self.V)
                Nv = NURBSinC.multiBasisFuns(vspan,uv[1],self.q,self.V)
                denominator = np.zeros(uv.shape[1],)
                for k in xrange(self.n+1):
                    for l in xrange(self.m+1):
                        denominator += Nu[:,k]*Nv[:,l]*self.w[k,l]
                return Nu[:,i]*Nv[:,j]*self.w[i,j] / denominator
            else:
                return NURBSinC.multiSurfaceBasis(uv,self.p,self.n,self.U,self.q,self.m,self.V,self.w)
        else:
            print "Need to 'NURBSbasis' for single point"

    def decompose(self):
        """Decompose NURBS surface into rational Bezier patches"""
        return NURBSinC.DecomposeSurface(self.n,self.p,self.U,self.m,self.q,self.V,self.Pw)
        
    def refineknotvector(self,XI,knotvec_choice):
        """Insert the knots contained in XI into U or V"""
        XI = np.asarray(XI,np.float).reshape(-1,)
        if knotvec_choice == 'U':
            if np.min(XI)<=np.min(self.U):
                raise "Knots cannot be inserted: some elements too small"
            if np.max(XI)>=np.max(self.U):
                raise "Knots cannot be inserted: some elements too large"
            if np.any([np.equal(xi,self.U).sum()+np.equal(xi,XI).sum()>self.p for xi in np.unique(XI)]):
                raise "Knots cannot be inserted: some multiplicities too large"
            self.U,self.V,self.Pw = NURBSinC.RefineKnotVectSurface(self.n,self.p,self.U,self.m,self.q,self.V,self.Pw,XI,0)
            self.r = self.U.shape[0] - 1
            self.n = self.r - self.p - 1
            self.pspans = np.unique(self.U)
            
        if knotvec_choice == 'V':
            if np.min(XI)<=np.min(self.V):
                raise "Knots cannot be inserted: some elements too small"
            if np.max(XI)>=np.max(self.V):
                raise "Knots cannot be inserted: some elements too large"
            if np.any([np.equal(xi,self.V).sum()+np.equal(xi,XI).sum()>self.q for xi in np.unique(XI)]):
                raise "Knots cannot be inserted: some multiplicities too large"            
            self.V,self.U,self.Pw = NURBSinC.RefineKnotVectSurface(self.m,self.q,self.V,self.n,self.p,self.U,np.transpose(self.Pw,(1,0,2)),XI,0)
            self.Pw = np.transpose(self.Pw,(1,0,2))        
            
            self.s = self.V.shape[0] - 1
            self.m = self.s - self.q - 1
            self.qspans = np.unique(self.V)
            
        self.P = (self.Pw[:,:,:3].T/self.Pw[:,:,3].T).T
        self.w = self.Pw[:,:,3]           
            


def PointToLine(S,T,Q):
    """Finds P, the orthogonal projection of Q on the line ST"""
    # See p738 of logbook
    # Should take all vectors regardless of shape and return
    # result in the same shape

    shape=Q.shape
    S=S.reshape(-1,)
    T=T.reshape(-1,)
    Q=Q.reshape(-1,)

    c = np.float(np.dot(T,Q-S)) / np.dot(T,T)

    return (c*T + S).reshape(shape)

def VecNormalise(X):
    """Returns the magnitude of vector"""
    X=X.reshape(-1,)
    return np.sqrt(np.dot(X,X))

def VecCrossProd(X,Y):
    """Returns X cross Y"""
    shape=X.shape
    X=X.reshape(-1,)
    Y=Y.reshape(-1,)
    return np.cross(X,Y).reshape(shape)
    
def Intersect3DLines(P0,T0,P1,T1):
    """Finds the intersection points of two lines"""
    # See p736 of logbook
    # P0 - origin of first line
    # T0 - direction vector of first line
    # P1 - origin of second line
    # T1 - direction vector of second line

    A = np.array([T0[0],-T1[0],
                  T0[1],-T1[1],
                  T0[2],-T1[2]]).reshape(3,2)

    b = np.array([P1[0]-P0[0],P1[1]-P0[1],P1[2]-P0[2]]).reshape(3,1)

    t = np.dot(pinv(A),b)

    intersect = np.zeros((3,1))
    intersect[0] = P0[0]+t[0]*T0[0]
    intersect[1] = P0[1]+t[0]*T0[1]
    intersect[2] = P0[2]+t[0]*T0[2]

    return intersect.reshape(P0.shape)

def MakeRevolvedSurface(S,T,theta,q,V,m,Pj,wj):
    """Creates a NURBS surface of revolution"""
    S = np.array(S,np.float)
    T = np.array(T,np.float)
    Pj = np.array(Pj,np.float)
    wj = np.array(wj,np.float)
    
    if theta <= np.pi/2:
        narcs=1
        U = np.array([0,0,0,1,1,1],dtype=np.float)
    elif theta <= np.pi:
        narcs=2
        U = np.array([0,0,0,0.5,0.5,1,1,1])
    elif theta <= 3*np.pi/2:
        narcs=3
        U = np.array([0,0,0,1./3.,1./3.,2./3.,2./3.,1,1,1])
    else:
        narcs=4
        U = np.array([0,0,0,0.25,0.25,0.5,0.5,0.75,0.75,1,1,1])

    dtheta = theta/narcs
    wm = np.cos(dtheta/2.0)
    angle = np.linspace(0,narcs*dtheta,narcs+1)
    cosines = np.cos(angle)
    sines = np.sin(angle)

    Pij = np.zeros((2*narcs+1,m+1,3))
    wij = np.zeros((2*narcs+1,m+1))

    for j in xrange(m+1):
        # Loop and compute each u row of ctrl pts and weights

        Oh = PointToLine(S,T,Pj[j])
        X = Pj[j] - Oh
        r = VecNormalise(X)
        r=1.0
        Y = VecCrossProd(T,X)
        Pij[0][j] = P0 = Pj[j]
        wij[0][j] = wj[j]
        T0 = Y
        index = 0
        for i in xrange(1,narcs+1):
            P2 = Oh + r*cosines[i]*X + r*sines[i]*Y
            Pij[index+2][j] = P2
            wij[index+2][j] = wj[j]
            T2 = -sines[i]*X + cosines[i]*Y
            Pij[index+1][j] = Intersect3DLines(P0,T0,P2,T2)
            wij[index+1][j] = wm*wj[j]
            index +=2
            if i < narcs:
                P0=P2
                T0=T2

    Pw = np.zeros((Pij.shape[0],Pij.shape[1],4))
    for i in xrange(3):
        Pw[:,:,i] = Pij[:,:,i]*wij
    Pw[:,:,3] = wij

    return NURBS_Surface(U,2,V,q,Pw)
    
