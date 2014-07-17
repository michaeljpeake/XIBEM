import numpy as np
cimport numpy as np


##############################################################################
#####                                                                    #####
#####                              FindSpan                              #####
#####                                                                    #####
##############################################################################

def FindSpan(int n, int p, double u, np.ndarray[np.float_t] U):
    """Determine the knot span index
    # Input:
    # n: number of basis functions - 1 (n=m-p-1)
    # p: order of the basis
    # u: local coordinate of evaluation
    # U: knot vector
    # Return:
    # n or mid: The knot span index"""
    #
    cdef int low,mid,high

    if u==U[n+1]: return n;

    low=p
    high=n+1
    mid=(low+high)/2

    while u < U[mid] or u >= U[mid+1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid=(low+high)/2
    return mid

##############################################################################
#####                                                                    #####
#####                          multiFindSpan                             #####
#####                                                                    #####
##############################################################################

def multiFindSpan(int n, int p, np.ndarray[np.float_t] us, np.ndarray[np.float_t] U):
    """Determine the knot span index
    # Input:
    # n: number of basis functions - 1 (n=m-p-1)
    # p: order of the basis
    # us: local coordinates of evaluation
    # U: knot vector
    # Return:
    # n or mid: The knot span index"""
    #
    cdef np.ndarray[np.int_t] spans = np.zeros((us.shape[0],),dtype=int)

    cdef int low,mid,high,num
    cdef double u

    for num in xrange(us.shape[0]):

        u = us[num]

        if u==U[n+1]:
            spans[num] = n
        else:
            low=p
            high=n+1
            mid=(low+high)/2

            while u < U[mid] or u >= U[mid+1]:
                if u < U[mid]:
                    high = mid
                else:
                    low = mid
                mid=(low+high)/2
                
            spans[num] = mid

    return spans

##############################################################################
#####                                                                    #####
#####                            BasisFuns                               #####
#####                                                                    #####
##############################################################################

def BasisFuns(int i, double u, int p, np.ndarray[np.float_t] U):
    """Compute the nonvanishing basis functions at u
    # Input:
    # i: span index
    # u: local coordinate of evaluation
    # p: order of the B-spline basis
    # U: knot vector
    # Return:
    # N: Non-zero basis functions N_i-p,..,N_i"""
    #

    cdef np.ndarray[np.float_t] N = np.zeros((p+1,))
    cdef np.ndarray[np.float_t] left = np.zeros((p+1,))
    cdef np.ndarray[np.float_t] right = np.zeros((p+1,))

    cdef int j,r
    cdef double saved,temp

    N[0] = 1
    for j in xrange(1,p+1):
        left[j] = u-U[i+1-j]
        right[j] = U[i+j]-u
        saved = 0

        for r in xrange(j):
            try:
                temp = N[r]/(right[r+1]+left[j-r])
            except ZeroDivisionError:
                temp = 0
            N[r] = saved+right[r+1]*temp
            saved = left[j-r]*temp

        N[j] = saved

    return N

##############################################################################
#####                                                                    #####
#####                           multiBasisFuns                           #####
#####                                                                    #####
##############################################################################

def multiBasisFuns(np.ndarray[np.int_t] spans, np.ndarray[np.float_t] us, int p, np.ndarray[np.float_t] U):
    """Compute the nonvanishing basis functions at u
    # Input:
    # spans: span indices
    # us: local coordinates of evaluation
    # p: order of the B-spline basis
    # U: knot vector
    # Return:
    # N: Non-zero basis functions N_i-p,..,N_i"""
    #

    cdef np.ndarray[np.float_t,ndim=2] N_num = np.zeros((us.shape[0],U.shape[0]-p-1))
    cdef np.ndarray[np.float_t] N = np.zeros((p+1,))
    cdef np.ndarray[np.float_t] left = np.zeros((p+1,))
    cdef np.ndarray[np.float_t] right = np.zeros((p+1,))

    cdef int j,r,i,num
    cdef double saved,temp,u

    for num in xrange(us.shape[0]):

        u=us[num]
        i=spans[num]

        N[0] = 1
        for j in xrange(1,p+1):
            left[j] = u-U[i+1-j]
            right[j] = U[i+j]-u
            saved = 0

            for r in xrange(j):
                try:
                    temp = N[r]/(right[r+1]+left[j-r])
                except ZeroDivisionError:
                    temp = 0
                N[r] = saved+right[r+1]*temp
                saved = left[j-r]*temp

            N[j] = saved

        N_num[num,i-p:i+1] = N

    return N_num

##############################################################################
#####                                                                    #####
#####                           DersBasisFuns                            #####
#####                                                                    #####
##############################################################################

def DersBasisFuns(int i, double u, int p, int n, np.ndarray[np.float_t] U):
    """Compute nonzero basis functions and their derivatives
    # Input:
    # i: span index
    # u: local coordinate of evaluation
    # p: order of the B-spline basis
    # n: number of derivatives
    # U: knot vector
    # Return:
    # ders: two-dimensional array ders[k][j]
    #       kth derivative of N_i-p-j,p
    #       0<=k<=n and 0<=j<=p"""
    #
    cdef np.ndarray[np.float_t, ndim=2] ders = np.zeros((n+1,p+1))

    cdef np.ndarray[np.float_t, ndim=2] ndu = np.zeros((p+1,p+1))
    cdef np.ndarray[np.float_t] left = np.zeros((p+1,))
    cdef np.ndarray[np.float_t] right = np.zeros((p+1,))
    cdef np.ndarray[np.float_t, ndim=2] a = np.zeros((p,p+1))
    cdef int j,s1,s2,k,j1,j2
    cdef int r,rk,pk
    cdef double saved,temp,d

    ndu[0,0]=1.0
    for j in xrange(1,p+1):
        left[j] = u-U[i+1-j]
        right[j] = U[i+j]-u
        saved = 0.0
        for r in xrange(j):
            # lower triangle
            ndu[j,r] = right[r+1]+left[j-r]
            temp = ndu[r,j-1]/ndu[j,r]
            #upper triangle
            ndu[r,j] = saved+right[r+1]*temp
            saved = left[j-r]*temp
        ndu[j,j] = saved
    for j in xrange(p+1): # Load the basis functions
        ders[0,j] = ndu[j,p]
    # This section computes the derivatives
    for r in xrange(p+1): # Loop over function index
        s1=0
        s2=1
        a[0,0] = 1.0
        # Loop to compute kth derivative
        for k in xrange(1,n+1):
            d = 0.0
            rk = r-k
            pk = p-k
            if r >= k:
                a[s2,0] = a[s1,0]/ndu[pk+1,rk]
                d = a[s2,0]*ndu[rk,pk]
            if rk >= -1:
                j1 = 1
            else:
                j1= -rk
            if r-1 <= pk:
                j2 = k-1
            else:
                j2 = p-r
            for j in xrange(j1,j2+1):
                a[s2,j] = (a[s1,j]-a[s1,j-1])/ndu[pk+1,rk+j]
                d += a[s2,j]*ndu[rk+j,pk]
            if r <= pk:
                a[s2,k] = -a[s1,k-1]/ndu[pk+1,r]
                d += a[s2,k]*ndu[r,pk]
            ders[k,r] = d
            j=s1
            s1=s2
            s2=j # switch rows
    # Multiply through by the correct factors
    r = p
    for k in xrange(1,n+1):
        for j in xrange(p+1):
            ders[k,j] *= r
        r *= (p-k)

    return ders

##############################################################################
#####                                                                    #####
#####                        multiDersBasisFuns                          #####
#####                                                                    #####
##############################################################################

def multiDersBasisFuns(np.ndarray[np.int_t] spans, np.ndarray[np.float_t] us, int p, int n, np.ndarray[np.float_t] U):
    """Compute nonzero basis functions and their derivatives
    # Input:
    # spans: span indices
    # us: local coordinates of evaluation
    # p: order of the B-spline basis
    # n: number of derivatives
    # U: knot vector
    # Return:
    # ders: two-dimensional array N_num[num][k][j]
    #       kth derivative of N_i-p-j,p
    #       0<=k<=n and 0<=j<=p"""
    #
        
    cdef np.ndarray[np.float_t,ndim=3] N_num = np.zeros((us.shape[0],n+1,U.shape[0]-p-1))
        
    cdef np.ndarray[np.float_t, ndim=2] ders = np.zeros((n+1,p+1))
    cdef np.ndarray[np.float_t, ndim=2] ndu = np.zeros((p+1,p+1))
    cdef np.ndarray[np.float_t] left = np.zeros((p+1,))
    cdef np.ndarray[np.float_t] right = np.zeros((p+1,))
    cdef np.ndarray[np.float_t, ndim=2] a = np.zeros((p,p+1))
    cdef int j,s1,s2,k,j1,j2
    cdef int r,rk,pk,i,num
    cdef double saved,temp,d,u

    for num in xrange(us.shape[0]):

        u=us[num]
        i=spans[num]

        ndu[0,0]=1.0
        for j in xrange(1,p+1):
            left[j] = u-U[i+1-j]
            right[j] = U[i+j]-u
            saved = 0.0
            for r in xrange(j):
                # lower triangle
                ndu[j,r] = right[r+1]+left[j-r]
                temp = ndu[r,j-1]/ndu[j,r]
                #upper triangle
                ndu[r,j] = saved+right[r+1]*temp
                saved = left[j-r]*temp
            ndu[j,j] = saved
        for j in xrange(p+1): # Load the basis functions
            ders[0,j] = ndu[j,p]
        # This section computes the derivatives
        for r in xrange(p+1): # Loop over function index
            s1=0
            s2=1
            a[0,0] = 1.0
            # Loop to compute kth derivative
            for k in xrange(1,n+1):
                d = 0.0
                rk = r-k
                pk = p-k
                if r >= k:
                    a[s2,0] = a[s1,0]/ndu[pk+1,rk]
                    d = a[s2,0]*ndu[rk,pk]
                if rk >= -1:
                    j1 = 1
                else:
                    j1= -rk
                if r-1 <= pk:
                    j2 = k-1
                else:
                    j2 = p-r
                for j in xrange(j1,j2+1):
                    a[s2,j] = (a[s1,j]-a[s1,j-1])/ndu[pk+1,rk+j]
                    d += a[s2,j]*ndu[rk+j,pk]
                if r <= pk:
                    a[s2,k] = -a[s1,k-1]/ndu[pk+1,r]
                    d += a[s2,k]*ndu[r,pk]
                ders[k,r] = d
                j=s1
                s1=s2
                s2=j # switch rows
        # Multiply through by the correct factors
        r = p
        for k in xrange(1,n+1):
            for j in xrange(p+1):
                ders[k,j] *= r
            r *= (p-k)

        N_num[num,:,i-p:i+1] =  ders

    return N_num

##############################################################################
#####                                                                    #####
#####                            NURBSbasis                              #####
#####                                                                    #####
##############################################################################

def NURBSbasis(int i, double u, int p, np.ndarray[np.float_t] U,np.ndarray[np.float_t] w):
    """Compute nonzero basis functions
    # Input:
    # i: span index
    # u: local coordinate of evaluation
    # p: order of the B-spline basis
    # U: knot vector
    # w: weights associated with control points
    # Return:
    # Non-zero basis functions R_i-p,..,R_i"""
    #
    cdef np.ndarray[np.float_t] R = np.zeros((U.shape[0]-p-1))
    cdef np.ndarray[np.float_t] N = BasisFuns(i,u,p,U)

    cdef int j
    cdef double W = 0.0

    for j in xrange(p+1):
        W = W + N[j] * w[i-p+j]

    for j in xrange(p+1):
        R[i-p+j] = N[j] * w[i-p+j] / W

    return R
    
    
##############################################################################
#####                                                                    #####
#####                          multiNURBSbasis                           #####
#####                                                                    #####
##############################################################################

def multiNURBSbasis(np.ndarray[np.int_t] spans, np.ndarray[np.float_t] us, int p, np.ndarray[np.float_t] U,np.ndarray[np.float_t] w):
    """Compute nonzero basis functions
    # Input:
    # spans: span indices
    # us: local coordinates of evaluation
    # p: order of the B-spline basis
    # U: knot vector
    # w: weights associated with control points
    # Return:
    # Non-zero basis functions R_i-p,..,R_i"""
    #
    cdef np.ndarray[np.float_t,ndim=2] R = np.zeros((us.shape[0],U.shape[0]-p-1))
    cdef np.ndarray[np.float_t,ndim=2] N = multiBasisFuns(spans,us,p,U)

    cdef int j,i,num
    cdef double u,W

    for num in xrange(us.shape[0]):
        u=us[num]
        i=spans[num]
        
        W = 0.0

        for j in xrange(p+1):
            W = W + N[num,i-p+j] * w[i-p+j]

        for j in xrange(p+1):
            R[num,i-p+j] = N[num,i-p+j] * w[i-p+j] / W

    return R

##############################################################################
#####                                                                    #####
#####                                Bin                                 #####
#####                                                                    #####
##############################################################################

def Bin(int k, int j):

    cdef int a
    cdef double res
    
    if j < 0 or j > k:
        return 0
    if j > k - j:
        j = k - j
    res = 1.0
    for a in range(j):
        res = res * (k - (j - (a+1.0)))
        res = res // (a+1)
    return res

##############################################################################
#####                                                                    #####
#####                          dersNURBSbasis                            #####
#####                                                                    #####
##############################################################################

def dersNURBSbasis(int i, double u, int p, int n, np.ndarray[np.float_t] U,np.ndarray[np.float_t] w):
    """Compute nonzero basis functions
    # Input:
    # i: span index
    # u: local coordinate of evaluation
    # p: order of the B-spline basis
    # n: number of derivatives
    # U: knot vector
    # w: weights associated with control points
    # Return:
    # Non-zero basis functions R_i-p,..,R_i"""
    #
    cdef np.ndarray[np.float_t, ndim=2] R = np.zeros((n+1,U.shape[0]-p-1))
    R[0,:] = NURBSbasis(i,u,p,U)
    cdef np.ndarray[np.float_t, ndim=2] N = DersBasisFuns(i,u,p,n,U)

    cdef np.ndarray[np.float_t] W = np.zeros((n+1,))
    cdef int d,j,k
    cdef double numer

    for d in xrange(n+1):
        for j in xrange(i-p,i+1):
            W[d] = W[d] + N[d,j]*w[j]

    for d in xrange(1,n+1):
        for j in xrange(i-p,i+1):
            numer = w[j] * N[d,j]
            for k in xrange(1,d+1):
                numer = numer - Bin(d,k)*W[k]*R[d-k,j]
            R[d,j] = numer / W[0]

    return R

##############################################################################
#####                                                                    #####
#####                        multidersNURBSbasis                         #####
#####                                                                    #####
##############################################################################

def multidersNURBSbasis(np.ndarray[np.int_t] spans, np.ndarray[np.float_t] us, int p, int n, np.ndarray[np.float_t] U,np.ndarray[np.float_t] w):
    """Compute nonzero basis functions
    # Input:
    # spans: span indices
    # us: local coordinates of evaluation
    # p: order of the B-spline basis
    # n: number of derivatives
    # U: knot vector
    # w: weights associated with control points
    # Return:
    # Non-zero basis functions R_i-p,..,R_i"""
    #
    cdef np.ndarray[np.float_t, ndim=3] R = np.zeros((us.shape[0],n+1,U.shape[0]-p-1))
    R[:,0,:] = multiNURBSbasis(spans,us,p,U,w)
    cdef np.ndarray[np.float_t, ndim=3] N = multiDersBasisFuns(spans,us,p,n,U)
    cdef np.ndarray[np.float_t] W = np.zeros((n+1,))
    cdef int d,j,k,num,i
    cdef double numer

    for num in xrange(us.shape[0]):

        u=us[num]
        i=spans[num]
        W = np.zeros((n+1,))

        for d in xrange(n+1):
            for j in xrange(i-p,i+1):
                W[d] = W[d] + N[num,d,j]*w[j]

        for d in xrange(1,n+1):
            for j in xrange(i-p,i+1):
                numer = w[j] * N[num,d,j]
                for k in xrange(1,d+1):
                    numer = numer - Bin(d,k)*W[k]*R[num,d-k,j]
                R[num,d,j] = numer / W[0]

    return R

##############################################################################
#####                                                                    #####
#####                        RefineKnotVectCurve                         #####
#####                                                                    #####
##############################################################################

def RefineKnotVectCurve(int n, int p, np.ndarray[np.float_t] UP, np.ndarray[np.float_t,ndim=2] Pw, np.ndarray[np.float_t] X):
    """Insert a set of knots X into a NURBS curve
    # Input:
    # n: number of old basis - 1
    # p: order of the B-spline basis
    # UP: old knot vector
    # Pw: old homogenous control points
    # X: knots to be inserted
    # Return:
    # nq: new number of basis
    # UQ: new knot vector
    # Q: new control points
    # wq: new weights"""
    #   
    cdef int r = X.size-1
    cdef int nq = n+r+1
    
    cdef int m = n+p+1
    cdef int a = FindSpan(n,p,X[0],UP)
    cdef int b = FindSpan(n,p,X[r],UP) + 1

    cdef np.ndarray[np.float_t] UQ = np.zeros((UP.size+X.size,))
    cdef np.ndarray[np.float_t, ndim=2] Qw = np.zeros((Pw.shape[0]+r+1,3))
    
    cdef int i,k,l,ind
    cdef int j
    cdef double alfa

    Qw[:a-p+1,:] = Pw[:a-p+1,:]
    Qw[b+r:,:] = Pw[b-1:,:]
    UQ[:a+1] = UP[:a+1]    
    UQ[b+p+r+1:] = UP[b+p:]

    i = b+p-1
    k = b+p+r

    for j in xrange(r,-1,-1):
        while X[j]<=UP[i] and i>a:
            Qw[k-p-1] = Pw[i-p-1]
            UQ[k] = UP[i]
            k -= 1
            i -= 1
        Qw[k-p-1] = Qw[k-p]
        for l in xrange(1,p+1):
            ind = k-p+l
            alfa = UQ[k+l]-X[j]
            if abs(alfa) == 0.0:
                Qw[ind-1] = Qw[ind]
            else:
                alfa /= UQ[k+l]-UP[i-p+l]
                Qw[ind-1] = alfa*Qw[ind-1] + (1.0-alfa)*Qw[ind]
        UQ[k] = X[j]
        k -= 1

    return nq,UQ,Qw


##############################################################################
#####                                                                    #####
#####                            AllBernstein                            #####
#####                                                                    #####
##############################################################################

def AllBernstein(int n, double u):
    """Compute all nth-degree Bernstein polynomials
    # Input (n,u)
    # n: degree
    # u: coordinate of evaluation
    # Output:
    # B: Berstein polynomials B[0],...,B[n]"""

    cdef np.ndarray[np.float_t] B = np.zeros((n+1,))
    cdef int j,k
    cdef double u1,saved,temp

    B[0] = 1.0
    u1 = 1.0 - u
    for j in xrange(1,n+1):
        saved = 0.0
        for k in xrange(j):
            temp = B[k]
            B[k] = saved+u1*temp
            saved = u*temp
        B[j] = saved

    return B

##############################################################################
#####                                                                    #####
#####                         multiAllBernstein                          #####
#####                                                                    #####
##############################################################################

def multiAllBernstein(int n, np.ndarray[np.float_t] u):
    """Compute all nth-degree Bernstein polynomials
    # Input (n,u)
    # n: degree
    # u: coordinates of evaluation
    # Output:
    # B: Berstein polynomials B[u,0],...,B[u,n]"""

    cdef np.ndarray[np.float_t,ndim=2] B = np.zeros((u.shape[0],n+1))
    cdef int j,k,num
    cdef double u1,saved,temp

    for num in xrange(u.shape[0]):

        B[num,0] = 1.0
        u1 = 1.0 - u[num]
        for j in xrange(1,n+1):
            saved = 0.0
            for k in xrange(j):
                temp = B[num,k]
                B[num,k] = saved+u1*temp
                saved = u[num]*temp
            B[num,j] = saved

    return B

##############################################################################
#####                                                                    #####
#####                          AllBernsteinDers                          #####
#####                                                                    #####
##############################################################################

def AllBernsteinDers(int n, double u, int numd):
    """Compute nth-degree Bernstein polynomials and their derivatives
    # Input (n,u,numd)
    # n: degree
    # u: coordinate of evaluation
    # numd: number of derivatives
    # Output:
    # B: Berstein polynomials B[0,0],...,B[numd,n]"""

    cdef np.ndarray[np.float_t,ndim=2] B = np.zeros((numd+1,n+1))
    cdef np.ndarray[np.float_t] Bnminus
    cdef int d,j
    B[0,:] = AllBernstein(n,u)

    for d in xrange(1,numd+1):
        Bnminus = np.zeros((n+1-d+2,))
        Bnminus[1:-1] = AllBernstein(n-d,u)
        for j in xrange(n+1):
            B[d,j] = n * (Bnminus[j]-Bnminus[j+1])

    return B

##############################################################################
#####                                                                    #####
#####                        multiAllBernsteinDers                       #####
#####                                                                    #####
##############################################################################

def multiAllBernsteinDers(int n, np.ndarray[np.float_t] u, int numd):
    """Compute nth-degree Bernstein polynomials and their derivatives
    # Input (n,u,numd)
    # n: degree
    # u: coordinates of evaluation
    # numd: number of derivatives
    # Output:
    # B: Berstein polynomials B[u,0,0],...,B[u,numd,n]"""

    cdef np.ndarray[np.float_t,ndim=3] B = np.zeros((u.shape[0],numd+1,n+1))
    cdef np.ndarray[np.float_t,ndim=2] Bnminus
    cdef int d,j,num
    B[:,0,:] = multiAllBernstein(n,u)

    for d in xrange(1,numd+1):
        Bnminus = np.zeros((u.shape[0],n+3))
        Bnminus[:,1:2+n-d] = multiAllBernstein(n-d,u)
        for j in xrange(n+1):
            B[:,d,j] = n * (Bnminus[:,j]-Bnminus[:,j+1])

    return B

##############################################################################
#####                                                                    #####
#####                         RationalBernstein                          #####
#####                                                                    #####
##############################################################################
                
def RationalBernstein(int n, np.ndarray[np.float_t,ndim=2] Pw, double u):
    """Compute all nth-degree rational Bernstein polynomials
    # Input: (n,Pw,u)
    # n: degree
    # Pw: homogenous control points
    # u: coordinate of evaluation
    # Return:
    # B: Bernstein polynomials B[0],...,B[n]"""

    cdef np.ndarray[np.float_t] B = AllBernstein(n,u)
    cdef double W = 0.0
    cdef int j

    for j in xrange(n+1):
        W = W + B[j]*Pw[j,-1]
    for j in xrange(n+1):
        B[j] = B[j]/W*Pw[j,-1]

    return B


##############################################################################
#####                                                                    #####
#####                      multiRationalBernstein                        #####
#####                                                                    #####
##############################################################################
                
def multiRationalBernstein(int n, np.ndarray[np.float_t,ndim=2] Pw, np.ndarray[np.float_t] u):
    """Compute all nth-degree rational Bernstein polynomials
    # Input: (n,Pw,u)
    # n: degree
    # Pw: homogenous control points
    # u: coordinates of evaluation
    # Return:
    # B: Bernstein polynomials B[u,0],...,B[u,n]"""

    cdef np.ndarray[np.float_t,ndim=2] B = multiAllBernstein(n,u)
    cdef np.ndarray[np.float_t] W = np.zeros((u.shape[0],))
    cdef int j

    for j in xrange(n+1):
        W = W + B[:,j]*Pw[j,-1]
    for j in xrange(n+1):
        B[:,j] = B[:,j]/W*Pw[j,-1]

    return B

##############################################################################
#####                                                                    #####
#####                     multiRationalBernsteinDers                     #####
#####                                                                    #####
##############################################################################

def multiRationalBernsteinDers(int n, np.ndarray[np.float_t,ndim=2] Pw, np.ndarray[np.float_t] u, int numd):
    """Compute all nth-degree rational Bernstein polynomials
    # Input: (n,Pw,u,numd)
    # n: degree
    # Pw: homogenous control points
    # u: coordinates of evaluation
    # numd: number of derivatives to evaluate
    # Return:
    # B: Bernstein polynomials B[u,0,0],...,B[u,n,numd]"""

    cdef np.ndarray[np.float_t,ndim=3] B = multiAllBernsteinDers(n,u,numd)
    cdef np.ndarray[np.float_t,ndim=3] RatB = np.zeros((u.shape[0],n+1,numd+1))
    cdef int i,j,k
    cdef np.ndarray[np.float_t] saved
    cdef np.ndarray[np.float_t,ndim=2] Wj = np.zeros((u.shape[0],numd+1))
    RatB[:,:,0] = multiRationalBernstein(n,Pw,u)
    
    for j in xrange(numd+1):
        for i in xrange(n+1):
            Wj[:,j] = Wj[:,j] + B[:,j,i] * Pw[i,-1]

    for k in xrange(1,numd+1):
        for i in xrange(n+1):
            saved = Pw[i,-1] * B[:,k,i]
            for j in xrange(1,k+1):
                saved = saved - Bin(k,j) * Wj[:,j] * RatB[:,i,k-j]
            RatB[:,i,k] = saved / Wj[:,0]

    return RatB


##############################################################################
#####                                                                    #####
#####                           DecomposeCurve                           #####
#####                                                                    #####
##############################################################################

def DecomposeCurve(int n, int p, np.ndarray[np.float_t] U, np.ndarray[np.float_t, ndim=2] Pw):
    """Decompose curve into Bezier segments
    # Input:
    # n: number of old basis - 1
    # p: order of the B-spline basis
    # U: knot vector
    # Pw: homogenous control points
    # Return:
    # nb: number of Bezier segments
    # Qw: Qw[j][k] is the kth control point of the jth segment"""
    #
    cdef  int m = n+p+1
    cdef  int a = p
    cdef  int b = p+1
    cdef  int nb = 0

    cdef np.ndarray[np.float_t, ndim=3] Qw = np.zeros((np.size(np.unique(U))-1,p+1,3))
    cdef np.ndarray[np.float_t] alphas = np.zeros((p,))

    cdef  int i,mult,r,save,s
    cdef int j,k
    cdef double numer=0,alpha

    for i in xrange(p+1):
        Qw[nb][i] = Pw[i]
    while b < m:
        i = b
        while b < m and U[b+1] == U[b]:
            b+=1
        mult = b-i+1
        if mult < p:
            numer = U[b]-U[a] # numerator of alpha
        # compute and store alphas
        for j in xrange(p,mult,-1):
            alphas[j-mult-1] = numer/(U[a+j]-U[a])
        r = p-mult
        for j in xrange(1,r+1):
            save = r-j
            s = mult+j # this many new points
            for k in xrange(p,s-1,-1):
                alpha = alphas[k-s]
                Qw[nb][k] = alpha*Qw[nb][k] + (1.-alpha)*Qw[nb][k-1]
            if b < m:
                Qw[nb+1][save] = Qw[nb][p] # next segment
        nb = nb+1 # Bexier segment completed
        if b < m:
            # initialize for next segment
            for i in xrange(p-mult,p+1):
                Qw[nb][i] = Pw[b-p+i]
            a=b
            b+=1

    return nb,Qw







##############################################################################
###                                                                        ###
###                        NURBS SURFACE CODE                              ###
###                                                                        ###
##############################################################################


##############################################################################
#####                                                                    #####
#####                           SurfacePoint                             #####
#####                                                                    #####
##############################################################################

def SurfacePoint( int n, int p, np.ndarray[np.float_t] U,  int m,  int q, np.ndarray[np.float_t] V, np.ndarray[np.float_t, ndim=3] Pw, double u, double v):
    """Compute point on rational B-spline surface
    # Input:
    # u direction
    # n: number of basis - 1
    # p: order of basis
    # U: knot vector
    # v direction
    # m: number of basis - 1
    # q: order of basis
    # V: knot vector
    # Pw: homogenous control points
    # u: local coordinate of evaluation
    # v: local coordinate of evaluation
    # Return:
    # Sw: x,y,z coordinates of point"""
    #
    cdef  int uspan = FindSpan(n,p,u,U)
    cdef np.ndarray[np.float_t] Nu = BasisFuns(uspan,u,p,U)
    cdef  int vspan = FindSpan(m,q,v,V)
    cdef np.ndarray[np.float_t] Nv = BasisFuns(vspan,v,q,V)

    cdef  int l,k
    cdef np.ndarray[np.float_t] temp = np.zeros((q+1,))

    for l in xrange(q+1):
        temp[l] = 0.0
        for k in xrange(p+1):
            temp[l] = temp[l] + Nu[k]*Pw[uspan-p+k][vspan-q+l]

    cdef np.ndarray[np.float_t,ndim=2] Sw = np.zeros((1,4))
    for l in xrange(q+1):
        Sw = Sw + Nv[l]*temp[l]

    return (Sw[:,:3].T/Sw[:,3].T).T

##############################################################################
#####                                                                    #####
#####                        multiSurfacePoint                           #####
#####                                                                    #####
##############################################################################

def multiSurfacePoint( int n,  int p, np.ndarray[np.float_t] U,  int m,  int q, np.ndarray[np.float_t] V, np.ndarray[np.float_t, ndim=3] Pw, np.ndarray[np.float_t] us, np.ndarray[np.float_t] vs):
    """Compute point on rational B-spline surface
    # Input:
    # u direction
    # n: number of basis - 1
    # p: order of basis
    # U: knot vector
    # v direction
    # m: number of basis - 1
    # q: order of basis
    # V: knot vector
    # uv surface
    # Pw: homogenous control points
    # us: local coordinates of evaluation
    # vs: local coordinates of evaluation
    # Return:
    # Sw: x,y,z coordinates of point"""
    #
    cdef np.ndarray[np.int_t] uspans = multiFindSpan(n,p,us,U)
    cdef np.ndarray[np.float_t] Nu
    cdef np.ndarray[np.int_t] vspans = multiFindSpan(m,q,vs,V)
    cdef np.ndarray[np.float_t] Nv
    cdef np.ndarray[np.float_t,ndim=2] S  = np.zeros((us.shape[0],4))
    cdef np.ndarray[np.float_t,ndim=2] Sw = np.zeros((1,4))

    cdef  int l,k,num,uspan,vspan
    cdef np.ndarray[np.float_t,ndim=2] temp = np.zeros((q+1,4))

    for num in xrange(us.shape[0]):

        uspan = uspans[num]
        Nu = BasisFuns(uspan,us[num],p,U)
        vspan = vspans[num]
        Nv = BasisFuns(vspan,vs[num],q,V)

        for l in xrange(q+1):
            for k in xrange(p+1):
                temp[l] = temp[l] + Nu[k]*Pw[uspan-p+k][vspan-q+l]

        Sw = np.zeros((1,4))
        for l in xrange(q+1):
            Sw = Sw + Nv[l]*temp[l]

        S[num]=Sw        

    return (S[:,:3].T/S[:,3].T).T

##############################################################################
#####                                                                    #####
#####                         multiSurfaceBasis                          #####
#####                                                                    #####
##############################################################################

def multiSurfaceBasis(np.ndarray[np.float_t,ndim=2] uv,  int p,  int n, np.ndarray[np.float_t] U,  int q,  int m, np.ndarray[np.float_t] V, np.ndarray[np.float_t,ndim=2] w):

    cdef np.ndarray[np.int_t] uspan = multiFindSpan(n,p,uv[0],U)
    cdef np.ndarray[np.float_t,ndim=2] Nu = multiBasisFuns(uspan,uv[0],p,U)
    cdef np.ndarray[np.int_t] vspan = multiFindSpan(m,q,uv[1],V)
    cdef np.ndarray[np.float_t,ndim=2] Nv = multiBasisFuns(vspan,uv[1],q,V)
    
    cdef np.ndarray[np.float_t,ndim=3] R = np.zeros((uv.shape[1],n+1,m+1))

    cdef  int num,i,j,k,l

    cdef double denom

    for num in xrange(uv.shape[1]):
        denom=0.0

        for k in xrange(n+1):
            for l in xrange(m+1):
                denom = denom + Nu[num,k]*Nv[num,l]*w[k,l]

        i=uspan[num]
        j=vspan[num]
        for k in xrange(p+1):
            for l in xrange(q+1):
                R[num,i-p+k,j-q+l] = Nu[num,i-p+k] * Nv[num,j-q+l] * w[i-p+k,j-q+l] / denom

    return R



##############################################################################
#####                                                                    #####
#####                          DecomposeSurface                          #####
#####                                                                    #####
##############################################################################

def DecomposeSurface( int n,  int p, np.ndarray[np.float_t] U,  int m,  int q, np.ndarray[np.float_t] V, np.ndarray[np.float_t,ndim=3] Pw):
    """Decompose surface into Bezier patches
    # Input:
    # n: number of basis - 1
    # p: order of basis
    # U: knot vector
    # m: number of basis - 1
    # q: order of basis
    # V: knot vector
    # Pw: homogeneous control points
    # Output:
    # nb: number of Bezier patches
    # Qw: control points of Bezier patches"""

    cdef  int a,b,i,nb,row,mult,r,save,s,num
    cdef int j,k
    cdef double numer,alfa
    cdef np.ndarray[np.float_t] alphas
    cdef  int nu = np.size(np.unique(U))-1
    cdef  int nv = np.size(np.unique(V))-1
    
    cdef np.ndarray[np.float_t,ndim=4] Qw = np.zeros((nu,p+1,m+1,4))
    cdef np.ndarray[np.float_t,ndim=4] Qw2 = np.zeros((nu*nv,p+1,q+1,4))

    
    m2=n+p+1
    a=p
    b=p+1
    nb=0
    for i in xrange(p+1):
        for row in xrange(m+1):
            Qw[nb][i][row] = Pw[i][row]
    while b<m2:
        i=b
        while b < m2 and U[b+1] == U[b]:
            b+=1
        mult = b-i+1
        if mult<p:
            numer = U[b]-U[a]
            alphas=np.zeros((p-mult,))
            for j in xrange(p,mult,-1):
                alphas[j-mult-1] = numer/(U[a+j]-U[a])
            r = p-mult
            for j in xrange(1,r+1):
                save=r-j
                s=mult+j
                for k in xrange(p,s-1,-1):
                    alfa = alphas[k-s]
                    for row in xrange(m+1):
                        Qw[nb][k][row] = alfa*Qw[nb][k][row] + (1.0-alfa)*Qw[nb][k-1][row]
                if b<m2:
                    for row in xrange(m+1):
                        Qw[nb+1][save][row] = Qw[nb][p][row]
            nb = nb+1
            if b<m2:
                for i in xrange(p-mult,p+1):
                    for row in xrange(m+1):
                        Qw[nb][i][row] = Pw[b-p+i][row]
                a=b
                b=b+1

    
    n=p # As now decomposed in u direction
    
    for num in xrange(nu):

        Pw = Qw[num]
        
        m2=m+q+1
        a=q
        b=q+1
        nb=0
        for i in xrange(q+1):
            for row in xrange(n+1):
                Qw2[num*nv+nb][row][i] = Pw[row][i]
        while b<m2:
            i=b
            while b < m2 and V[b+1] == V[b]:
                b+=1
            mult = b-i+1
            if mult<q:
                numer = V[b]-V[a]
                alphas=np.zeros((q-mult,))
                for j in xrange(q,mult,-1):
                    alphas[j-mult-1] = numer/(V[a+j]-V[a])
                r = q-mult
                for j in xrange(1,r+1):
                    save=r-j
                    s=mult+j
                    for k in xrange(q,s-1,-1):
                        alfa = alphas[k-s]
                        for row in xrange(n+1):
                            Qw2[num*nv+nb][row][k] = alfa*Qw2[num*nv+nb][row][k] + (1.0-alfa)*Qw2[num*nv+nb][row][k-1]
                    if b<m2: 
                        for row in xrange(n+1):
                            Qw2[num*nv+nb+1][row][save] = Qw2[num*nv+nb][row][q]
                nb = nb+1
                if b<m2:
                    for i in xrange(q-mult,q+1):
                        for row in xrange(n+1):
                            Qw2[num*nv+nb][row][i] = Pw[row][b-q+i]
                    a=b
                    b=b+1

    return Qw2


    
##############################################################################
#####                                                                    #####
#####                        RefineKnotVectSurface                       #####
#####                                                                    #####
##############################################################################

def RefineKnotVectSurface( int n,  int p, np.ndarray[np.float_t] U,  int m,  int q, np.ndarray[np.float_t] V, np.ndarray[np.float_t,ndim=3] Pw, np.ndarray[np.float_t] X,  int d):
    """Insert a set of knots X into a knot vector U or V

    Input:
        n  : number of basis function in U
        p  : order of U
        U  : knot vector
        m  : number of basis function in V
        q  : order of V
        V  : knot vector
        Pw : homogenous control points
        X  : knots to be inserted
        d  : choose which knot vector to insert into
                d=0 --> U    ;    d=1 --> V
    
    Return:
        Ubar : new U knot vector
        Vbar : new V knot vector
        Qw   : new homogeonous control points"""
    
    cdef  int r = X.size-1
    cdef  int a,b,row,i,k,l
    cdef np.ndarray[np.float_t] Ubar
    cdef np.ndarray[np.float_t] Vbar
    cdef int j
    cdef np.ndarray[np.float_t,ndim=3] Qw
    
    if d == 0:
        # Initialise Qw
        Qw = np.zeros((Pw.shape[0]+r+1,Pw.shape[1],4))
        a = int(FindSpan(n,p,X[0],U))
        b = int(FindSpan(n,p,X[r],U) + 1)
        # Intialise Ubar
        Ubar = np.zeros((U.size+r+1,))
        Ubar[:a+1] = U[:a+1]
        Ubar[b+p+r+1:] = U[b+p:]
        # Copy V in Vbar
        Vbar = V
        
        # Save unaltered control points
        for row in xrange(m+1):
            for k in xrange(a-p+1): Qw[k][row] = Pw[k][row]
            for k in xrange(b-1,n+1): Qw[k+r+1][row] = Pw[k][row]
            
        i = int(b+p-1)
        k = int(b+p+r)
        
        for j in xrange(r,-1,-1):
            while X[j]<=U[i] and i>a:
                Ubar[k] = U[i]
                for row in xrange(m+1): Qw[k-p-1][row] = Pw[i-p-1][row]
                k -= 1
                i -= 1
            for row in xrange(m+1): Qw[k-p-1][row] = Qw[k-p][row]
            
            for l in xrange(1,p+1):
                ind = k-p+l
                alfa = Ubar[k+l]-X[j]
                if abs(alfa) == 0.0:
                    for row in xrange(m+1): Qw[ind-1][row] = Qw[ind][row]
                else:
                    alfa /= Ubar[k+l]-U[i-p+l]
                    for row in xrange(m+1): Qw[ind-1][row] = alfa*Qw[ind-1][row] + (1.0-alfa)*Qw[ind][row]
            Ubar[k] = X[j]
            k -= 1
            
    if d == 1:
        j=0
        while j < X.size:
            x = X[j]
            r = np.size(np.where(X==x)[0])
            j += r
            U,V,Pw = SurfaceKnotIns(n,p,U,m,q,V,Pw,1,x,r,np.size(np.where(V==x)[0]))
            m += r
        ## Initialise Qw
        #Qw = np.zeros((Pw.shape[0],Pw.shape[1]+r+1,4))
        #a = int(FindSpan(m,q,X[0],V))
        #b = int(FindSpan(m,q,X[r],V) + 1)
        ## Intialise Vbar
        #Vbar = np.zeros((V.size+r+1,))
        #Vbar[:a+1] = V[:a+1]
        #Vbar[b+q+r+1:] = V[b+q:]
        ## Copy U in Ubar
        #Ubar = U
        #
        ## Save unaltered control points
        #for row in xrange(n+1):
        #    for k in xrange(a-q+1): Qw[row][k] = Pw[row][k]
        #    for k in xrange(b-1,m+1): Qw[row][k+r+1] = Pw[row][k]
        #    
        #i = int(b+q-1)
        #k = int(b+q+r)
        #
        #for j in xrange(r,-1,-1):
        #    while X[j]<=V[i] and i>a:
        #        Vbar[k] = V[i]
        #        for row in xrange(n+1): Qw[row][k-q-1] = Pw[row][i-q-1]
        #        k -= 1
        #        i -= 1
        #    for row in xrange(n+1): Qw[row][k-q-1] = Qw[row][k-q]
        #    
        #    for l in xrange(1,q+1):
        #        ind = k-q+l
        #        alfa = Vbar[k+l]-X[j]
        #        if abs(alfa) == 0.0:
        #            for row in xrange(n+1): Qw[row][ind-1] = Qw[row][ind]
        #        else:
        #            alfa /= Vbar[k+l]-V[i-q+l]
        #            for row in xrange(n+1): Qw[row][ind-1] = alfa*Qw[row][ind-1] + (1.0-alfa)*Qw[row][ind]
        #    Vbar[k] = X[j]
        #    k -= 1
            
            
    return Ubar,Vbar,Qw


##############################################################################
#####                                                                    #####
#####                         SurfaceKnotIns                             #####
#####                                                                    #####
##############################################################################

def SurfaceKnotIns( int n,  int p, np.ndarray[np.float_t] UP,  int m,  int q, np.ndarray[np.float_t] VP, np.ndarray[np.float_t,ndim=3] Pw,  int d, double uv,  int r,  int s):
    """Insert, r times, the knot uv into U or V

    Input:
        n  : number of basis function in U
        p  : order of U
        U  : knot vector
        m  : number of basis function in V
        q  : order of V
        V  : knot vector
        Pw : homogenous control points
        d  : choose which knot vector to insert into
                d=0 --> U    ;    d=1 --> V
        uv : knot to be inserted
        r  : multiplicity
        s  : current multiplicity
    
    Return:
        UQ   : new U knot vector
        VQ   : new U knot vector
        Qw   : new homogeonous control points"""
        
    cdef  int i,j,k,L,row
    cdef np.ndarray[np.float_t] VQ
    cdef np.ndarray[np.float_t] UQ
    cdef np.ndarray[np.float_t,ndim=2] alpha
    cdef np.ndarray[np.float_t,ndim=2] Rw
    cdef np.ndarray[np.float_t,ndim=3] Qw
    
    if d == 0:
        pass
        
    if d == 1:
        alpha = np.zeros((q-s,r+1))
        Rw = np.zeros((q-s+1,4))
        Qw = np.zeros((Pw.shape[0],Pw.shape[1]+r,4))
        # Load new V vector
        k = int(FindSpan(m,q,uv,VP))
        VQ = np.zeros((VP.size+r,))
        for i in xrange(k+1): VQ[i] = VP[i]
        for i in xrange(1,r+1): VQ[k+i] = uv
        for i in xrange(k+1,m+q+2): VQ[r+i] = VP[i]
        # Copy UP into UQ
        UQ = UP
        # Save the alphas
        for j in xrange(1,r+1):
            L = k-q+j
            for i in xrange(q-j-s+1):
                alpha[i][j] = (uv-VP[L+i])/(VP[i+k+1]-VP[L+i])
        for row in xrange(n+1):
            # Save unaltered control points
            for i in xrange(k-q+1): Qw[row][i] = Pw[row][i]
            for i in xrange(k-s,m+1): Qw[row][i+r] = Pw[row][i]
            # Save auxiliary control points
            for i in xrange(q-s+1): Rw[i] = Pw[row][k-q+i]
            for j in xrange(1,r+1):
                # Insert knot r times
                L = k-q+j
                for i in xrange(q-j-s+1): Rw[i] = alpha[i][j]*Rw[i+1] + (1.0-alpha[i][j])*Rw[i]
                Qw[row][L] = Rw[0]
                Qw[row][k+r-j-s] = Rw[q-j-s]
            # Load the remaining control points
            for i in xrange(L+1,k-s): Qw[row][i] = Rw[i-L]
    
    return UQ,VQ,Qw
    
