# Michael Peake
# Durham University

import numpy as np
   
def Uniform_Boundary_Evalutation(Domain,DoF_Coefficients,num_points,return_points=False):
    """Evaluate potential at num_points uniformly spaced over a boundary.
    Note: Send one domain (boundary) at a time.
    """
    # This code can be changed to evaluate all the boundaries at once but I did
    # it this way so the boundary of individual scatterers could be investigated
    # independently
    
    potentials=[]
    pxlist=[]
    pylist=[]
        
    s = np.linspace(0,Domain.length(),num_points,endpoint=False)
    slow=0
    shigh=0

    for e in Domain.eList:

        DoF_mapping = [dof.ID for shapefun in e.shapeFunList for dof in shapefun.DegreesOfFreedomList]

        shigh += e.length()
        sToPlot = np.where((slow<=s) == (s<shigh)) # Find values of s that are on this element
        xi = e.limits[0] + (e.limits[1]-e.limits[0])*(s[sToPlot]-slow)/e.length() # Find values of xi for s
        slow += e.length()
        
        Basis = e.shape_functions(xi)[:,:,0]
        if e.ExactElement:
            p = e.vals(xi)
        else:
            p = np.dot(e.P,Basis.T)
            
        Basis = np.repeat(Basis,e.shapeFunList[0].M,axis=1) # Assumes global M !!!!
        Basis = np.asarray(Basis,np.complex) # Make InterpolationBasis a complex matrix
        Basis *= np.vstack([dof(p[0],p[1]) for shapefun in e.shapeFunList for dof in shapefun.DegreesOfFreedomList]).T
        
        pxlist = np.append(pxlist,p[0])
        pylist = np.append(pylist,p[1])

        element_coefficients = DoF_Coefficients[DoF_mapping].reshape(1,-1)
        potentials = np.append(potentials,np.dot(element_coefficients,Basis.T))
            
    if return_points: return potentials,np.vstack([pxlist,pylist])
    return potentials