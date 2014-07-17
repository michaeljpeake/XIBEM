 #Michael Peake
# Durham University

import numpy as np




class Shape_Function(object):

    def __init__(self,ID):
        """A class to store shape function ID and to apply enrichment to later."""
        self.ID = ID



class PlanewaveEnrichment(object):
    
    def __init__(self,degree_of_freedom_ID,direction_of_propagation,wavenumber):
        """Returns a planewave enrichment to multiply by shape functions.
        """
        self.ID = degree_of_freedom_ID
        self.mx = direction_of_propagation[0]
        self.my = direction_of_propagation[1]
        self.k = wavenumber

    def __call__(self,x,y):
        return np.exp(1j*self.k*(self.mx*x+self.my*y))




class NullEnrichment(object):

    def __init__(self,dof_ID):
        """Returns a null enrichment. Must be used when the enrichment is not being used"""
        self.ID = dof_ID

    def __call__(self,x,y):
        """Values will be discarded. Returns ones"""
        x=np.asarray(x)
        return np.ones(x.shape)




def Function_Variable_Approximation_Basis(mesh,num_waves_in_enrichment=False,wavenumber=None,direction_of_propagation=None):
    """Apply functional approximation even if not using enrichment.
    """

    # Assign IDs to shape functions
    ID = 0
    for d in mesh.dList:
        for e in d.eList:
            e.shapeFunList = [Shape_Function(ID+i) for i in xrange(e.P.shape[1])]
            ID += e.P.shape[1] - 1
        ID = d.eList[-1].shapeFunList[-1].ID
        d.eList[-1].shapeFunList[-1].ID = d.eList[0].shapeFunList[0].ID

    if num_waves_in_enrichment == False:
        mesh.enriched=False
        for e in mesh.eList:
            for s in e.shapeFunList:
                s.DegreesOfFreedomList = [NullEnrichment(s.ID)]
                s.M = 1
    else:
        if wavenumber==None: raise "Must supply wavenumber, k"
        if direction_of_propagation==None: raise "Must supply direction of propagation of incident planewave"
        
        mesh.enriched=True
        theta = np.linspace(0,2*np.pi,num_waves_in_enrichment,endpoint=False)
        theta += np.arctan2(direction_of_propagation[1],direction_of_propagation[0])
        mxy = np.vstack([np.cos(theta),np.sin(theta)]).T
                
        for e in mesh.eList:
            for s in e.shapeFunList:
                ID = s.ID * num_waves_in_enrichment
                s.DegreesOfFreedomList = [PlanewaveEnrichment(ID+i,mxy[i],wavenumber) for i in xrange(num_waves_in_enrichment)]
                s.M = num_waves_in_enrichment
                
    mesh.ndof = np.max([dof.ID for e in mesh.eList for s in e.shapeFunList for dof in s.DegreesOfFreedomList]) + 1
