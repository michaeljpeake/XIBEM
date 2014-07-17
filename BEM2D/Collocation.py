# Michael Peake
# Durham University

import numpy as np
from random import uniform
from matplotlib.path import Path

def EquallySpacedInLocalCoordinate(mesh,points_per_element=None):

    for e in mesh.eList:
        if points_per_element == None:
            points_per_element = np.sum([s.M for s in e.shapeFunList][:-1])
            # [:-1] above ignores last shape function so exact  
            # number of collocation points required are obtained
        e.collocationXi = np.linspace(e.limits[0],e.limits[1], points_per_element ,endpoint=False)
        e.collocation_points = e.vals(e.collocationXi)

    # List collocation points for each domain
    for d in mesh.dList:
        d.collocation_points = np.hstack([e.collocation_points for e in d.eList])
        
    # List all collocation points in mesh
    mesh.collocation_points = np.hstack([d.collocation_points for d in mesh.dList])
    
    mesh.numBoundaryCollocation = mesh.collocation_points.shape[1]
    
    
    

def CHIEF(mesh,number_of_CHIEF_points=None,fraction_extra_collocation=None):
    """
    Finds CHIEF collocation points to put inside each domain.
    Must supply either number_of_CHIEF_points or a fraction_extra_collocation.
    This will be applied to each domain seperately.
    """
    
    if number_of_CHIEF_points==None and fraction_extra_collocation==None:
        raise "Must supply number of CHIEF points or fraction of extra collcation points"

    for d in mesh.dList:
        
        if fraction_extra_collocation != None:
            number_of_CHIEF_points = int(np.ceil(fraction_extra_collocation*d.collocation_points.shape[1]))

        boundary = Path(d.collocation_points.T)
        xmin,ymin = np.min(d.collocation_points,axis=1)
        xmax,ymax = np.max(d.collocation_points,axis=1)
        centre = np.mean(d.collocation_points,axis=1).reshape(2,1)
        
        CHIEF_points=np.vstack([0.0,0.0]) # temporary CHIEF point
        
        while CHIEF_points.shape[1] <= number_of_CHIEF_points:
            # Generate a random point inside the polygon minima and maxima
            new_point = np.vstack([uniform(xmin,xmax),uniform(ymin,ymax)])
            # Check whether the point is inside the polygon
            inside = boundary.contains_point(new_point)
            if inside:
                # Pull point away from edge of polygon
                offset = 0.1
                r = np.sqrt(np.sum((centre-new_point)**2))
                new_point -= offset * (new_point-centre) / r
                # Make sure point isn't too close to an existing one (0.05 is current limit)
                if np.all(np.sqrt(np.sum((CHIEF_points-new_point)**2,axis=0)) > 0.05):
                    CHIEF_points = np.hstack([CHIEF_points,new_point])
                    
        CHIEF_points=CHIEF_points[:,1:] # Removes tempoary CHIEF point
        
        d.numCHIEF = number_of_CHIEF_points
        # Add CHIEF points to list of mesh collocation points
        mesh.collocation_points = np.hstack([mesh.collocation_points,CHIEF_points])

    mesh.numCHIEF = sum([d.numCHIEF for d in mesh.dList])

    
