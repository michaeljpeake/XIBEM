# Michael Peake
# Durham University

import numpy as np
from random import uniform


def collocate_at_nodes(mesh):

    for d in mesh.dList:
        # Set up collXYZ with an initial value
        coll_store=d.eList[0].vals(d.eList[0].limits[0],d.eList[0].limits[2])
        
        for e in d.eList:
            
            xi1 = np.linspace(e.limits[0],e.limits[1],e.m+1)
            xi2 = np.linspace(e.limits[2],e.limits[3],e.n+1)
            xi1,xi2=np.meshgrid(xi1,xi2)
            e.collocationXi = np.vstack([xi1.reshape(-1,),xi2.reshape(-1,)])
            
            if e.__class__.__name__ == 'SerendipityQuadraticElement':
                e.collocationXi = np.delete(e.collocationXi,4,axis=1)
            
            e.collocation_points = e.vals(e.collocationXi[0],e.collocationXi[1])
            
            # Check for repeated points in new set
            px,py,pz = e.collocation_points
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)
            qx,qy,qz=e.collocation_points
            rx=px-qx ; ry=py-qy ; rz=pz-qz
            rx=np.tril(rx, -1)[:,:-1]+np.triu(rx, 1)[:,1:]
            ry=np.tril(ry, -1)[:,:-1]+np.triu(ry, 1)[:,1:]
            rz=np.tril(rz, -1)[:,:-1]+np.triu(rz, 1)[:,1:]
            r = np.sqrt( rx**2 + ry**2 + rz**2 )
            delete = np.where(np.any(r<1e-10,axis=1))[0]
            e.collocation_points = np.delete(e.collocation_points,delete[1:],axis=1)
            e.collocationXi = np.delete(e.collocationXi,delete[1:],axis=1)
            
            # Check for repeated points in large set
            px,py,pz = e.collocation_points
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)
            
            qx,qy,qz=coll_store
    
            r = np.sqrt( (qx-px)**2 + (qy-py)**2 + (qz-pz)**2 )
            delete = np.where(np.any(r<1e-10,axis=1))[0]
            
            coll_store = np.hstack([coll_store,np.delete(e.collocation_points,delete,axis=1)])
            
        d.collocation_points = coll_store

    # Apply to mesh
    mesh.collocation_points = np.hstack([d.collocation_points for d in mesh.dList])
    mesh.numBoundaryCollocation = mesh.collocation_points.shape[1]

   
    
def enriched_collocation(mesh,N=0):
    """Unless N is defined, collocates using an NxN grid such that 
    there are more collocation points than degrees of freedom.
    """ 
        
    for dom in mesh.dList:
        coll_store=dom.eList[0].vals(dom.eList[0].limits[0],dom.eList[0].limits[2])
        
        def number_of_collocation_points(N):
            a = dom.numElements * N**2
            b = dom.edges * (N-2)
            c = dom.corners * 3
            d = dom.extraordinary_points
            return a-b-c+d 

        for e in mesh.eList:
            
            if N == 0:
                xiN = 1
                while number_of_collocation_points(xiN) <= mesh.ndof: xiN+=1
            else: xiN=N
 
            while 1:
        
                xi1 = np.linspace(e.limits[0],e.limits[1],xiN)
                xi2 = np.linspace(e.limits[2],e.limits[3],xiN)
                xi1,xi2=np.meshgrid(xi1,xi2)
                xi1=xi1.reshape(-1,) ; xi2=xi2.reshape(-1,)
                
                e.collocationXi = np.vstack([xi1,xi2])
                e.collocation_points = e.vals(e.collocationXi[0],e.collocationXi[1])
                    
                # Check for repeated points in new set
                px,py,pz = e.collocation_points
                px=px.reshape(-1,1)
                py=py.reshape(-1,1)
                pz=pz.reshape(-1,1)
                qx,qy,qz=e.collocation_points
                rx=px-qx ; ry=py-qy ; rz=pz-qz
                rx=np.tril(rx, -1)[:,:-1]+np.triu(rx, 1)[:,1:]
                ry=np.tril(ry, -1)[:,:-1]+np.triu(ry, 1)[:,1:]
                rz=np.tril(rz, -1)[:,:-1]+np.triu(rz, 1)[:,1:]
                r = np.sqrt( rx**2 + ry**2 + rz**2 )
                delete = np.where(np.any(r<1e-10,axis=1))[0]
                e.collocation_points = np.delete(e.collocation_points,delete[1:],axis=1)
                e.collocationXi = np.delete(e.collocationXi,delete[1:],axis=1)
                
                if e.collocationXi.shape[1]>=N**2: break
                else: xiN+=1
            
            # Check for repeated points in large set
            px,py,pz = e.collocation_points
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)        
            qx,qy,qz=coll_store
            r = np.sqrt( (qx-px)**2 + (qy-py)**2 + (qz-pz)**2 )
            delete = np.where(np.any(r<1e-12,axis=1))[0]
            
            coll_store = np.hstack([coll_store,np.delete(e.collocation_points,delete,axis=1)])

            dom.collocation_points = coll_store

    # Apply to mesh
    mesh.collocation_points = np.hstack([d.collocation_points for d in mesh.dList])
    mesh.numBoundaryCollocation = mesh.collocation_points.shape[1]
    
        
        
        
########################################
#                CHIEF                 #
########################################

# I have not developed an algorithm that works for arbitrary scatterers yet
# Instead, I have CHIEF schemes that work for a unit sphere and for a torus geometry
  
def CHIEF_sphere(mesh,number_of_CHIEF_points=None,fraction_extra_collocation=None):
    """Finds CHIEF collocation points to put inside a sphere.
    Must supply either number_of_CHIEF_points or a fraction_extra_collocation.
    Should work for multiple cylinders with a little alteration for the difference
    centres.
    """

    if number_of_CHIEF_points==None and fraction_extra_collocation==None:
        raise "Must supply number of CHIEF points or fraction of extra collcation points"
        
    for d in mesh.dList:
        
        if fraction_extra_collocation != None:
            number_of_CHIEF_points = int(np.ceil(fraction_extra_collocation*d.collocation_points.shape[1]))
            
        xmin,xmax = -0.9,0.9
        ymin,ymax = -0.9,0.9
        zmin,zmax = -0.9,0.9       
        centre = np.array([0.0,0.0,0.0]).reshape(3,1)         
            
        CHIEF_points=np.vstack([0.0,0.0,0.0]) # Temporary point
        while CHIEF_points.shape[1]<=number_of_CHIEF_points:
            # Generate a random point
            p = np.vstack([uniform(xmin,xmax),uniform(ymin,ymax),uniform(zmin,zmax)])
                       
            # Check whether the point is inside
            r = np.sqrt(np.sum(p**2))
            if r < 0.9: inside = True
            else: inside = False
                
            if inside:
                # Translate point using centre of sphere
                p += centre
                # Make sure point isn't too close to an existing one         
                if np.all(np.sqrt(np.sum((CHIEF_points-p)**2,axis=0)) > 0.05):
                    CHIEF_points = np.hstack([CHIEF_points,p])
                    
        CHIEF_points=CHIEF_points[:,1:] # Remove temporary point
        
        d.numCHIEF = number_of_CHIEF_points
        # Add CHIEF points to list of mesh collocation points
        mesh.collocation_points = np.hstack([mesh.collocation_points,CHIEF_points])
        
    mesh.numCHIEF = sum([d.numCHIEF for d in mesh.dList])


def CHIEF_torus(mesh,minor_radius,major_radius,number_of_CHIEF_points=None,fraction_extra_collocation=None):
    
    # See page 886 of logbook
    
    if number_of_CHIEF_points==None and fraction_extra_collocation==None:
        raise "Must supply number of CHIEF points or fraction of extra collcation points"
        
    for d in mesh.dList:
        
        if fraction_extra_collocation != None:
            number_of_CHIEF_points = int(np.ceil(fraction_extra_collocation*d.collocation_points.shape[1]))
            
        offset = 0.1
        zmin = -minor_radius
        zmax = minor_radius
        centre = np.array([0.0,0.0,0.0]).reshape(3,1)
        
        CHIEF_points=np.vstack([0.0,0.0,0.0])
        while CHIEF_points.shape[1]<=number_of_CHIEF_points:
            # Find Z component and magnitude of (X,Y)
            p = np.array([uniform(zmin,zmax),uniform(zmin,zmax)])
            if np.sqrt(np.sum(p**2)) < minor_radius-offset:
                mag = p[0]+major_radius
                theta = uniform(0,2*np.pi)
                x,y = mag*np.cos(theta),mag*np.sin(theta)
                p = np.vstack([x,y,p[1]])
                p+=centre
                if np.all(np.sqrt(np.sum((CHIEF_points-p)**2,axis=0)) > 0.05):
                    CHIEF_points = np.hstack([CHIEF_points,p])
                    
        CHIEF_points=CHIEF_points[:,1:] # Remove temporary point
        
        d.numCHIEF = number_of_CHIEF_points
        # Add CHIEF points to list of mesh collocation points
        mesh.collocation_points = np.hstack([mesh.collocation_points,CHIEF_points])
        
    mesh.numCHIEF = sum([d.numCHIEF for d in mesh.dList])
                
            
        



        
        


    


