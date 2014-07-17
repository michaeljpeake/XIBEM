# Michael Peake
# Durham University

import numpy as np
from random import gauss




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
        self.mz = direction_of_propagation[2]
        self.k = wavenumber

    def __call__(self,x,y,z):
        return np.exp(1j*self.k*(self.mx*x+self.my*y+self.mz*z))
        

        

class NullEnrichment(object):

    def __init__(self,dof_ID):
        """Returns a null enrichment. Must be used when the enrichment is not being used"""
        self.ID = dof_ID

    def __call__(self,x,y,z):
        """Values will be discarded. Returns ones"""
        x=np.asarray(x)
        return np.ones(x.shape)
        
        
        

def Function_Variable_Approximation_Basis(mesh,num_waves_in_enrichment=False,wavenumber=None,direction_of_propagation=[1.0,0.0,0.0],method='CoulombSphere'):
    """Apply functional approximation even if not using enrichment.
    """
    # Assign IDs to shape functions
    ID=0
    nodestore=False
    for d in mesh.dList:
        for e in d.eList:            
            # Create blanks
            e.shapeFunList = [NullEnrichment(-1) for i in xrange(e.P.shape[1])]
            for i in xrange(e.P.shape[1]):
                if np.any(nodestore): 
                    r = np.sqrt(np.sum((nodestore-e.P[:,i].reshape(3,1))**2,axis=0))
                    if np.any(r<1e-10): e.shapeFunList[i].ID = np.where(r<1e-10)[0][0]
                    else:
                        e.shapeFunList[i].ID = ID
                        ID += 1
                        nodestore = np.hstack([nodestore,e.P[:,i].reshape(3,1)])
                else:
                    e.shapeFunList[i].ID = ID
                    ID += 1
                    nodestore = e.P[:,i].reshape(3,1)

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
        if method=='CoulombSphere':
            mxyz = CoulombSphere(num_waves_in_enrichment,direction_of_propagation)
        if method=='StructuredGrid':
            num_waves_in_enrichment = int(np.ceil(np.sqrt((num_waves_in_enrichment-2.0)/6.0)))
            mxyz = StructuredGrid(num_waves_in_enrichment,direction_of_propagation)
            num_waves_in_enrichment = 6*num_waves_in_enrichment**2+2
        for e in mesh.eList:
            for s in e.shapeFunList:
                ID = s.ID * num_waves_in_enrichment
                s.DegreesOfFreedomList = [PlanewaveEnrichment(ID+i,mxyz[:,i],wavenumber) for i in xrange(num_waves_in_enrichment)]
                s.M = num_waves_in_enrichment

                        
    mesh.ndof = np.max([dof.ID for e in mesh.eList for s in e.shapeFunList for dof in s.DegreesOfFreedomList]) + 1




#####################################
#        POINTS ON A SPHERE         #
#####################################

def CoulombSphere(N,fixed_point=[1,0,0]):
    """ Uniformly places points over a unit sphere using
    Coulomb's law. One point is fixed in a prescribed direction
    N : number of points
    """
    positions = np.array([gauss(0,1) for i in xrange(3*N)])
    positions = positions.reshape(3,-1)
    positions /= np.sqrt(np.sum(positions**2,axis=0)) # Force r=1
    positions[:,0] = fixed_point
    velocity = np.zeros((3,N)) # initial velocites are zero
    
    num_time_steps = 500
    coulomb_force_const = 100
    damping = 10
    dt = 0.01
    
    for step in xrange(num_time_steps):
        
        # Calculate the force on each charge
        force = np.zeros((3,N))
        px,py,pz=positions
        px=px.reshape(-1,1) ; py=py.reshape(-1,1) ; pz=pz.reshape(-1,1)
        qx,qy,qz=positions
        rx=px-qx ; ry=py-qy ; rz=pz-qz
        rx=np.tril(rx, -1)[:,:-1]+np.triu(rx, 1)[:,1:]
        ry=np.tril(ry, -1)[:,:-1]+np.triu(ry, 1)[:,1:]
        rz=np.tril(rz, -1)[:,:-1]+np.triu(rz, 1)[:,1:]
        rmag = np.sqrt(rx**2 + ry**2 + rz**2)

        force[0,:] = np.sum(rx/rmag**3,axis=1) * coulomb_force_const
        force[1,:] = np.sum(ry/rmag**3,axis=1) * coulomb_force_const
        force[2,:] = np.sum(rz/rmag**3,axis=1) * coulomb_force_const
        force[:,0]=0.0

        # Project force on each mass onto sphere surface
        Q = np.vstack([np.cross(positions[:,i],force[:,i]) for i in xrange(N)])
        Q = Q.T
        R = np.vstack([np.cross(Q[:,i],positions[:,i]) for i in xrange(N)])
        force = R.T

        # Calculate acceleration of each mass and update vel and pos variables
        acceleration = force - damping*velocity
        velocity += acceleration*dt
        positions += velocity*dt

        # Snap positions back to sphere surface
        positions /= np.sqrt(np.sum(positions**2,axis=0))
        positions /= np.sqrt(np.sum(positions**2,axis=0))
        
    return positions


def StructuredGrid(n,incident_direction=[1,0,0]):
    """Find 6n^2 + 2 equally spaced wave directions on a cube grid"""

    nsq = (n+1)*(n+1)
    directions = np.zeros((6*nsq,3))

    # 2D grid for each plane
    ax1,ax2 = np.meshgrid(np.linspace(-1,1,n+1),np.linspace(-1,1,n+1))
    ax1 = ax1.reshape(-1,)
    ax2 = ax2.reshape(-1,)

    # x = +1
    directions[0*nsq:1*nsq,0] = -1*np.ones((nsq,))
    directions[0*nsq:1*nsq,1] = ax1
    directions[0*nsq:1*nsq,2] = ax2

    # x = -1
    directions[1*nsq:2*nsq,0] = +1*np.ones((nsq,))
    directions[1*nsq:2*nsq,1] = ax1
    directions[1*nsq:2*nsq,2] = ax2

    # y = +1
    directions[2*nsq:3*nsq,0] = ax1
    directions[2*nsq:3*nsq,1] = -1*np.ones((nsq,))
    directions[2*nsq:3*nsq,2] = ax2

    # y = -1
    directions[3*nsq:4*nsq,0] = ax1
    directions[3*nsq:4*nsq,1] = +1*np.ones((nsq,))
    directions[3*nsq:4*nsq,2] = ax2

    # z = +1
    directions[4*nsq:5*nsq,0] = ax1
    directions[4*nsq:5*nsq,1] = ax2
    directions[4*nsq:5*nsq,2] = -1*np.ones((nsq,))

    # z = -1
    directions[5*nsq:6*nsq,0] = ax1
    directions[5*nsq:6*nsq,1] = ax2
    directions[5*nsq:6*nsq,2] = +1*np.ones((nsq,))

    # Remove duplicates
    # n.b. this does not preserve the order of the points
    directions = np.array([np.array(x) for x in set(tuple(x) for x in directions)])

    # Normalise directions
    for i in xrange(directions.shape[0]):
        mag = np.sqrt(np.sum(directions[i]**2))
        directions[i] /= mag
        
    directions=directions.T
    
    # Rotate
    def rotation_matrix(axis,theta):
        axis = axis/np.sqrt(np.dot(axis,axis))
        a = np.cos(theta/2)
        b,c,d = -axis*np.sin(theta/2)
        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    A = rotation_matrix(np.array([0.0,1.0,0.0]),-np.pi/4)
    B = rotation_matrix(np.array([0.0,0.0,1.0]),np.arcsin(1/np.sqrt(3)))
    rotation_matrix = np.dot(B,A)
    directions = np.dot(rotation_matrix,directions)
    
    directions /= np.sqrt(np.sum(directions**2,axis=0)) # Normalise
     
    return directions
