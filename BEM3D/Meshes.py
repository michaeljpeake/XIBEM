 # Michael Peake
# Durham University

import numpy as np
import LagrangianElements as Lagrange
import BezierElements as Bezier




#############################################################
###                      UNIT SPHERE                      ###
#############################################################

def Sphere_CubeToStandardLagrangeQuad(N):

    def transform(x,y,z):
        xx = x**2
        yy = y**2
        zz = z**2
        xbar = x*np.sqrt(1 - yy/2 - zz/2 + yy*zz/3)
        ybar = y*np.sqrt(1 - xx/2 - zz/2 + xx*zz/3)
        zbar = z*np.sqrt(1 - xx/2 - yy/2 + xx*yy/3)
        return Lagrange.StandardQuadraticElement(np.vstack([xbar,ybar,zbar]))
    def get_coordinates(i,j,xis,etas):
        xi = np.hstack([xis[2*j:2*j+2],xis[2*j+2],xis[2*j+2],xis[2*j+2:2*j:-1],xis[2*j],xis[2*j],xis[2*j+1]])
        eta = np.hstack([etas[2*i],etas[2*i],etas[2*i:2*i+2],etas[2*i+2],etas[2*i+2],etas[2*i+2:2*i:-1],etas[2*i+1]])
        return xi,eta
    ones = np.ones(9,)
    element_list=[]
    xis = np.linspace(1,-1,2*N+1)
    etas = np.linspace(-1,1,2*N+1)
    # x=+1 side
    for i in xrange(N):
        for j in xrange(N):
            x = ones
            y,z = get_coordinates(i,j,xis,etas)
            element_list.append(transform(x,y,z))
    # y=+1 side
    for i in xrange(N):
        for j in xrange(N):
            y = ones
            x,z = get_coordinates(i,j,xis,etas)
            x=-x
            element_list.append(transform(x,y,z))
    # x=-1 side
    for i in xrange(N):
        for j in xrange(N):
            x = -ones
            y,z = get_coordinates(i,j,xis,etas)
            y=-y
            element_list.append(transform(x,y,z))
    # y=-1 side
    for i in xrange(N):
        for j in xrange(N):
            y = -ones
            x,z = get_coordinates(i,j,xis,etas)
            element_list.append(transform(x,y,z))
    # z=+1 side
    for i in xrange(N):
        for j in xrange(N):
            z = ones
            x,y = get_coordinates(i,j,xis,etas)
            x=-x ; y=-y
            element_list.append(transform(x,y,z))
    # z=-1 side
    for i in xrange(N):
        for j in xrange(N):
            z = -ones
            x,y = get_coordinates(i,j,xis,etas)
            x=-x
            element_list.append(transform(x,y,z))
    mesh = Lagrange.Mesh([Lagrange.Domain(element_list)])
    # For determining points created using NxN grid on each element
    mesh.dList[0].edges = mesh.numElements * 2
    mesh.dList[0].corners = 6*N**2 + 2
    mesh.dList[0].extraordinary_points = 8
    return mesh


def Sphere_CubeToSerendipityLagrangeQuad(N):

    def transform(x,y,z):
        xx = x**2
        yy = y**2
        zz = z**2
        xbar = x*np.sqrt(1 - yy/2 - zz/2 + yy*zz/3)
        ybar = y*np.sqrt(1 - xx/2 - zz/2 + xx*zz/3)
        zbar = z*np.sqrt(1 - xx/2 - yy/2 + xx*yy/3)
        return Lagrange.SerendipityQuadraticElement(np.vstack([xbar,ybar,zbar]))
    def get_coordinates(i,j,xis,etas):
        xi = np.hstack([xis[2*j:2*j+2],xis[2*j+2],xis[2*j+2],xis[2*j+2:2*j:-1],xis[2*j],xis[2*j]])
        eta = np.hstack([etas[2*i],etas[2*i],etas[2*i:2*i+2],etas[2*i+2],etas[2*i+2],etas[2*i+2:2*i:-1]])
        return xi,eta
    ones = np.ones(8,)
    element_list=[]
    xis = np.linspace(1,-1,2*N+1)
    etas = np.linspace(-1,1,2*N+1)
    # x=+1 side
    for i in xrange(N):
        for j in xrange(N):
            x = ones
            y,z = get_coordinates(i,j,xis,etas)
            element_list.append(transform(x,y,z))
    # y=+1 side
    for i in xrange(N):
        for j in xrange(N):
            y = ones
            x,z = get_coordinates(i,j,xis,etas)
            x=-x
            element_list.append(transform(x,y,z))
    # x=-1 side
    for i in xrange(N):
        for j in xrange(N):
            x = -ones
            y,z = get_coordinates(i,j,xis,etas)
            y=-y
            element_list.append(transform(x,y,z))
    # y=-1 side
    for i in xrange(N):
        for j in xrange(N):
            y = -ones
            x,z = get_coordinates(i,j,xis,etas)
            element_list.append(transform(x,y,z))
    # z=+1 side
    for i in xrange(N):
        for j in xrange(N):
            z = ones
            x,y = get_coordinates(i,j,xis,etas)
            x=-x ; y=-y
            element_list.append(transform(x,y,z))
    # z=-1 side
    for i in xrange(N):
        for j in xrange(N):
            z = -ones
            x,y = get_coordinates(i,j,xis,etas)
            x=-x
            element_list.append(transform(x,y,z))
    mesh = Lagrange.Mesh([Lagrange.Domain(element_list)])
    # For determining points created using NxN grid on each element
    mesh.dList[0].edges = mesh.numElements * 2
    mesh.dList[0].corners = 6*N**2 + 2
    mesh.dList[0].extraordinary_points = 8
    return mesh


def Sphere_ExactSerendipityLagrangeQuad():
    """This creates a six element sphere from a cube. Geometry points are
    calculated analytically"""

    mesh = Sphere_CubeToSerendipityLagrangeQuad(1)
    
    ################
    # Modifications for exact sphere
    ################
    # x=+1 side
    def posXvals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.ones(xi1.shape);yb=np.array(-xi1);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;
        x = xb*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        y = yb*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        z = zb*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        return np.vstack([x,y,z])
    def posXnormals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.ones(xi1.shape);yb=np.array(-xi1);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;
        dxdxi1 = -1.0 * xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-yb+2.0*yb*zz/3.0)
        dxdxi2 = 1.0 * xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-zb+2.0*yy*zb/3.0)
        dydxi1 = -1.0 * np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        dydxi2 = 1.0 * yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-zb+2.0*xx*zb/3.0)
        dzdxi1 = -1.0 * zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-yb+2.0*xx*yb/3.0)
        dzdxi2 = 1.0 * np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        magnitude = np.sqrt(np.sum(J**2,axis=1))
        return J.T/magnitude
    def posXJ(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.ones(xi1.shape);yb=np.array(-xi1);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;
        dxdxi1 = -1.0 * xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-yb+2.0*yb*zz/3.0)
        dxdxi2 = 1.0 * xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-zb+2.0*yy*zb/3.0)
        dydxi1 = -1.0 * np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        dydxi2 = 1.0 * yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-zb+2.0*xx*zb/3.0)
        dzdxi1 = -1.0 * zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-yb+2.0*xx*yb/3.0)
        dzdxi2 = 1.0 * np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        return np.sqrt(np.sum(J**2,axis=1))
    mesh.eList[0].vals = posXvals
    mesh.eList[0].normals = posXnormals
    mesh.eList[0].J = posXJ
        
    def posYvals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(xi1);yb=np.ones(xi1.shape);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;
        x = xb*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        y = yb*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        z = zb*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        return np.vstack([x,y,z])
    def posYnormals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(xi1);yb=np.ones(xi1.shape);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;
        dxdxi1 = 1.0*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        dxdxi2 = 1.0*xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-zb+2.0*yy*zb/3.0)
        dydxi1 = 1.0*yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-xb+2.0*xb*zz/3.0)
        dydxi2 = 1.0*yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-zb+2.0*xx*zb/3.0)
        dzdxi1 = 1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-xb+2.0*xb*yy/3.0)
        dzdxi2 = 1.0*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        magnitude = np.sqrt(np.sum(J**2,axis=1))
        return J.T/magnitude
    def posYJ(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(xi1);yb=np.ones(xi1.shape);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;
        dxdxi1 = 1.0*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        dxdxi2 = 1.0*xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-zb+2.0*yy*zb/3.0)
        dydxi1 = 1.0*yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-xb+2.0*xb*zz/3.0)
        dydxi2 = 1.0*yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-zb+2.0*xx*zb/3.0)
        dzdxi1 = 1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-xb+2.0*xb*yy/3.0)
        dzdxi2 = 1.0*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        return np.sqrt(np.sum(J**2,axis=1))
    mesh.eList[1].vals = posYvals
    mesh.eList[1].normals = posYnormals
    mesh.eList[1].J = posYJ
    
    # x=-1 side
    def negXvals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=-np.ones(xi1.shape);yb=np.array(xi1);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;
        x = xb*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        y = yb*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        z = zb*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        return np.vstack([x,y,z])
    def negXnormals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=-np.ones(xi1.shape);yb=np.array(xi1);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;
        dxdxi1 = 1.0*xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-yb+2.0*yb*zz/3.0)
        dxdxi2 = 1.0*xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-zb+2.0*yy*zb/3.0)
        dydxi1 = 1.0*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        dydxi2 = 1.0*yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-zb+2.0*xx*zb/3.0)
        dzdxi1 = 1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-yb+2.0*xx*yb/3.0)
        dzdxi2 = 1.0*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        magnitude = np.sqrt(np.sum(J**2,axis=1))
        return J.T/magnitude
    def negXJ(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=-np.ones(xi1.shape);yb=np.array(xi1);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;
        dxdxi1 = 1.0*xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-yb+2.0*yb*zz/3.0)
        dxdxi2 = 1.0*xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-zb+2.0*yy*zb/3.0)
        dydxi1 = 1.0*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        dydxi2 = 1.0*yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-zb+2.0*xx*zb/3.0)
        dzdxi1 = 1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-yb+2.0*xx*yb/3.0)
        dzdxi2 = 1.0*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        return np.sqrt(np.sum(J**2,axis=1))
    mesh.eList[2].vals = negXvals
    mesh.eList[2].normals = negXnormals
    mesh.eList[2].J = negXJ

    # y=-1 side
    def negYvals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(-xi1);yb=-np.ones(xi1.shape);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;
        x = xb*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        y = yb*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        z = zb*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        return np.vstack([x,y,z])
    def negYnormals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(-xi1);yb=-np.ones(xi1.shape);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;       
        dxdxi1 = -1.0*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        dxdxi2 = 1.0*0.5*xb*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5*(-zb+2.0*yy*zb/3.0)
        dydxi1 = -1.0*0.5*yb*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5*(-xb+2.0*xb*zz/3.0)
        dydxi2 = 1.0*0.5*yb*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5*(-zb+2.0*xx*zb/3.0)
        dzdxi1 = -1.0*0.5*zb*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5*(-xb+2.0*xb*yy/3.0)
        dzdxi2 = 1.0*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)        
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        magnitude = np.sqrt(np.sum(J**2,axis=1))
        return J.T/magnitude
    def negYJ(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(-xi1);yb=-np.ones(xi1.shape);zb=np.array(xi2);
        xx=xb**2;yy=yb**2;zz=zb**2;       
        dxdxi1 = -1.0*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        dxdxi2 = 1.0*0.5*xb*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5*(-zb+2.0*yy*zb/3.0)
        dydxi1 = -1.0*0.5*yb*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5*(-xb+2.0*xb*zz/3.0)
        dydxi2 = 1.0*0.5*yb*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5*(-zb+2.0*xx*zb/3.0)
        dzdxi1 = -1.0*0.5*zb*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5*(-xb+2.0*xb*yy/3.0)
        dzdxi2 = 1.0*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)        
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        return np.sqrt(np.sum(J**2,axis=1))
    mesh.eList[3].vals = negYvals
    mesh.eList[3].normals = negYnormals
    mesh.eList[3].J = negYJ
    
    # z=+1 side
    def posZvals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(xi1)
        yb=np.array(-xi2)
        zb=np.ones(xi1.shape)
        xx=xb**2;yy=yb**2;zz=zb**2;
        x = xb*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        y = yb*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        z = zb*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        return np.vstack([x,y,z])
    def posZnormals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(xi1);yb=np.array(-xi2);zb=np.ones(xi1.shape);
        xx=xb**2;yy=yb**2;zz=zb**2;
        dxdxi1 = 1.0*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        dxdxi2 = -1.0*xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-yb+2.0*yb*zz/3.0)
        dydxi1 = 1.0*yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-xb+2.0*xb*zz/3.0)
        dydxi2 = -1.0*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        dzdxi1 = 1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-xb+2.0*xb*yy/3.0)
        dzdxi2 = -1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-yb+2.0*xx*yb/3.0)
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        magnitude = np.sqrt(np.sum(J**2,axis=1))
        return J.T/magnitude
    def posZJ(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(xi1);yb=np.array(-xi2);zb=np.ones(xi1.shape);
        xx=xb**2;yy=yb**2;zz=zb**2;
        dxdxi1 = 1.0*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        dxdxi2 = -1.0*xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-yb+2.0*yb*zz/3.0)
        dydxi1 = 1.0*yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-xb+2.0*xb*zz/3.0)
        dydxi2 = -1.0*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        dzdxi1 = 1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-xb+2.0*xb*yy/3.0)
        dzdxi2 = -1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-yb+2.0*xx*yb/3.0)
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        return np.sqrt(np.sum(J**2,axis=1))
    mesh.eList[4].vals = posZvals
    mesh.eList[4].normals = posZnormals
    mesh.eList[4].J = posZJ
    
    # z=-1 side
    def negZvals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(xi1);yb=np.array(xi2);zb=-np.ones(xi1.shape);
        xx=xb**2;yy=yb**2;zz=zb**2;
        x = xb*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        y = yb*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        z = zb*np.sqrt(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)
        return np.vstack([x,y,z])
    def negZnormals(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(xi1);yb=np.array(xi2);zb=-np.ones(xi1.shape);
        xx=xb**2;yy=yb**2;zz=zb**2;
        dxdxi1 = 1.0*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        dxdxi2 = 1.0*xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-yb+2.0*yb*zz/3.0)
        dydxi1 = 1.0*yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-xb+2.0*xb*zz/3.0)
        dydxi2 = 1.0*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        dzdxi1 = 1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-xb+2.0*xb*yy/3.0)
        dzdxi2 = 1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-yb+2.0*xx*yb/3.0)
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        magnitude = np.sqrt(np.sum(J**2,axis=1))
        return J.T/magnitude
    def negZJ(xi1,xi2):
        xi1=np.asarray(xi1,np.float).reshape(-1,)
        xi2=np.asarray(xi2,np.float).reshape(-1,)
        xb=np.array(xi1);yb=np.array(xi2);zb=-np.ones(xi1.shape);
        xx=xb**2;yy=yb**2;zz=zb**2;
        dxdxi1 = 1.0*np.sqrt(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)
        dxdxi2 = 1.0*xb*0.5*(1.0 - yy/2.0 - zz/2.0 + yy*zz/3.0)**-0.5 * (-yb+2.0*yb*zz/3.0)
        dydxi1 = 1.0*yb*0.5*(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)**-0.5 * (-xb+2.0*xb*zz/3.0)
        dydxi2 = 1.0*np.sqrt(1.0 - xx/2.0 - zz/2.0 + xx*zz/3.0)
        dzdxi1 = 1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-xb+2.0*xb*yy/3.0)
        dzdxi2 = 1.0*zb*0.5*(1.0 - xx/2.0 - yy/2.0 + xx*yy/3.0)**-0.5 * (-yb+2.0*xx*yb/3.0)
        J = np.array([[dxdxi1,dxdxi2],[dydxi1,dydxi2],[dzdxi1,dzdxi2]]).T
        J = np.cross(J[:,0,:],J[:,1,:])
        return np.sqrt(np.sum(J**2,axis=1))
    mesh.eList[5].vals = negZvals
    mesh.eList[5].normals = negZnormals
    mesh.eList[5].J = negZJ
    
    for e in mesh.eList:
        e.ExactElement = True
    
    return mesh


def Sphere_Bezier(refinements=0):
    """8 element sphere. Each element can be split into
    N x N grids of elements"""
    
    S = np.array([0,0,0])
    T = np.array([0,0,1])
    V = np.array([0,0,0,0.5,0.5,1,1,1])
    q = 2
    m = 4
    Pj = np.array([[0,0,1],[1,0,1],[1,0,0],[1,0,-1],[0,0,-1]])
    wj = np.array([1,1/np.sqrt(2),1,1/np.sqrt(2),1])

    sphere = Bezier.MakeRevolvedSurface(S,T,2*np.pi,q,V,m,Pj,wj)
    
    # Make knot refinements before decomposing
    if refinements:
        uniqueU = np.unique(sphere.U)
        for i in xrange(uniqueU.size-1):
            XI = np.linspace(uniqueU[i],uniqueU[i+1],refinements+2)[1:-1]
            sphere.refineknotvector(XI,'U')
        uniqueV = np.unique(sphere.V)
        for i in xrange(uniqueV.size-1):
            XI = np.linspace(uniqueV[i],uniqueV[i+1],refinements+2)[1:-1]
            sphere.refineknotvector(XI,'V')
               
    sphere=sphere.decompose()

    element_list = [Bezier.BezierElement(sphere[i,:,:,:]) for i in xrange(sphere.shape[0])]
    
    
    sphere_domain = Bezier.Domain(element_list)
    mesh = Bezier.Mesh([sphere_domain])
    
    mesh.dList[0].edges = mesh.numElements * 2
    mesh.dList[0].corners = 8
    mesh.dList[0].extraordinary_points = 8

    
    return mesh








#############################################################
###                        TORUS                          ###
#############################################################


def Torus_Bezier(xsec_refinements=0,circum_refinements=1,r=0.5,R=1.0):
    
    if r>=R: raise "R must be greater than r"
    
    q=2
    Pj = np.array(([r,0,0],[r,0,-r],[0,0,-r],
                  [-r,0,-r],[-r,0,0],[-r,0,r],
                  [0,0,r],[r,0,r],[r,0,0]),np.float)
    Pj[:,0] += R
    wj = np.array([1,1/np.sqrt(2),1,1/np.sqrt(2),1,1/np.sqrt(2),1,1/np.sqrt(2),1])
    V = np.array([0,0,0,0.25,0.25,0.5,0.5,0.75,0.75,1,1,1])
    m = (V.size-1) - q - 1

    S = np.array([0,0,0])
    T = np.array([0,0,1])
    
    torus = Bezier.MakeRevolvedSurface(S,T,2*np.pi,q,V,m,Pj,wj)
    
    # Make knot refinements before decomposing
    if circum_refinements:
        uniqueU = np.unique(torus.U)
        for i in xrange(uniqueU.size-1):
            XI = np.linspace(uniqueU[i],uniqueU[i+1],circum_refinements+2)[1:-1]
            torus.refineknotvector(XI,'U')
    if xsec_refinements:
        uniqueV = np.unique(torus.V)
        for i in xrange(uniqueV.size-1):
            XI = np.linspace(uniqueV[i],uniqueV[i+1],xsec_refinements+2)[1:-1]
            torus.refineknotvector(XI,'V')
               
    torus=torus.decompose()

    element_list = [Bezier.BezierElement(torus[i,:,:,:]) for i in xrange(torus.shape[0])]
    
    torus_domain = Bezier.Domain(element_list)
    mesh = Bezier.Mesh([torus_domain])
    
    mesh.dList[0].edges = 32*(1+xsec_refinements)*(1+circum_refinements)
    mesh.dList[0].corners = 16*(1+xsec_refinements)*(1+circum_refinements)
    mesh.dList[0].extraordinary_points = 0
    
    return mesh


def Torus_Quad(xsec_refinements=0,circum_refinements=1,r=0.5,R=1.0,element_type='Standard'):
    
    if element_type!= 'Standard' and element_type!='Serendipity':
        raise "element_type must be Standard or Serendipity"
    mesh = Torus_Bezier(xsec_refinements,circum_refinements,r,R)
    
    xi1=np.array([0,0.5,1,1,1,0.5,0,0])
    xi2=np.array([0,0,0,0.5,1,1,1,0.5])
    if element_type=='Standard':
        xi1=np.append(xi1,0.5)
        xi2=np.append(xi2,0.5) 
        
    
    if element_type=='Standard': elements=[Lagrange.StandardQuadraticElement(e.vals(xi1,xi2)) for e in mesh.eList]
    else: elements=[Lagrange.SerendipityQuadraticElement(e.vals(xi1,xi2)) for e in mesh.eList]

    mesh = Lagrange.Mesh([Lagrange.Domain(elements)])

    mesh.dList[0].edges = 32*(1+xsec_refinements)*(1+circum_refinements)
    mesh.dList[0].corners = 16*(1+xsec_refinements)*(1+circum_refinements)
    mesh.dList[0].extraordinary_points = 0
        
    return mesh
    
    
def Torus_PUBEM(xsec_refinements=0,circum_refinements=1,r=0.5,R=1.0,element_type='Serendipity'):
    
    if element_type!= 'Standard' and element_type!='Serendipity':
        raise "element_type must be Standard or Serendipity"
    mesh = Torus_Bezier(xsec_refinements,circum_refinements,r,R)
    
    elements=[]
    for e in mesh.eList:
        p1=e.vals(0,0)
        p2=e.vals(1,1)
        #theta2limits = [np.arcsin(p1[2]/r),np.arcsin(p2[2]/r)]
        theta2limits = np.array([np.arctan2(p1[2],np.sqrt(p1[0]**2+p1[1]**2)-R),np.arctan2(p2[2],np.sqrt(p2[0]**2+p2[1]**2)-R)]).reshape(-1,)
        #theta1limits = [np.arctan2(p1[1],p1[0]),np.arctan2(p2[1],p2[0])]
        p1 /= R+r*np.cos(theta2limits[0])
        p2 /= R+r*np.cos(theta2limits[1])
        theta1limits = np.array([np.arctan2(p1[1],p1[0]),np.arctan2(p2[1],p2[0])]).reshape(-1,)
        if theta2limits[0]<0 and theta2limits[1]>0: theta2limits[0]+=2*np.pi
        if theta1limits[1]<0 and theta1limits[0]>0: theta1limits[1]+=2*np.pi
        if element_type=='Standard': elements.append(Lagrange.TorusSegment_Standard(R,r,theta1limits,theta2limits))
        else: elements.append(Lagrange.TorusSegment_Serendipity(R,r,theta1limits,theta2limits))
    
    mesh = Lagrange.Mesh([Lagrange.Domain(elements)])

    mesh.dList[0].edges = 32*(1+xsec_refinements)*(1+circum_refinements)
    mesh.dList[0].corners = 16*(1+xsec_refinements)*(1+circum_refinements)
    mesh.dList[0].extraordinary_points = 0
        
    return mesh
        
        
    
    
    






    

if  __name__ == "__main__": 
    
    #mesh = Sphere_CubeToStandardLagrangeQuad(3)
    #mesh = Sphere_CubeToSerendipityLagrangeQuad(3)
    #mesh = Sphere_ExactSerendipityLagrangeQuad()
    mesh = Sphere_Bezier()
    
    from mayavi import mlab
    mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0))
    N=50
    for e in mesh.eList:
        xi1,xi2 = np.mgrid[e.limits[0]:e.limits[1]:N*1j,e.limits[2]:e.limits[3]:N*1j]
        xi1=xi1.reshape(-1,); xi2=xi2.reshape(-1,)
        x,y,z = e.vals(xi1,xi2)
        x=x.reshape(N,N); y=y.reshape(N,N); z=z.reshape(N,N);
        mlab.mesh(x,y,z,color=(0,0.5,1),opacity=0.75)
    for e in mesh.eList:
        xi1s=[e.limits[0],e.limits[1],e.limits[1],e.limits[1],e.limits[1],e.limits[0],e.limits[0],e.limits[0]]
        xi2s=[e.limits[3],e.limits[3],e.limits[3],e.limits[2],e.limits[2],e.limits[2],e.limits[2],e.limits[3]]
        for i in xrange(4):
            xi1 = np.linspace(xi1s[2*i],xi1s[2*i+1],N)
            xi2 = np.linspace(xi2s[2*i],xi2s[2*i+1],N)
            x,y,z=e.vals(xi1,xi2)
            mlab.plot3d(x,y,z,color=(1,1,1),tube_radius=0.01)