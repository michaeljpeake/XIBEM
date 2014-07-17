# Michael Peake
# Durham University

import numpy as np
import LagrangianElements as Lagrange
import BezierElements as Bezier




#############################################################
###                   SINGLE CYLINDER                     ###
#############################################################

def Cylinder_Quadratic(num_elements,radius=1.0):
    theta = np.linspace(2*np.pi,0,num=2*num_elements+1)
    x,y=np.vstack([radius*np.cos(theta),radius*np.sin(theta)])
    elements = [Lagrange.QuadraticElement(np.vstack([x[2*i:2*i+3],y[2*i:2*i+3]])) for i in xrange(num_elements)]
    cylinder = Lagrange.Domain(elements)
    return Lagrange.Mesh([cylinder])
    
def Cylinder_Exact(num_elements,radius=1.0):
    theta = np.linspace(2*np.pi,0,num_elements+1)
    elements = [Lagrange.CircularArc(0,0,theta[i],theta[i+1],radius) for i in xrange(num_elements)]
    cylinder = Lagrange.Domain(elements)
    return Lagrange.Mesh([cylinder])
    
def Cylinder_Bezier(elements_per_quarter_circle=1,r=1.0):

    P = np.array(([r,0],[r,-r],[0,-r],[-r,-r],[-r,0],[-r,r],[0,r],[r,r],[r,0]),np.float)
    degree = 2
    KnotVector = np.array([0,0,0,0.25,0.25,0.5,0.5,0.75,0.75,1,1,1])
    a = 1/np.sqrt(2)
    w = np.array([1,a,1,a,1,a,1,a,1])
    Pw = np.array([P[:,0]*w,P[:,1]*w,w]).T
    cylinder = Bezier.NURBSCurve(degree,KnotVector,Pw)
        
    if elements_per_quarter_circle>1:
        uniqueknots = np.unique(cylinder.U)
        for i in xrange(uniqueknots.size-1):
            XI = np.linspace(uniqueknots[i],uniqueknots[i+1],elements_per_quarter_circle+1)[1:-1]
            cylinder.refineknotvector(XI)
            
    nb,Qw = cylinder.decompose_into_Bezier_segments()
    elements = [Bezier.BezierElement(Qw[i].T) for i in xrange(nb)]
    cylinder = Bezier.Domain(elements)
    return Bezier.Mesh([cylinder])




#############################################################
###                      CAPSULE                          ###
#############################################################

# This is not the mesh used in my research. Instead, it is geometry
# designed by Ganesh Diwan that ensure all elements are the same length.

def Capsule_Exact(radius=1.0,straight_length_factor=1,E_per_segment=1):
    """Length of straight is length of arc multiplied by straight_length_factor
    Number of elements in arc is E_per_segment
    Number of elements in straight is E_per_segment*straight_length_factor
    NOTE: straight_length_factor must be integer > 0
    """
    temp = 0.5*radius*np.pi*straight_length_factor
    temp = np.linspace(-temp,temp,E_per_segment*straight_length_factor+1)
    lines1 = [Lagrange.Line([temp[i],radius],[temp[i+1],radius]) for i in xrange(E_per_segment*straight_length_factor)]    
    temp = np.linspace(0.5*np.pi,-0.5*np.pi,E_per_segment+1)
    arcs1 = [Lagrange.CircularArc(0.5*radius*np.pi*straight_length_factor,0,temp[i],temp[i+1],1) for i in xrange(E_per_segment)]    
    temp = 0.5*radius*np.pi*straight_length_factor
    temp = np.linspace(temp,-temp,E_per_segment*straight_length_factor+1)
    lines2 = [Lagrange.Line([temp[i],-radius],[temp[i+1],-radius]) for i in xrange(E_per_segment*straight_length_factor)]    
    temp = np.linspace(1.5*np.pi,0.5*np.pi,E_per_segment+1)
    arcs2 = [Lagrange.CircularArc(-0.5*radius*np.pi*straight_length_factor,0,temp[i],temp[i+1],1) for i in xrange(E_per_segment)]    
    elements = lines1,arcs1,lines2,arcs2
    elements = [element for section in elements for element in section]
    return Lagrange.Mesh([Lagrange.Domain(elements)])

def Capsule_Quadratic(radius=1.0,straight_length_factor=1,E_per_segment=1):
    """Length of straight is length of arc multiplied by straight_length_factor
    Number of elements in arc is E_per_segment
    Number of elements in straight is E_per_segment*straight_length_factor
    NOTE: straight_length_factor must be integer > 0
    """
    capsule = Capsule_Exact(radius,straight_length_factor,E_per_segment)
    return Lagrange.Mesh([Lagrange.Domain([Lagrange.QuadraticElement(e.P) for e in capsule.eList])])
    
def Capsule_Bezier(radius=1.0,split_segments_into=1):
    """This is default coded for two elements per segment and a straight_length_factor of 1"""
    U = np.array([0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,8],dtype=np.float)
    b=0.5*np.pi
    P = np.array(([-b,1],[-0.5*b,1],[0,1],[0.5*b,1],[b,1],[b+1,1],[b+1,0],[b+1,-1],[b,-1],[0.5*b,-1],[0,-1],[-0.5*b,-1],[-b,-1],[-b-1,-1],[-b-1,0],[-b-1,1],[-b,1]),dtype=np.float)
    w = np.array([1,1,1,1,1,1/np.sqrt(2),1,1/np.sqrt(2),1,1,1,1,1,1/np.sqrt(2),1,1/np.sqrt(2),1],dtype=np.float)
    Pw = np.array([P[:,0]*w,P[:,1]*w,w]).T
    capsule = Bezier.NURBSCurve(2,U,Pw)  
    
    if split_segments_into>1:
        uniqueknots = np.unique(capsule.U)
        for i in xrange(uniqueknots.size-1):
            XI = np.linspace(uniqueknots[i],uniqueknots[i+1],split_segments_into+1)[1:-1]
            capsule.refineknotvector(XI)
                   
    nb,Qw = capsule.decompose_into_Bezier_segments()
    return Bezier.Mesh([Bezier.Domain([Bezier.BezierElement(Qw[i].T) for i in xrange(nb)])])



#############################################################
###                   FIVE CYLINDERS                      ###
#############################################################

# A set of five cylinders of radius a equally spaced on the edge
# of an imaginary circle of radius r

def FiveCylinders_Exact(num_elements,a=1,r=3):
    theta = np.linspace(0,2*np.pi,5,endpoint=False)
    cx = r*np.cos(theta)
    cy = r*np.sin(theta)    
    theta = np.linspace(2*np.pi,0,num_elements+1)
    cylinders = [Lagrange.Domain([Lagrange.CircularArc(cx[j],cy[j],theta[i],theta[i+1],a) 
                        for i in xrange(num_elements)]) for j in xrange(5)]
    return Lagrange.Mesh(cylinders)
    
def FiveCylinders_Quadratic(num_elements,a=1.0,r=3.0): 
    mesh = FiveCylinders_Exact(num_elements,a,r)
    return Lagrange.Mesh([Lagrange.Domain([Lagrange.QuadraticElement(e.P) for e in d.eList]) for d in mesh.dList])

def FiveCylinders_Bezier(elements_per_quarter,a=1,r=3):
    def cylinder(elements_per_quarter,xp,yp,r=1.0):
        P = np.array(([r,0],[r,-r],[0,-r],[-r,-r],[-r,0],[-r,r],[0,r],[r,r],[r,0]),np.float)
        P[:,0]+=xp
        P[:,1]+=yp
        p = 2
        U = np.array([0,0,0,0.25,0.25,0.5,0.5,0.75,0.75,1,1,1])
        s = 1/np.sqrt(2)
        w = np.array([1,s,1,s,1,s,1,s,1])
        Pw = np.array([P[:,0]*w,P[:,1]*w,w]).T
        cylinder = Bezier.NURBSCurve(p,U,Pw)
        if elements_per_quarter>1:
            uniqueknots = np.unique(cylinder.U)
            for i in xrange(uniqueknots.size-1):
                XI = np.linspace(uniqueknots[i],uniqueknots[i+1],elements_per_quarter+1)[1:-1]
                cylinder.refineknotvector(XI)       
        nb,Qw = cylinder.decompose_into_Bezier_segments()
        elements = [Bezier.BezierElement(Qw[i].T) for i in xrange(nb)]
        return Bezier.Domain(elements)
    theta = np.linspace(0,2*np.pi,5,endpoint=False)
    xp = r*np.cos(theta)
    yp = r*np.sin(theta)    
    return Bezier.Mesh([cylinder(elements_per_quarter,xp[i],yp[i],a) for i in xrange(5)])
    
    
    

#############################################################
###               CAPSULES AND CYLINDERS                  ###
#############################################################

# I have misplaced the code for the NURBS / Bezier version of
# this geometry. It is easy to reproduce however.

def CapsAndCyl_Exact(elements_per_semicircular_arc=1,elements_per_line=1):
    # Cylinder
    cylE = 2*elements_per_semicircular_arc
    theta = np.linspace(2*np.pi,0,cylE+1)
    cylinder = Lagrange.Domain([Lagrange.CircularArc(1,0,theta[i],theta[i+1],1) for i in xrange(cylE)])
    
    # Capsule 1
    arcE = elements_per_semicircular_arc
    linE = elements_per_line

    x = np.linspace(-np.sqrt(2)-1,-1,linE+1)
    y = np.linspace(2,np.sqrt(2)+2,linE+1)
    lines1=[Lagrange.Line([x[i],y[i]],[x[i+1],y[i+1]]) for i in xrange(linE)]
    
    theta=np.linspace(0.75*np.pi,-0.25*np.pi,arcE+1)
    arcs1 = [Lagrange.CircularArc(1/np.sqrt(2)-1,1/np.sqrt(2)+2,theta[i],theta[i+1],1) for i in xrange(arcE)]
    
    x = np.linspace(np.sqrt(2)-1,-1,linE+1)
    y = np.linspace(2,-np.sqrt(2)+2,linE+1)
    lines2=[Lagrange.Line([x[i],y[i]],[x[i+1],y[i+1]]) for i in xrange(linE)]
    
    theta=np.linspace(1.75*np.pi,0.75*np.pi,arcE+1)
    arcs2 = [Lagrange.CircularArc(-1/np.sqrt(2)-1,-1/np.sqrt(2)+2,theta[i],theta[i+1],1) for i in xrange(arcE)]
    
    elements = lines1,arcs1,lines2,arcs2
    capsule1 = Lagrange.Domain([element for section in elements for element in section])
    
    # Capsule 2
    
    x = np.linspace(-1,np.sqrt(2)-1,linE+1)
    y = np.linspace(np.sqrt(2)-2,-2,linE+1)
    lines1=[Lagrange.Line([x[i],y[i]],[x[i+1],y[i+1]]) for i in xrange(linE)]
    
    theta=np.linspace(0.25*np.pi,-0.75*np.pi,arcE+1)
    arcs1 = [Lagrange.CircularArc(1/np.sqrt(2)-1,-1/np.sqrt(2)-2,theta[i],theta[i+1],1) for i in xrange(arcE)]
    
    x = np.linspace(-1,-np.sqrt(2)-1,linE+1)
    y = np.linspace(-np.sqrt(2)-2,-2,linE+1)
    lines2=[Lagrange.Line([x[i],y[i]],[x[i+1],y[i+1]]) for i in xrange(linE)]
    
    theta=np.linspace(1.25*np.pi,0.25*np.pi,arcE+1)
    arcs2 = [Lagrange.CircularArc(-1/np.sqrt(2)-1,1/np.sqrt(2)-2,theta[i],theta[i+1],1) for i in xrange(arcE)]
    
    elements = lines1,arcs1,lines2,arcs2
    capsule2 = Lagrange.Domain([element for section in elements for element in section])
    
    return Lagrange.Mesh([cylinder,capsule1,capsule2])

def CapsAndCyl_Quadratic(elements_per_semicircular_arc=1,elements_per_line=1):
    mesh = CapsAndCyl_Exact(elements_per_semicircular_arc,elements_per_line)
    return Lagrange.Mesh([Lagrange.Domain([Lagrange.QuadraticElement(e.P) for e in d.eList]) for d in mesh.dList])

    
    


if __name__ is "__main__":
    
    mesh = Capsule_Exact(straight_length_factor=2)
    
    import matplotlib.pyplot as plt
    plt.figure()
    ax1 = plt.axes(aspect='equal')
    for e in mesh.eList:
        p=e.vals(np.linspace(e.limits[0],e.limits[1],1000))
        plt.plot(p[0],p[1],'-k',linewidth=2)
    plt.show()