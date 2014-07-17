# Michael Peake
# Durham University

import numpy as np
from scipy.special import jvp,hankel1,h1vp
from matplotlib.pyplot import figure,show
from matplotlib.path import Path



def getError(exact,simulation):
    diff = exact-simulation
    return np.sqrt(np.sum(np.abs(diff)**2) / np.sum(np.abs(exact)**2))

def cylinder(wavenumber,xPlotPoints,yPlotPoints,radius=1.0):
    """ Analytical potential on hard cylinder with [1,0] incident wave (Jones)"""
    x = xPlotPoints
    y = yPlotPoints
    theta = np.arctan2(y,x)
    numPoints = len(x)

    nans=False,False
    N = 100
    
    while any(nans)==False:
        
        N += 50
        n = np.arange(0,N)
        neumann = 2*np.ones(N); neumann[0]=1

        dJ = jvp(n,wavenumber*radius,1)
        H1 = hankel1(n,wavenumber*radius)
        dH1 = h1vp(n,wavenumber*radius,1)
    
        nans = np.isnan(dJ) + np.isnan(H1) + np.isnan(dH1)
    
    dJ = dJ.compress(np.logical_not(nans))
    H1 = H1.compress(np.logical_not(nans))
    dH1 = dH1.compress(np.logical_not(nans))
    neumann = neumann.compress(np.logical_not(nans))
    n = n.compress(np.logical_not(nans))

    total = (-neumann * (1j**n) * dJ * H1 / dH1).reshape(-1,1)

    cosines = np.cos(n.reshape(-1,1)*theta)

    total = total * cosines

    incPotential = np.exp(1j*wavenumber*x).reshape(numPoints)
    fullPotential = np.sum(total,axis=0) + incPotential
    return fullPotential



##############################################################################
####                                                                      ####
####                  METHOD OF FUNDAMENTAL SOLUTIONS                     ####
####                                                                      ####
##############################################################################

def evaluate_dphidn(mesh,k,incAmp,incDir):
    phi = incAmp * np.exp(1j*k*np.dot(mesh.sampVals.T,incDir))
    return (1j*k*np.dot(mesh.sampNormals.T,incDir)*phi).reshape(-1,)

def evaluate_T(mesh,k):
    px = mesh.sourceVals[0].reshape(-1,1)
    py = mesh.sourceVals[1].reshape(-1,1)
    qx = mesh.sampVals[0].reshape(1,-1)
    qy = mesh.sampVals[1].reshape(1,-1)
    r = np.sqrt( (qx-px)**2 + (qy-py)**2 )
    drdn = ( (qx-px)*mesh.sampNormals[0] + (qy-py)*mesh.sampNormals[1] ) / r
    return 0.25j * k * drdn * hankel1(1,k*r)

def get_potentials(xy,mesh,k,incAmp,incDir):
    incident = incAmp * np.exp( 1j*k*np.dot(xy.T,incDir) )
    x=xy[0].reshape(-1,1)
    y=xy[1].reshape(-1,1)
    sourcex=mesh.sourceVals[0].reshape(1,-1)
    sourcey=mesh.sourceVals[1].reshape(1,-1)
    r = np.sqrt( (x-sourcex)**2 + (y-sourcey)**2 )
    scattered = np.sum(0.25j * mesh.amplitudes * hankel1(0,k*r),axis=1)
    return scattered + incident.reshape(-1,)

def MFS(mesh,plotx,ploty,k,incDir,incAmp=1.0,tau=10,frac_samp=2,offset=0.15,numSource=None):
    """MFS(mesh,plotx,ploty,k,incDir,incAmp=1.0,tau=10,frac_samp=2,offset=0.15)
    
    Evaluates a solution using the Method of Fundamental Solutions
    Input:
        mesh      : mesh of the problem
        plotx     : x-coordinates of points at which to evaluate solution
        ploty     : y-coordinates of points at which to evaluate solution
        k         : wavenumber of incident wave
        incDir    : direction of propagation in form [x,y]
        incAmp    : incident wave amplitude
        tau       : number of source points per wavelength in the boundary
        frac_samp : fraction of sampling points to source points ( >= 1 )
        offset    : how much to move source points away from boundary
        
    Output:
        Wave potentials at the coordinates [plotx,ploty]
        """
    mesh.MFS=True
    
    for d in mesh.dList:
        if numSource:
            d.numSource = numSource
        else:
            d.numSource = int(np.ceil(tau*k*d.length()/(2*np.pi)))
        d.numSamp = int(frac_samp*d.numSource)
        source = np.linspace(0,d.length(),d.numSource,endpoint=False)
        samp = np.linspace(0,d.length(),d.numSamp,endpoint=False)
        low=high=0
        for e in d.eList:
            high+=e.length()
            xi = np.where((low<=source)==(source<high))
            xi = e.limits[0] + (e.limits[1]-e.limits[0])*(source[xi]-low)/e.length()
            e.sourceNormals = e.normals(xi)
            e.sourceVals = e.vals(xi) + offset*e.sourceNormals
            xi = np.where((low<=samp) == (samp<high))
            xi = e.limits[0] + (e.limits[1]-e.limits[0])*(samp[xi]-low)/e.length()
            e.sampVals = e.vals(xi)
            e.sampNormals = e.normals(xi)
            low+=e.length()
        d.sourceVals = np.hstack([e.sourceVals for e in d.eList])
        d.sourceNormals = np.hstack([e.sourceNormals for e in d.eList])
        d.sampVals = np.hstack([e.sampVals for e in d.eList])
        d.sampNormals = np.hstack([e.sampNormals for e in d.eList])
    mesh.sourceVals = np.hstack([d.sourceVals for d in mesh.dList])
    mesh.sourceNormals = np.hstack([d.sourceNormals for d in mesh.dList])
    mesh.sampVals = np.hstack([d.sampVals for d in mesh.dList])
    mesh.sampNormals = np.hstack([d.sampNormals for d in mesh.dList])

    dphidn = evaluate_dphidn(mesh,k,incAmp,incDir) # derivative phi_inc wrt n

    T = evaluate_T(mesh,k)

    A = np.dot(T,T.T)

    b = np.sum(T*dphidn,axis=1)

    mesh.amplitudes = np.linalg.solve(A,b)
    
    return get_potentials(np.vstack([plotx,ploty]),mesh,k,incAmp,incDir)


def Alternative_MFS(mesh,plotx,ploty,k,incDir,incAmp=1.0,tau=10,frac_samp=2,offset=0.15):

    mesh.MFS=True
    
    for d in mesh.dList:
        d.numSource = int(np.ceil(tau*k*d.length()/(2*np.pi)))
        d.numSamp = int(frac_samp*d.numSource)
        source = np.linspace(0,d.length(),d.numSource,endpoint=False)
        samp = np.linspace(0,d.length(),d.numSamp,endpoint=False)
        low=high=0
        for e in d.eList:
            high+=e.length()
            xi = np.where((low<=source)==(source<high))
            xi = e.limits[0] + (e.limits[1]-e.limits[0])*(source[xi]-low)/e.length()
            e.sourceNormals = e.normals(xi)
            e.sourceVals = e.vals(xi) + offset*e.sourceNormals
            xi = np.where((low<=samp) == (samp<high))
            xi = e.limits[0] + (e.limits[1]-e.limits[0])*(samp[xi]-low)/e.length()
            e.sampVals = e.vals(xi)
            e.sampNormals = e.normals(xi)
            low+=e.length()
        d.sourceVals = np.hstack([e.sourceVals for e in d.eList])
        d.sourceNormals = np.hstack([e.sourceNormals for e in d.eList])
        d.sampVals = np.hstack([e.sampVals for e in d.eList])
        d.sampNormals = np.hstack([e.sampNormals for e in d.eList])
    mesh.sourceVals = np.hstack([d.sourceVals for d in mesh.dList])
    mesh.sourceNormals = np.hstack([d.sourceNormals for d in mesh.dList])
    mesh.sampVals = np.hstack([d.sampVals for d in mesh.dList])
    mesh.sampNormals = np.hstack([d.sampNormals for d in mesh.dList])    
    
    # A 
    qx = mesh.sourceVals[0].reshape(1,-1)
    qy = mesh.sourceVals[1].reshape(1,-1)
    px = mesh.sampVals[0].reshape(-1,1)
    py = mesh.sampVals[1].reshape(-1,1)
    npx = mesh.sampNormals[0].reshape(-1,1)
    npy = mesh.sampNormals[1].reshape(-1,1)
    r = np.sqrt( (qx-px)**2 + (qy-py)**2 )
    drdn = ( (qx-px)*npx + (qy-py)*npy ) / r
    A = 0.25j * k * drdn * hankel1(1,k*r)    
    
    # b vector
    phi = incAmp * np.exp(1j*k*np.dot(mesh.sampVals.T,incDir))
    b = -(1j*k*np.dot(mesh.sampNormals.T,incDir)*phi).reshape(-1,1)
    
    mesh.amplitudes,residuals,rank,s = np.linalg.lstsq(A,b,rcond=1e-10)
    mesh.amplitudes = mesh.amplitudes.reshape(-1,)
    
    return get_potentials(np.vstack([plotx,ploty]),mesh,k,incAmp,incDir)



def converged_MFS(mesh,plotx,ploty,k,incDir,incAmp=1.0,mintau=1,maxtau=20,steps=38):
    """converged_(mesh,plotx,ploty,k,incDir,incAmp=1.0,mintau=1,maxtau=20,steps=38)
    
    Finds a converged solution and evaluates using the Method of Fundamental Solutions
    Input:
        mesh   : mesh of the problem
        plotx  : x-coordinates of points at which to evaluate solution
        ploty  : y-coordinates of points at which to evaluate solution
        k      : wavenumber of incident wave
        incDir : direction of propagation in form [x,y]
        incAmp : incident wave amplitude
        mintau : minimum value of tau to try
        maxtau : maximum value of tau to try
        steps  : how many steps between mintau and maxtau
        
    Output:
        Wave potentials at the coordinates [plotx,ploty]
        Will also print the value of tau used for final solution
        """
        
    testx=plotx[0]
    testy=ploty[0]
    taus=np.linspace(mintau,maxtau,steps+1)
    vals=[MFS(mesh,testx,testy,k,incDir,incAmp,tau) for tau in taus]
    vals = np.array([np.abs(vals[i]-vals[i+1]) for i in xrange(steps)])
    tau = taus[ np.where(vals==np.min(vals))[0][0] +1 ]
    print "MFS solution settled at tau:",tau
    return MFS(mesh,plotx,ploty,k,incDir,incAmp,tau)
    


def PlotScattering(mesh,k,incAmp,incDir,xmin,xmax,xres,ymin,ymax,yres,grey=False,return_vals=False):
    try: mesh.MFS
    except: 
        print "Must run MFS solution before plotting"
        return
    data = np.zeros((yres,xres),np.complex)
    X,Y = np.mgrid[xmin:xmax:xres*1j,ymin:ymax:yres*1j]
    X=X.T
    Y=Y.T
    print "Starting plot"
    frac=10
    comp_old=10
    for row in xrange(yres):
        if 100*(row+1)/yres > comp_old:
            string = "%d percent complete" % comp_old
            print string
            comp_old += frac
        XY = np.vstack([X[row],Y[row]])
        data[row] = get_potentials(XY,mesh,k,incAmp,incDir)
    print "100 percent complete"
    # Remove data from within scatterer
    FindBoundary(mesh)
    X=X.reshape(-1,)
    Y=Y.reshape(-1,)
    XY=np.vstack([X,Y]).T
    for d in mesh.dList:
        inside = d.boundary.contains_points(XY)
        inside = inside.reshape(yres,xres)
        data[inside] = None
    fig=figure(figsize=(10,10),dpi=40)
    ax=fig.add_axes([0,0,1,1])
    if grey:
        ax.imshow(np.abs(data),origin='lower',extent=[xmin,xmax,ymin,ymax],cmap='gray')
    else:
        ax.imshow(np.abs(data),origin='lower',extent=[xmin,xmax,ymin,ymax])
    for d in mesh.dList:
        #ax.plot(d.boundary.vertices[:,0],d.boundary.vertices[:,1],'-k',linewidth=4)
        for e in d.eList:
            p=e.vals(np.linspace(e.limits[0],e.limits[1],1000))
            ax.plot(p[0],p[1],'-k',linewidth=4)
    ax.axis([xmin,xmax,ymin,ymax])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    show()
    if return_vals:
        return data
        
        
def FindBoundary(mesh,N=1000):
    
    for d in mesh.dList:
        px=[]
        py=[]
        s = np.linspace(0,d.length(),N,endpoint=False)
        slow=0
        shigh=0
        for e in d.eList:
            shigh += e.length()
            spoints = np.where((slow<=s) == (s<shigh))
            xi = e.limits[0] + (e.limits[1]-e.limits[0])*(s[spoints]-slow)/e.length()
            xy = e.vals(xi)
            slow += e.length()
            px = np.append(px,xy[0])
            py = np.append(py,xy[1])
        d.boundary = Path(np.vstack([px,py]).T)
    
    



##############################################################################
####                                                                      ####
####                       LINTON AND EVANS SOLUTION                      ####
####                                                                      ####
##############################################################################

def linton_evans_coeffs(k,incAng,numCylinders,centres,radii,M):
    # Four-cylinder problem
    #if k<=20:
    #    M=40
    #elif 20<k<=40:
    #    M=80
    #elif 40<k<=60:
    #    M=120
    #elif 60<k<=120:
    #    M=180
    #elif k>120:
    #    M=300
        
    N = numCylinders
    origin = centres
    a = radii

    i = np.arange(0,N).reshape(-1,1)
    j = np.arange(0,N).reshape(1,-1)
    
    R = np.sqrt((origin[j,0]-origin[i,0])**2+(origin[j,1]-origin[i,1])**2)
    alpha = np.angle((origin[j,0]-origin[i,0]) +1j*(origin[j,1]-origin[i,1]))

    q = np.repeat(np.arange(N),2*M+1).reshape(-1,1) # rows
    p = np.repeat(np.arange(N),2*M+1).reshape(1,-1) # cols
    m = np.repeat([np.arange(-M,M+1)],N,axis=0).reshape(-1,1) # rows
    n = np.repeat([np.arange(-M,M+1)],N,axis=0).reshape(1,-1) # cols

    # Right-hand vector
    Iq = np.exp(1j*k*(origin[q,0]*np.cos(incAng)+origin[q,1]*np.sin(incAng)))
    exponential1 = np.exp(1j*m*(np.pi/2-incAng))
    RHS = - Iq * exponential1
    
    # Left-hand side matrix
    Znp = jvp(n,k*a[p]) / h1vp(n,k*a[p])
    exponential2 = np.exp(1j*(n-m)*alpha[p,q])
    Hnm = hankel1(n-m,k*R[p,q])
    LHS = Znp * exponential2 * Hnm

    Identity = np.identity(2*M+1)
    for cyl in range(N):
        LHS[cyl*(2*M+1):(cyl+1)*(2*M+1),cyl*(2*M+1):(cyl+1)*(2*M+1)] = Identity

    # Solve for values of Anq
    A=np.linalg.solve(LHS,RHS)

    return A

def linton_evans_potential(k,Q,aq,thetaq,Acoeffs,M):

    ## Four-cylinder problem
    #if k<=20:
    #    M=40
    #elif 20<k<=40:
    #    M=80
    #elif 40<k<=60:
    #    M=120
    #elif 60<k<=120:
    #    M=180
    #elif k>120:
    #    M=300

    # Find potential at (aq,thetaq)
    Anq = Acoeffs[Q*(2*M+1):(Q+1)*(2*M+1)]
    n2 = np.arange(-M,M+1).reshape(-1,1)

    summation = np.sum( Anq / h1vp(n2,k*aq) * np.exp(1j*n2*thetaq) , axis=0)

    phi = -2j / (np.pi*k*aq) * summation

    return phi







    

if  __name__ == "__main__":
    
    import ResearchMeshes

    radius=1.0
    wavenum=20
    wavenum=25
    incAng = 0.25*np.pi
    incDir = [np.cos(incAng),np.sin(incAng)]
    incDir = [0.5,np.sqrt(3)/2]
    incAmp=1.0
    frac_samp=1
    offset=0.05

    
    
    #mesh = ResearchMeshes.FiveCylinders_Exact(2,0.5,1.5)
    #mesh = ResearchMeshes.FiveCylinders_Exact(2)
    #mesh = ResearchMeshes.CapsAndCyl_Exact(1)
    #mesh = ResearchMeshes.Cylinder_Exact(2,r=1.0)
    mesh = ResearchMeshes.Ganesh_Capsule()
    #mesh = ResearchMeshes.CapsAndCyl_Exact(1)
    FindBoundary(mesh)
    px = np.vstack([d.boundary.vertices[:,0] for d in mesh.dList])
    py = np.vstack([d.boundary.vertices[:,1] for d in mesh.dList])
    
    simulation = MFS(mesh,px,py,wavenum,incDir,tau=3)
    phi=PlotScattering(mesh,wavenum,incAmp,incDir,-3,3,600,-3,3,600,return_vals=True)
    #phi=PlotScattering(mesh,wavenum,incAmp,incDir,-5,5,500,-5,5,500,return_vals=True)
    #import pickle
    #newfile=open("temp.pkl","wb")
    #pickle.dump([phi],newfile)
    #newfile.close()
    
    #mesh = ResearchMeshes.CapulePlus_Exact()
    #FindBoundary(mesh)
    #px = np.hstack([d.boundary.vertices[:,0] for d in mesh.dList])
    #py = np.hstack([d.boundary.vertices[:,1] for d in mesh.dList])
    #simulation = MFS(mesh,px,py,wavenum,incDir,tau=3)
    #PlotScattering(mesh,wavenum,incAmp,incDir,-5,5,150,-5,5,150)
    
    #mesh = ResearchMeshes.Pluses_Exact(a=0.25)
    #FindBoundary(mesh)
    #px = np.hstack([d.boundary.vertices[:,0] for d in mesh.dList])
    #py = np.hstack([d.boundary.vertices[:,1] for d in mesh.dList])
    #simulation = MFS(mesh,px,py,wavenum,incDir,tau=3)
    #data=PlotScattering(mesh,wavenum,incAmp,incDir,-3,3,500,-3,3,500,return_vals=1)
    
    
    
    
    
    ## Linton Evans
    #mesh = ResearchMeshes.FourCylinders(2,radius)
    #theta=np.linspace(0,2*np.pi,1000)
    #plotx=np.hstack([radius*np.cos(theta)-2,radius*np.cos(theta)+2,radius*np.cos(theta)+2,radius*np.cos(theta)-2])
    #ploty=np.hstack([radius*np.sin(theta)-2,radius*np.sin(theta)-2,radius*np.sin(theta)+2,radius*np.sin(theta)+2])
    #simulation = MFS(mesh,plotx,ploty,wavenum,incDir,numSource=150,incAmp=incAmp,frac_samp=frac_samp,offset=offset)
    #centres = np.array([[-2,-2],[2,-2],[2,2],[-2,2]],np.float)
    #radii = np.array([1,1,1,1],np.float)
    #LE_coeffs = linton_evans_coeffs(wavenum,angle,4,centres,radii)
    #LE_sol = np.hstack([linton_evans_potential(wavenum,i,1.0,theta,LE_coeffs) for i in xrange(4)])
    

    
