import numpy as np
from scipy.special import sph_jn,sph_yn
from warnings import simplefilter

def Pn(n,x):
    
    if n==0:
        return np.ones(x.size,1)
    else:
        P = np.zeros((x.size,n+2))
        P[:,1]=1
        P[:,2]=x
        for i in xrange(1,n):
            P[:,i+2]=((2.0*i+1.0)/(i+1.0))*x*P[:,i+1]-i/(i+1.0)*P[:,i]

        return P[:,1:]


def SpherePlaneWave(k,x):
    """Find acoustic potential from planewave [1,0,0] scattered
    by sphere of radius 1.
    
    Potentials are solved for a points
    p = r*cos(x) and so r must be the same for all points."""
    ka=k
    kr=k
    
    x=np.asarray(x,np.float).reshape(-1,)
    theta = np.arccos(x)

    N=0
    badcells=False,False
    while any(badcells)==False:
        
        N += 100
        n = np.arange(N+1)

        djnka = sph_jn(N,ka)[1]
        dynka = sph_yn(N,ka)[1]
        dhnka = djnka + 1j*dynka
        
        jnkr = sph_jn(N,kr)[0]
        ynkr = sph_yn(N,kr)[0]
        hnkr = jnkr + 1j*ynkr

        simplefilter("ignore")
        pscat= - (1j**n) * (2*n+1) * djnka * hnkr / dhnka
        simplefilter("default")

        badcells = np.isnan(pscat)+np.isinf(pscat)
        

    
    pscat = np.repeat([pscat],x.size,axis=0) * Pn(N,np.cos(theta)) 
    
    pscat = pscat.compress(np.logical_not(badcells),axis=1)

    pinc = np.exp(1j*k*x)

    return np.sum(pscat,axis=1) + pinc
    
    
    
def SpherePlaneWave2(k,a,r,theta,just_scattered=False):
    """Find acoustic potential from planewave [1,0,0] scattered
    by sphere of radius 1."""
    ka=k*a
    x = r*np.cos(theta)
    
    N=0
    badcells=False,False
    while np.any(badcells)==False:
        
        N += 100
        n = np.arange(N+1)

        djnka = sph_jn(N,ka)[1]
        dynka = sph_yn(N,ka)[1]
        dhnka = djnka + 1j*dynka
        
        djnka = np.repeat([djnka],r.size,axis=0).reshape(r.size,N+1)
        dhnka = np.repeat([dhnka],r.size,axis=0).reshape(r.size,N+1)
        
        jnkr = np.vstack([sph_jn(N,kr)[0] for kr in k*r])
        ynkr = np.vstack([sph_yn(N,kr)[0] for kr in k*r])
        hnkr = jnkr + 1j*ynkr

        simplefilter("ignore")
        pscat= - (1j**n) * (2*n+1) * djnka * hnkr / dhnka
        simplefilter("default")

        badcells = np.isnan(pscat)+np.isinf(pscat)
    
    pscat *= Pn(N,np.cos(theta)) 
    
    pscat = pscat.compress(np.all(np.logical_not(badcells)==True,axis=0),axis=1)
    
    if just_scattered: return np.sum(pscat,axis=1)
    else: return np.sum(pscat,axis=1) + np.exp(1j*k*x)
    


def getError(exact,simulation):
    exact = np.array(exact).reshape(-1,)
    simulation = np.array(simulation).reshape(-1,)
    diff = exact-simulation
    return np.sqrt(np.sum(np.abs(diff)**2) / np.sum(np.abs(exact)**2))



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
    pz = mesh.sourceVals[2].reshape(-1,1)
    qx = mesh.sampVals[0].reshape(1,-1)
    qy = mesh.sampVals[1].reshape(1,-1)
    qz = mesh.sampVals[2].reshape(1,-1)
    nx = mesh.sampNormals[0]
    ny = mesh.sampNormals[1]
    nz = mesh.sampNormals[2]
    r = np.sqrt( (qx-px)**2 + (qy-py)**2 + (qz-pz)**2 )
    drdn = ( (qx-px)*nx + (qy-py)*ny + (qz-pz)*nz ) / r
    return np.exp(1j*k*r)*(1j*k*r-1)*drdn/(4*np.pi*r**2)
    #return 0.25j * k * drdn * hankel1(1,k*r)

def get_potentials(xyz,mesh,k,incAmp,incDir):
    incident = incAmp * np.exp( 1j*k*np.dot(xyz.T,incDir) )
    x=xyz[0].reshape(-1,1)
    y=xyz[1].reshape(-1,1)
    z=xyz[2].reshape(-1,1)
    sourcex=mesh.sourceVals[0].reshape(1,-1)
    sourcey=mesh.sourceVals[1].reshape(1,-1)
    sourcez=mesh.sourceVals[2].reshape(1,-1)
    r = np.sqrt( (x-sourcex)**2 + (y-sourcey)**2 + (z-sourcez)**2 )
    scattered = np.sum(mesh.amplitudes*np.exp(1j*k*r)/(4*np.pi*r),axis=1)
#    scattered = np.sum(0.25j * mesh.amplitudes * hankel1(0,k*r),axis=1)
    return scattered + incident.reshape(-1,)    
    
def MFS(mesh,plotx,ploty,plotz,k,incDir,incAmp=1.0,tau=10,frac_samp=2,numSource=0,numSamp=0,offset=0.15):
    """MFS(mesh,plotx,ploty,plotz,k,incDir,incAmp=1.0,tau=10,frac_samp=2,offset=0.15)
    
    Evaluates a solution using the Method of Fundamental Solutions
    Input:
        mesh      : mesh of the problem
        plotx     : x-coordinates of points at which to evaluate solution
        ploty     : y-coordinates of points at which to evaluate solution
        plotz     : z-coordinates of points at which to evaluate solution
        k         : wavenumber of incident wave
        incDir    : direction of propagation in form [x,y]
        incAmp    : incident wave amplitude
        tau       : number of source points per wavelength in the boundary
        frac_samp : fraction of sampling points to source points ( >= 1 )
        offset    : how much to move source points away from boundary
        numSource : override for number of source points
        numSamp   : override for number of sample points
        
    Output:
        Wave potentials at the coordinates [plotx,ploty]
        """
    mesh.MFS=True
    
    if numSource == 0:
        for d in mesh.dList:
            d.numSource = int(np.ceil(tau**2*k**2*d.area()/(4*np.pi*np.pi)))
            #d.numSource = int(np.ceil(tau*k*d.length()/(2*np.pi)))
            #d.numSamp = int(frac_samp*d.numSource)
        
    def number_of_points(d,N):
            a = d.numelements * N**2
            b = d.edges * (N-2)
            c = d.corners * 3
            d = d.extraordinary_points
            return a-b-c+d    
    
    # Singular (source) points
    for d in mesh.dList:
        N=1
        if numSource == 0:
            while number_of_points(d,N) < d.numSource: N+=1
        else:
            while number_of_points(d,N) < numSource: N+=1
        d.numSource = number_of_points(d,N)
        xi1 = np.linspace(d.eList[0].limits[0],d.eList[0].limits[1],N)
        xi2 = np.linspace(d.eList[0].limits[2],d.eList[0].limits[3],N)        
        xi1,xi2=np.meshgrid(xi1,xi2)
        xi1=xi1.reshape(-1,) ; xi2=xi2.reshape(-1,)  
        souvals = d.eList[0].vals(d.eList[0].limits[0],d.eList[0].limits[2])
        sounorms = d.eList[0].normals(d.eList[0].limits[0],d.eList[0].limits[2])
        for e in d.eList:
            newvals = e.vals(xi1,xi2)
            newnorms = e.normals(xi1,xi2)         
            px,py,pz = newvals
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)
            qx,qy,qz=newvals
            rx=px-qx ; ry=py-qy ; rz=pz-qz
            rx=np.tril(rx, -1)[:,:-1]+np.triu(rx, 1)[:,1:]
            ry=np.tril(ry, -1)[:,:-1]+np.triu(ry, 1)[:,1:]
            rz=np.tril(rz, -1)[:,:-1]+np.triu(rz, 1)[:,1:]
            r = np.sqrt( rx**2 + ry**2 + rz**2 )
            delete = np.where(np.any(r<1e-10,axis=1))[0]
            newvals = np.delete(newvals,delete[1:],axis=1)
            newnorms = np.delete(newnorms,delete[1:],axis=1)
            px,py,pz = newvals
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)            
            qx,qy,qz=souvals
            r = np.sqrt( (qx-px)**2 + (qy-py)**2 + (qz-pz)**2 )
            delete = np.where(np.any(r<1e-12,axis=1))[0]
            souvals = np.hstack([souvals,np.delete(newvals,delete,axis=1)])
            sounorms = np.hstack([sounorms,np.delete(newnorms,delete,axis=1)])      
        d.sourceVals = souvals + offset*sounorms
        d.sourceNormals = sounorms
    mesh.sourceVals = np.hstack([d.sourceVals for d in mesh.dList])
    mesh.sourceNormals = np.hstack([d.sourceNormals for d in mesh.dList])
    
    # Sampling points    
    for d in mesh.dList:
        N=1
        if numSamp == 0:
            while number_of_points(d,N) < frac_samp*d.numSource: N+=1
        else:
            while number_of_points(d,N) < numSource: N+=1
        d.numSamp = number_of_points(d,N)
        xi1 = np.linspace(d.eList[0].limits[0],d.eList[0].limits[1],N)
        xi2 = np.linspace(d.eList[0].limits[2],d.eList[0].limits[3],N)        
        xi1,xi2=np.meshgrid(xi1,xi2)
        xi1=xi1.reshape(-1,) ; xi2=xi2.reshape(-1,)      
        sampvals = d.eList[0].vals(d.eList[0].limits[0],d.eList[0].limits[0])
        sampnorms = d.eList[0].normals(d.eList[0].limits[0],d.eList[0].limits[0])
        for e in d.eList:
            newvals = e.vals(xi1,xi2)
            newnorms = e.normals(xi1,xi2)         
            px,py,pz = newvals
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)
            qx,qy,qz=newvals
            rx=px-qx ; ry=py-qy ; rz=pz-qz
            rx=np.tril(rx, -1)[:,:-1]+np.triu(rx, 1)[:,1:]
            ry=np.tril(ry, -1)[:,:-1]+np.triu(ry, 1)[:,1:]
            rz=np.tril(rz, -1)[:,:-1]+np.triu(rz, 1)[:,1:]
            r = np.sqrt( rx**2 + ry**2 + rz**2 )
            delete = np.where(np.any(r<1e-10,axis=1))[0]
            newvals = np.delete(newvals,delete[1:],axis=1)
            newnorms = np.delete(newnorms,delete[1:],axis=1)
            px,py,pz = newvals
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)            
            qx,qy,qz=sampvals
            r = np.sqrt( (qx-px)**2 + (qy-py)**2 + (qz-pz)**2 )
            delete = np.where(np.any(r<1e-12,axis=1))[0]    
            sampvals = np.hstack([sampvals,np.delete(newvals,delete,axis=1)])
            sampnorms = np.hstack([sampnorms,np.delete(newnorms,delete,axis=1)])      
        d.sampVals = sampvals
        d.sampNormals = sampnorms
    mesh.sampVals = np.hstack([d.sampVals for d in mesh.dList])
    mesh.sampNormals = np.hstack([d.sampNormals for d in mesh.dList]) 
        
    dphidn = evaluate_dphidn(mesh,k,incAmp,incDir) # derivative phi_inc wrt n
    
    T = evaluate_T(mesh,k)

    A = np.dot(T,T.T)
    b = np.sum(-T*dphidn,axis=1)

    # Solve for fundamental solution amplitudes
    mesh.amplitudes = np.linalg.solve(A,b)

    return get_potentials(np.vstack([plotx,ploty,plotz]),mesh,k,incAmp,incDir)


def Alt_MFS(mesh,k,incDir,incAmp=1.0,tau=10,frac_samp=2,numSource=0,numSamp=0,offset=0.15):
    mesh.MFS=True  
    if numSource == 0:
        for d in mesh.dList:
            d.numSource = int(np.ceil(tau**2*k**2*d.area()/(4*np.pi*np.pi)))
    def number_of_points(d,N):
            a = d.numElements * N**2
            b = d.edges * (N-2)
            c = d.corners * 3
            d = d.extraordinary_points
            return a-b-c+d    
    for d in mesh.dList:
        N=1
        if numSource == 0:
            while number_of_points(d,N) < d.numSource: N+=1
        else:
            while number_of_points(d,N) < numSource: N+=1
        d.numSource = number_of_points(d,N)
        xi1 = np.linspace(d.eList[0].limits[0],d.eList[0].limits[1],N)
        xi2 = np.linspace(d.eList[0].limits[2],d.eList[0].limits[3],N)        
        xi1,xi2=np.meshgrid(xi1,xi2)
        xi1=xi1.reshape(-1,) ; xi2=xi2.reshape(-1,)  
        souvals = d.eList[0].vals(d.eList[0].limits[0],d.eList[0].limits[2])
        sounorms = d.eList[0].normals(d.eList[0].limits[0],d.eList[0].limits[2])
        for e in d.eList:
            newvals = e.vals(xi1,xi2)
            newnorms = e.normals(xi1,xi2)         
            px,py,pz = newvals
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)
            qx,qy,qz=newvals
            rx=px-qx ; ry=py-qy ; rz=pz-qz
            rx=np.tril(rx, -1)[:,:-1]+np.triu(rx, 1)[:,1:]
            ry=np.tril(ry, -1)[:,:-1]+np.triu(ry, 1)[:,1:]
            rz=np.tril(rz, -1)[:,:-1]+np.triu(rz, 1)[:,1:]
            r = np.sqrt( rx**2 + ry**2 + rz**2 )
            delete = np.where(np.any(r<1e-10,axis=1))[0]
            newvals = np.delete(newvals,delete[1:],axis=1)
            newnorms = np.delete(newnorms,delete[1:],axis=1)
            px,py,pz = newvals
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)            
            qx,qy,qz=souvals
            r = np.sqrt( (qx-px)**2 + (qy-py)**2 + (qz-pz)**2 )
            delete = np.where(np.any(r<1e-12,axis=1))[0]
            souvals = np.hstack([souvals,np.delete(newvals,delete,axis=1)])
            sounorms = np.hstack([sounorms,np.delete(newnorms,delete,axis=1)])      
        d.sourceVals = souvals + offset*sounorms
        d.sourceNormals = sounorms
    mesh.sourceVals = np.hstack([d.sourceVals for d in mesh.dList])
    mesh.sourceNormals = np.hstack([d.sourceNormals for d in mesh.dList])
    
    for d in mesh.dList:
        N=1
        if numSamp == 0:
            while number_of_points(d,N) < frac_samp*d.numSource: N+=1
        else:
            while number_of_points(d,N) < numSource: N+=1
        d.numSamp = number_of_points(d,N)
        xi1 = np.linspace(d.eList[0].limits[0],d.eList[0].limits[1],N)
        xi2 = np.linspace(d.eList[0].limits[2],d.eList[0].limits[3],N)        
        xi1,xi2=np.meshgrid(xi1,xi2)
        xi1=xi1.reshape(-1,) ; xi2=xi2.reshape(-1,)      
        sampvals = d.eList[0].vals(d.eList[0].limits[0],d.eList[0].limits[0])
        sampnorms = d.eList[0].normals(d.eList[0].limits[0],d.eList[0].limits[0])
        for e in d.eList:
            newvals = e.vals(xi1,xi2)
            newnorms = e.normals(xi1,xi2)         
            px,py,pz = newvals
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)
            qx,qy,qz=newvals
            rx=px-qx ; ry=py-qy ; rz=pz-qz
            rx=np.tril(rx, -1)[:,:-1]+np.triu(rx, 1)[:,1:]
            ry=np.tril(ry, -1)[:,:-1]+np.triu(ry, 1)[:,1:]
            rz=np.tril(rz, -1)[:,:-1]+np.triu(rz, 1)[:,1:]
            r = np.sqrt( rx**2 + ry**2 + rz**2 )
            delete = np.where(np.any(r<1e-10,axis=1))[0]
            newvals = np.delete(newvals,delete[1:],axis=1)
            newnorms = np.delete(newnorms,delete[1:],axis=1)
            px,py,pz = newvals
            px=px.reshape(-1,1)
            py=py.reshape(-1,1)
            pz=pz.reshape(-1,1)            
            qx,qy,qz=sampvals
            r = np.sqrt( (qx-px)**2 + (qy-py)**2 + (qz-pz)**2 )
            delete = np.where(np.any(r<1e-12,axis=1))[0]    
            sampvals = np.hstack([sampvals,np.delete(newvals,delete,axis=1)])
            sampnorms = np.hstack([sampnorms,np.delete(newnorms,delete,axis=1)])      
        d.sampVals = sampvals
        d.sampNormals = sampnorms
    mesh.sampVals = np.hstack([d.sampVals for d in mesh.dList])
    mesh.sampNormals = np.hstack([d.sampNormals for d in mesh.dList])      
    
    dphidn = evaluate_dphidn(mesh,k,incAmp,incDir) # derivative phi_inc wrt n   
    T = evaluate_T(mesh,k)
    A = np.dot(T,T.T)
    b = np.sum(-T*dphidn,axis=1)
    mesh.amplitudes = np.linalg.solve(A,b)
    return mesh

def converged_MFS(mesh,plotx,ploty,plotz,k,incDir,incAmp=1.0,mintau=1,maxtau=20,steps=38):
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
    testz=plotz[0]
    taus=np.linspace(mintau,maxtau,steps+1)
    vals = np.zeros((steps+1,),dtype=np.complex)
    for i in xrange(steps+1):
	vals[i] = MFS(mesh,testx,testy,testz,k,incDir,incAmp,taus[i])[0]
    #vals=[MFS(mesh,testx,testy,testz,k,incDir,incAmp,tau) for tau in taus]
    vals = np.array([np.abs(vals[i]-vals[i+1]) for i in xrange(steps)])
    vals[np.where(vals==0)[0]]=100
    tau = taus[ np.where(vals==np.min(vals))[0][0] +1 ]
    print vals
    print "MFS solution settled at tau: %.2f" % (tau)
    return MFS(mesh,plotx,ploty,plotz,k,incDir,incAmp,tau)





if  __name__ == "__main__":

    #k=100
    #phi = np.zeros((100,),np.complex)
    #x = np.linspace(-1,1,100)
    #for i in xrange(100):
    #    phi[i] = SpherePlaneWave(k,x[i])[0]
    
    #import ResearchMeshes as Meshes
    #mesh = Meshes.CubeToExactSerendipitySphere()
    #
    #theta = np.linspace(0,2*np.pi,101)
    #x = np.cos(theta)
    #y = np.zeros(theta.shape)
    #z = np.sin(theta)
    #
    #k = 1.0
    #incDir = [1.0,0.0,0.0]
    #
    #phi = MFS(mesh,x,y,z,k,incDir,incAmp=1.0,tau=10,frac_samp=2,offset=0.15)
    phi, theta = np.mgrid[0:np.pi:200j,0:2*np.pi:200j]
    x = np.sin(phi)*np.cos(theta)
    x.reshape(-1,)
    phiexact = SpherePlaneWave(20,x)
    phiexact=phiexact.reshape(200,200)
    from pickle import dump
    savefile = open("plotdata.pkl","wb")
    dump([phiexact],savefile)
    savefile.close()
    #print getError(phiexact,phi)
    
    #import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')
    #ax.scatter(mesh.sourceVals[0],mesh.sourceVals[1],mesh.sourceVals[2],c='r',marker='o')
    #ax.scatter(mesh.sampVals[0],mesh.sampVals[1],mesh.sampVals[2],c='b',marker='^')
    #plt.show()
