# Michael Peake
# Durham University

import numpy as np
from scipy.special.orthogonal import p_roots
from warnings import simplefilter

# class GaussQuadrature at bottom of file

class Assembler(object):
    
    def __init__(self,mesh,wavenumber,incident_wave_amplitude,direction_of_propagation):
        """Returns an object that can assemble system matrices and vectors"""
        self.mesh = mesh
        self.k = wavenumber
        self.matrixCols = mesh.ndof
        self.matrixRows = mesh.collocation_points.shape[1]
        self.numBoundaryCollocation = mesh.numBoundaryCollocation
        self.incident_wave_amplitude = incident_wave_amplitude
        self.direction_of_propagation = direction_of_propagation
        
    def Incident_Wave_Vector(self):
        return self.incident_wave_amplitude*np.exp(1j*self.k*np.dot(self.mesh.collocation_points.T,self.direction_of_propagation))
        
    def Helmholtz_CBIE_Matrix(self,integration_order=6,Telles=True):
        quadrature=GaussQuadrature(integration_order,integration_order)
        print "Evaluating system matrix"
        system = self.Assemble_Helmholtz_CBIE_Matrix(quadrature,Telles)
        system[:self.numBoundaryCollocation,:self.matrixCols] += self.Assemble_Jump_Term_Matrix(0.5)
        return system
        
    def Helmholtz_RBIE_Matrix(self,integration_order=6):
        quadrature=GaussQuadrature(integration_order,integration_order)
        print "Evaluating system matrix"
        system = self.Assemble_Helmholtz_RBIE_Matrix(quadrature)
        system[:self.numBoundaryCollocation,:self.matrixCols] += self.Assemble_Jump_Term_Matrix(1.0)
        return system




##############################################################################   
#####                                                                    ##### 
#####                   Helmholtz CBIE MATRIX ASSEMBLY                   ##### 
#####                                                                    ##### 
##############################################################################  

    def Assemble_Helmholtz_CBIE_Matrix(self,quadrature,Telles):
        
        System_Matrix = np.zeros((self.matrixRows,self.matrixCols),np.complex)
     
        print "%s percent complete" % 0
        Es = np.array(self.mesh.eList)
        E = len(self.mesh.eList)
        comp_old = 25

        for e in self.mesh.eList:
            ## To see how far through simulation is
            eNum = np.where(e==Es)[0][0]
            if 100*eNum/E > comp_old:
                print "%s percent complete" % comp_old
                comp_old += 25
            ##
            Element_Matrix = self.Assemble_Helmholtz_CBIE_Element(e,quadrature,Telles)
            DegreeOfFreedomMapping = [dof.ID for s in e.shapeFunList for dof in s.DegreesOfFreedomList]
            # Next lines are different to 2D because you can have a repeated degree of freedom on the same element
            # when that element has a collapsed side
            # The following line combines the values of shared DoFs
            Unique_DoFs=np.unique(DegreeOfFreedomMapping)
            Element_Matrix=np.hstack([np.sum(Element_Matrix[:,np.where(DegreeOfFreedomMapping==dof)[0]],axis=1).reshape(-1,1) for dof in Unique_DoFs])
            System_Matrix[:,Unique_DoFs] += Element_Matrix
            
        print "%s percent complete" % 100
            
        return System_Matrix
            
    
    def Assemble_Helmholtz_CBIE_Element(self,e,quadrature,Telles):
  
        px,py,pz = self.mesh.collocation_points
        px=px.reshape(-1,1)
        py=py.reshape(-1,1)
        pz=pz.reshape(-1,1)
           
        # Change of integration interval to element local coordinate limits
        xi1 = (quadrature.xi1*0.5*(e.limits[1]-e.limits[0])+0.5*(e.limits[0]+e.limits[1])).reshape(-1,)
        xi2 = (quadrature.xi2*0.5*(e.limits[3]-e.limits[2])+0.5*(e.limits[2]+e.limits[3])).reshape(-1,)
        w1 = quadrature.w1*0.5*(e.limits[1]-e.limits[0])
        w2 = quadrature.w2*0.5*(e.limits[3]-e.limits[2])
        
        # Create integration subdivisions for large elements 
        # Force integration cells to be a maximum of lambda/4 in length
        #
        # Finding the area of polynomial BEM elements seems to take quite a while
        # so this only done for enriched simulations 
        try:
            e.num_integration_cells
        except:
            e.num_integration_cells=1
            if self.mesh.enriched:
                side_length = np.sqrt(e.area())
                if side_length > (2*np.pi/self.k)/4 :
                    e.num_integration_cells = int(np.ceil(2.0*self.k*side_length/np.pi))
        s = np.arange(0,e.num_integration_cells).reshape(-1,1)
        xi1 = ((xi1 - e.limits[0] + (e.limits[1]-e.limits[0])*s)/e.num_integration_cells + e.limits[0]).reshape(-1,)
        xi2 = ((xi2 - e.limits[2] + (e.limits[3]-e.limits[2])*s)/e.num_integration_cells + e.limits[2]).reshape(-1,)
        w1 = np.repeat([w1/e.num_integration_cells],e.num_integration_cells,axis=0).reshape(-1,)
        w2 = np.repeat([w2/e.num_integration_cells],e.num_integration_cells,axis=0).reshape(-1,)
        xi1,xi2 = np.meshgrid(xi1,xi2)
        w1,w2 = np.meshgrid(w1,w2)
        xi1=xi1.reshape(-1,); xi2=xi2.reshape(-1,)
        w1=w1.reshape(1,-1); w2=w2.reshape(1,-1)       
        Jacobian = w1*w2
        
        try: e.ExactElement
        except: e.ExactElement=False
        
        if e.ExactElement: 
            qx,qy,qz = e.vals(xi1,xi2)
            nq = e.normals(xi1,xi2)
            Jxi = e.J(xi1,xi2) 
            ShapeFunctions = e.shape_functions(xi1,xi2)
        else:
            # Save evaluating shape functions many times
            ShapeFunctions = e.shape_functions(xi1,xi2,1)
            qx,qy,qz = np.dot(e.P,ShapeFunctions[:,:,0].T)
            Jxi = np.zeros((np.size(xi1),2,3))
            Jxi[:,0,:] = np.dot(e.P,ShapeFunctions[:,:,1].T).T
            Jxi[:,1,:] = np.dot(e.P,ShapeFunctions[:,:,2].T).T
            Jxi = np.cross(Jxi[:,0,:],Jxi[:,1,:])
            nq = Jxi.T/np.sqrt(np.sum(Jxi**2,axis=1))
            Jxi = np.sqrt(np.sum(Jxi**2,axis=1))
            
        Jacobian *= Jxi    
       
        r = np.sqrt((qx-px)**2+(qy-py)**2+(qz-pz)**2)
        drdnq = ((qx-px)*nq[0]+(qy-py)*nq[1]+(qz-pz)*nq[2])/r

        # Evaluate kernel
        Kernel = np.exp(1j*self.k*r)
        Kernel *= (1j*self.k*r-1)
        Kernel *= drdnq
        simplefilter("ignore") # This simple filter stops warnings about divisions by 0
        Kernel /= (4*np.pi*r**2)
        simplefilter("default")
        Kernel[np.where(np.isnan(Kernel))]=0.0 # If an entry is NaN, replace by 0.0
        ######
        
        InterpolationBasis = np.repeat(ShapeFunctions[:,:,0],e.shapeFunList[0].M,axis=1) # Assumes global M !!!!
        InterpolationBasis = np.asarray(InterpolationBasis,np.complex) # Make InterpolationBasis a complex matrix
        InterpolationBasis *= np.vstack([dof(qx,qy,qz) for s in e.shapeFunList for dof in s.DegreesOfFreedomList]).T
            
        Element_Matrix = np.dot(Kernel,Jacobian.reshape(-1,1)*InterpolationBasis)
    
        ############################################################
        # Coordinate transformation for wealkly singular integrals #
        ############################################################
        if Telles:
        # Which rows need the transformation?
            for i in xrange(e.collocation_points.shape[1]):
                where=[np.where(np.abs(self.mesh.collocation_points[j]-e.collocation_points[j,i])<1e-10)[0] for j in xrange(3)]
                for j in xrange(where[0].size):
                    if where[0][j] in where[1]:
                        if where[0][j] in where[2]:
                            p = where[0][j]
                px,py,pz = self.mesh.collocation_points[:,p]
                xi1,xi2,w1,w2=quadrature.Telles(e.collocationXi[0,i],e.collocationXi[1,i],e,self.mesh,self.k)
                xi1 = (xi1*0.5*(e.limits[1]-e.limits[0])+0.5*(e.limits[0]+e.limits[1])).reshape(-1,)
                xi2 = (xi2*0.5*(e.limits[1]-e.limits[0])+0.5*(e.limits[0]+e.limits[1])).reshape(-1,)
                w1 = (w1*0.5*(e.limits[1]-e.limits[0])).reshape(1,-1)
                w2 = (w2*0.5*(e.limits[1]-e.limits[0])).reshape(1,-1)
                Jacobian = w1*w2
                                
                if e.ExactElement: 
                    qx,qy,qz = e.vals(xi1,xi2)
                    nq = e.normals(xi1,xi2)
                    Jxi = e.J(xi1,xi2) 
                    ShapeFunctions = e.shape_functions(xi1,xi2)
                else:
                    # Save evaluating shape functions many times
                    ShapeFunctions = e.shape_functions(xi1,xi2,1)
                    qx,qy,qz = np.dot(e.P,ShapeFunctions[:,:,0].T)
                    Jxi = np.zeros((np.size(xi1),2,3))
                    Jxi[:,0,:] = np.dot(e.P,ShapeFunctions[:,:,1].T).T
                    Jxi[:,1,:] = np.dot(e.P,ShapeFunctions[:,:,2].T).T
                    Jxi = np.cross(Jxi[:,0,:],Jxi[:,1,:])
                    nq = Jxi.T/np.sqrt(np.sum(Jxi**2,axis=1))
                    Jxi = np.sqrt(np.sum(Jxi**2,axis=1))
                      
                Jacobian *= Jxi    
                r = np.sqrt((qx-px)**2+(qy-py)**2+(qz-pz)**2)
                drdnq = ((qx-px)*nq[0]+(qy-py)*nq[1]+(qz-pz)*nq[2])/r
                # Evaluate kernel
                Kernel = np.exp(1j*self.k*r)
                Kernel *= (1j*self.k*r-1)
                Kernel *= drdnq
                simplefilter("ignore") # This simple filter stops warnings about divisions by 0
                Kernel /= (4*np.pi*r**2)
                simplefilter("default")
                Kernel[np.where(np.isnan(Kernel))]=0.0 # If an entry is NaN, replace by 0.0
                ######
                InterpolationBasis = np.repeat(ShapeFunctions[:,:,0],e.shapeFunList[0].M,axis=1) # Assumes global M !!!!
                InterpolationBasis = np.asarray(InterpolationBasis,np.complex) # Make InterpolationBasis a complex matrix
                InterpolationBasis *= np.vstack([dof(qx,qy,qz) for s in e.shapeFunList for dof in s.DegreesOfFreedomList]).T
                
                Element_Matrix[p] = np.dot(Kernel,Jacobian.reshape(-1,1)*InterpolationBasis)
        
        #############################################################
    
        return Element_Matrix
        
        
        
        

##############################################################################   
#####                                                                    ##### 
#####                   Helmholtz RBIE MATRIX ASSEMBLY                   ##### 
#####                                                                    ##### 
##############################################################################  
            
            
            
    def Assemble_Helmholtz_RBIE_Matrix(self,quadrature):
        """Assemble a system matrix using the supplied kernel"""
    
        # Determine matrix size
        System_Matrix = np.zeros((self.matrixRows,self.matrixCols),np.complex)
        System_Vector = np.zeros((self.numBoundaryCollocation,1),np.complex) # For subtracted for singular part
    
        print "%s percent complete" % 0
        Es = np.array(self.mesh.eList)
        E = len(self.mesh.eList)
        comp_old = 25

        for e in self.mesh.eList:
            
            ## To see how far through simulation is
            eNum = np.where(e==Es)[0][0]
            if 100*eNum/E > comp_old:
                print "%s percent complete" % comp_old
                comp_old += 25
            ##
            
            # Change of integration interval etc. is done here as for large elements, a situation can occur
            # where there are so many integration points that we get memory errors.
            # We work out the local coordinates first and then work out the integral row-by-row of integration
            # cells. It can also be worked out, cell by cell if required
            
            # Change of integration interval to element local coordinate limits
            xi1 = (quadrature.xi1*0.5*(e.limits[1]-e.limits[0])+0.5*(e.limits[0]+e.limits[1])).reshape(-1,)
            xi2 = (quadrature.xi2*0.5*(e.limits[3]-e.limits[2])+0.5*(e.limits[2]+e.limits[3])).reshape(-1,)
            w1 = quadrature.w1*0.5*(e.limits[1]-e.limits[0])
            w2 = quadrature.w2*0.5*(e.limits[3]-e.limits[2])
            
            # Create integration subdivisions for large elements 
            # Force integration cells to be a maximum of lambda/4 in length
            #
            # Finding the area of polynomial BEM elements seems to take quite a while
            # so this only done for enriched simulations 
            try:
                e.num_integration_cells
            except:
                e.num_integration_cells=1
                if self.mesh.enriched:
                    side_length = np.sqrt(e.area())
                    if side_length > (2*np.pi/self.k)/4 :
                        e.num_integration_cells = int(np.ceil(2.0*self.k*side_length/np.pi))

            s = np.arange(0,e.num_integration_cells).reshape(-1,1)
            xi1 = ((xi1 - e.limits[0] + (e.limits[1]-e.limits[0])*s)/e.num_integration_cells + e.limits[0]).reshape(-1,)
            xi2 = ((xi2 - e.limits[2] + (e.limits[3]-e.limits[2])*s)/e.num_integration_cells + e.limits[2]).reshape(-1,)
            w1 = np.repeat([w1/e.num_integration_cells],e.num_integration_cells,axis=0).reshape(-1,)
            w2 = np.repeat([w2/e.num_integration_cells],e.num_integration_cells,axis=0).reshape(-1,)
            xi1,xi2 = np.meshgrid(xi1,xi2)
            w1,w2 = np.meshgrid(w1,w2)
            xi1=xi1.reshape(-1,); xi2=xi2.reshape(-1,)
            w1=w1.reshape(-1); w2=w2.reshape(-1)
            

            S = e.num_integration_cells
            # quadmap : quadrature map
            # Loop either in xrange(S) --fast , or xrange(S**2) should never be memory error from integration
            for s in xrange(S):
                quadmap = range(s*S*quadrature.n1*quadrature.n2,(s+1)*S*quadrature.n1*quadrature.n2)
            #for s in xrange(S**2):     
                #quadmap = range(s*quadrature.n1*quadrature.n2,(s+1)*quadrature.n1*quadrature.n2)     
                Element_Matrix,Element_Vector = self.Assemble_Helmholtz_RBIE_Element(e,xi1[quadmap],xi2[quadmap],w1[quadmap],w2[quadmap])  
                System_Vector += Element_Vector[:self.numBoundaryCollocation]
                DegreeOfFreedomMapping = [dof.ID for s in e.shapeFunList for dof in s.DegreesOfFreedomList]
                Unique_DoFs=np.unique(DegreeOfFreedomMapping)
                Element_Matrix=np.hstack([np.sum(Element_Matrix[:,np.where(DegreeOfFreedomMapping==dof)[0]],axis=1).reshape(-1,1) for dof in Unique_DoFs])
                System_Matrix[:,Unique_DoFs] += Element_Matrix
        
            
        print "%s percent complete" % 100
        
        
        System_Matrix[:self.numBoundaryCollocation] -= System_Vector*self.Assemble_Jump_Term_Matrix(1.0)
        
        return System_Matrix
    
    def Assemble_Helmholtz_RBIE_Element(self,e,xi1,xi2,w1,w2):
        
        px,py,pz = self.mesh.collocation_points
        px=px.reshape(-1,1)
        py=py.reshape(-1,1)
        pz=pz.reshape(-1,1)  
        
        Jacobian = (w1*w2).reshape(1,-1)   
        
        try: e.ExactElement
        except: e.ExactElement=False
        
        if e.ExactElement: 
            qx,qy,qz = e.vals(xi1,xi2)
            nq = e.normals(xi1,xi2)
            Jxi = e.J(xi1,xi2) 
            ShapeFunctions = e.shape_functions(xi1,xi2)
        else:
            # Save evaluating shape functions many times
            ShapeFunctions = e.shape_functions(xi1,xi2,1)
            qx,qy,qz = np.dot(e.P,ShapeFunctions[:,:,0].T)
            Jxi = np.zeros((np.size(xi1),2,3))
            Jxi[:,0,:] = np.dot(e.P,ShapeFunctions[:,:,1].T).T
            Jxi[:,1,:] = np.dot(e.P,ShapeFunctions[:,:,2].T).T
            Jxi = np.cross(Jxi[:,0,:],Jxi[:,1,:])
            nq = Jxi.T/np.sqrt(np.sum(Jxi**2,axis=1))
            Jxi = np.sqrt(np.sum(Jxi**2,axis=1))
            
        Jacobian *= Jxi 
                                     
        r = np.sqrt((qx-px)**2+(qy-py)**2+(qz-pz)**2)
        simplefilter("ignore")
        drdnq = ((qx-px)*nq[0]+(qy-py)*nq[1]+(qz-pz)*nq[2])/r
        simplefilter("default")

        # Evaluate kernel
        Kernel1 = np.exp(1j*self.k*r)
        Kernel1 *= (1j*self.k*r-1)
        Kernel1 *= drdnq
        simplefilter("ignore") # This simple filter stops warnings about divisions by 0
        Kernel1 /= (4*np.pi*r**2)
        Kernel2 = -1 / (4*np.pi*r**2) * drdnq
        simplefilter("default")
        Kernel1[np.where(np.isnan(Kernel1))]=0.0 # If an entry is NaN, replace by 0.0
        Kernel2[np.where(np.isnan(Kernel2))]=0.0
        ######
        
        InterpolationBasis = np.repeat(ShapeFunctions[:,:,0],e.shapeFunList[0].M,axis=1) # Assumes global M !!!!
        InterpolationBasis = np.asarray(InterpolationBasis,np.complex) # Make InterpolationBasis a complex matrix
        InterpolationBasis *= np.vstack([dof(qx,qy,qz) for s in e.shapeFunList for dof in s.DegreesOfFreedomList]).T
        
        Element_Matrix = np.dot(Kernel1,Jacobian.reshape(-1,1)*InterpolationBasis)
        Element_Vector = Kernel2 * Jacobian
        Element_Vector = np.sum(Element_Vector,axis=1).reshape(-1,1)

        return Element_Matrix,Element_Vector
            
            
            
            
##############################################################################   
#####                                                                    ##### 
#####                   JUMP TERM MATRIX ASSEMBLY                        ##### 
#####                                                                    ##### 
############################################################################## 

    def Assemble_Jump_Term_Matrix(self,coefficient):
        Jump_Term_Matrix = np.zeros((self.numBoundaryCollocation,self.matrixCols),np.complex)
        
        for e in self.mesh.eList:
            
            #if e.collocationXi == None: continue
            
            Element_Matrix,CollocationPointMapping = self.Assemble_Jump_Term_Element(e)
            DoF_map = [dof.ID for s in e.shapeFunList for dof in s.DegreesOfFreedomList]
            ### NEEDED FOR SHARED DOFS ON ELEMENTS ###
            Unique_DoFs=np.unique(DoF_map)
            Element_Matrix=np.hstack([np.sum(Element_Matrix[:,np.where(DoF_map==dof)[0]],axis=1).reshape(-1,1) for dof in Unique_DoFs])
            ##########################################
            
            ## The two mappings put the entries of Element_Matrix in the right place
            Jump_Term_Matrix[np.meshgrid(CollocationPointMapping,Unique_DoFs)] = Element_Matrix.T

        return coefficient * Jump_Term_Matrix

            

        
    
    def Assemble_Jump_Term_Element(self,e):
        
        qx,qy,qz = e.collocation_points

        # Evaluate basis for each DOF
        InterpolationBasis = e.shape_functions(e.collocationXi[0],e.collocationXi[1])[:,:,0]
        InterpolationBasis = np.repeat(InterpolationBasis,e.shapeFunList[0].M,axis=1) # Assumes global M !!!!
        InterpolationBasis = np.asarray(InterpolationBasis,np.complex) # Make InterpolationBasis a complex matrix
        InterpolationBasis *= np.vstack([dof(qx,qy,qz) for s in e.shapeFunList for dof in s.DegreesOfFreedomList]).T
    
        # Row finding algorithm
        CollocationPointMapping=[]
        for i in xrange(e.collocation_points.shape[1]):
            where=[np.where(np.abs(self.mesh.collocation_points[j]-e.collocation_points[j,i])<1e-10)[0] for j in xrange(3)]
            for j in xrange(where[0].size):
                if where[0][j] in where[1]:
                    if where[0][j] in where[2]:
                        CollocationPointMapping.append(where[0][j])
    
        return InterpolationBasis,CollocationPointMapping





##############################################################################   
#####                                                                    ##### 
#####                     QUADRATURE CLASS                               ##### 
#####                                                                    ##### 
############################################################################## 




class GaussQuadrature:
    """Container class containing 3D quadrature points"""
    def __init__(self,n1,n2):
        
        self.n1=n1
        self.n2=n2
        self.xi1,self.w1 = p_roots(n1)
        self.xi2,self.w2 = p_roots(n2)

    def Telles(self,pxi1,pxi2,e,mesh,k):
        # Collocation point at (pxi1,pxi2)
        # See research log pg. 789
        xy = np.array(([-1,1,-1,-1],  # T1
                       [1,1,-1,1],    # T2
                       [1,-1,1,1],    # T3
                       [-1,-1,1,-1])) # T4
        # Ftheta function
        def F(i,theta):
            if i==0:
                return (-1-pxi2)/np.sin(theta)
            elif i==1:
                return (1-pxi1)/np.cos(theta)
            elif i==2:
                return (1-pxi2)/np.sin(theta)
            elif i==3:
                return (-1-pxi1)/np.cos(theta)

        # Which triangles to calculate
        which = np.array([False,False,False,False])
        if pxi2>-1:
            which[0]=True
        if pxi1<1:
            which[1]=True
        if pxi2<1:
            which[2]=True
        if pxi1>-1:
            which[3]=True
 
        xi1=[]
        xi2=[]
        w1=[]
        w2=[]
        J=[]
        for i in xrange(4):
            if which[i]:
                
                x1,x2,y1,y2 = xy[i]
                theta1 = np.arctan2(y1-pxi2,x1-pxi1)
                theta2 = np.arctan2(y2-pxi2,x2-pxi1)
                if theta1<0: theta1+=2*np.pi
                if i==1 and theta1>0: theta1-=2*np.pi
                if theta2<0: theta2+=2*np.pi
                theta,wtheta = p_roots(self.n1)
                theta = theta*0.5*(theta2-theta1)+0.5*(theta1+theta2)
                wtheta = wtheta*0.5*(theta2-theta1)
                
                if e.num_integration_cells>1:
                    s = np.arange(0,e.num_integration_cells).reshape(-1,1)
                    theta = ((theta - theta1 + (theta2-theta1)*s)/e.num_integration_cells + theta1).reshape(-1,)
                    wtheta = np.repeat([wtheta/e.num_integration_cells],e.num_integration_cells,axis=0).reshape(-1,)

                Ftheta = F(i,theta)
                theta = (np.repeat(theta,self.n2))
                wtheta = (np.repeat(wtheta,self.n2))
                rho,wrho = p_roots(self.n2)
                
                if e.num_integration_cells>1:
                    rho = ((rho - 1 + 2*s)/e.num_integration_cells + 1).reshape(-1,)
                    wrho = np.repeat([wrho/e.num_integration_cells],e.num_integration_cells,axis=0).reshape(-1,)
                        
                rho = np.vstack([rho*0.5*Ftheta[i]+0.5*Ftheta[i]] for i in xrange(self.n1)).reshape(-1,)
                wrho = np.vstack([wrho*0.5*Ftheta[i]] for i in xrange(self.n1)).reshape(-1,)

                xi1.append(rho*np.cos(theta)+pxi1)
                xi2.append(rho*np.sin(theta)+pxi2)
                w1.append(wtheta)
                w2.append(wrho)
                J.append(rho)
            
        xi1=np.array(xi1).reshape(-1,)
        xi2=np.array(xi2).reshape(-1,)
        w1=np.array(w1).reshape(-1,)
        w2=np.array(w2).reshape(-1,)
        J=np.array(J).reshape(-1,)

        return xi1,xi2,w1*J,w2*J
        
        
        
        
        
