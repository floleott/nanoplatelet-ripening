
"""
Simulation platform for nanoplatelets simulations.

Copyright (C) 2016 Florian Ott, David Ochsenbein

"""
# import statements
import numpy as np # lovely numpy
import matplotlib.pyplot as plt # plotting functions
from scipy.integrate import odeint # ode solver
from scipy import constants # import some pretty constants
import warnings # let's us warn users
import operator
import pdb
import multiprocessing
import itertools as itt
from popRatesFinal import *

np.seterr(all='warn')

class nplSimus():
    """
    nplSimus Class - for the calculation and plotting of nanoplatelet simulations.
    """
    def __init__(self, m = 6 , Nm = np.zeros(7), t = np.logspace(-2,20,1000), y0 = np.append(1E0,np.zeros(7)), params = np.array([1E0,1E8]), T = 450., rhoc = 1.0/59.7, Vsol = 1E2, sigma = 5.7, kappa = 37., cstar = 0.0*6.2E-3 + 1.0*3.52E-3 , V0 = 59.7, h = 3.10, newModel = 0, oldModel = 0 , popModel = 1, twoStep = 0 , tSplit = 0.0 , rc = 2.0 ):
        """
        Initialization stuff is written into two dicts, physicalProperties and modelSetup.
        """
        #np.seterr(all='warn') #prints a warning to stdout whenever there is a problematic np operation
        # dict containing all physical/thermodynamic properties units for now are meV, Angstrom for latt. related lengths, liters for conc. related volumes, and seconds
        self.physicalProperties = {
            'rhoc': rhoc,  # crystal density [monomer / A^3]
            'sigma' : sigma, # surface tension [meV/A^2]
            'kappa' : kappa, # line energy [meV/A]
            'cstar' : cstar, # bulk solubility [monomer / A^3??]
            'V0' : V0, # molecular volume [A^3]
            'h' : h, # Island/step height [A]
            'kB': constants.k/constants.eV*1E3}  # Boltzmann const. [meV/K]

        # dict containing exp. or modeling decisions
        self.modelSetup = {
            't' : t,   # time horizon for model [s]
            'm' : m, # maximum thickness to be considered [monomers]
            'Nm' : Nm, # number of platelets of thickness m per volume [#/m^3]
            'y0' : y0, # initial conditions [concentration, m platelets widths] (see units above)
            'T' : T, # temperature [K]
            'Vsol' : Vsol, # Total Volume of reaction solution
            'newModel' : newModel, # newModel switch 1/0 on/off
            'oldModel' : oldModel, # old model switch; 0 : don't use old model, 1 : use first-order old model, 2 : use second-order old model
            'popModel' : popModel, # population model switch; 0 : don't use population model, 1 : use it.
            'twoStep' : twoStep, # defines if there is a step-wise change in the concentration
            'tSplit' : tSplit, # is the time where the step-wise concentration change takes place
            'rc' : rc } # factor by which c is reduced at tSplit if twoStep=1

        self.params = params # kinetic parameters for growth (separate for potential fitting)
        
        # dict containing all simulation outputs
        self.output = {}
        
    def solve(self):
        """
        Take model and solve it using odeint
        """

        ## checking validity of inputs
        #if self.modelSetup['m'] != len(self.modelSetup['Nm']) or self.modelSetup['m'] + 1 != len(self.modelSetup['y0']):
        #    raise ValueError("The number of platelet types (m) and your initial conditions are inconsistent.")

        # Create a list of functions defining the rates (in length per time)
        gfuns = [lambda c,L,i=i: self.physicalProperties['h']*np.diff(self.rates(c,i,L))
                     for i in range(1,self.modelSetup['m']+1)]
        # the 'currying' (i=i) is important, but I don't fully understand it yet
        
        #creates a list of tuples (Dm,Im)[i]
        detAttFuns = [lambda c,L,i=i : self.rates(c,i,L)
                        for i in range(1,self.modelSetup['m']+1)]
        #same for the wide facet
        detAttFunsWide = [lambda c,L,i=i : self.orthoRates(c,i,L)
                          for i in range(1,self.modelSetup['m']+1)]
        
        if  self.modelSetup['newModel'] == 1:
            # check if there is a step in the concentration or not
            if self.modelSetup['twoStep'] == 0:
                # solve ode
                y = odeint(self.__plateletsEvolution,self.modelSetup['y0'],self.modelSetup['t'],args=(gfuns,),rtol=1e-12,atol=1e-5) # args passes additional arguments to function, notice the weird (LIST,)-structure to convert to tuple
                
                # write into output dict
                self.output['t'] = self.modelSetup['t'][::10]
                self.output['c'] = y[::10,0]
                self.output['Lm'] = y[::10,1:]
            
            elif self.modelSetup['twoStep'] == 1:
                # time domain of first concentration regime
                t1 = [t for t in self.modelSetup['t'] if t < self.modelSetup['tSplit']]
                # time space after concentration step
                t2 = self.modelSetup['t'][(len(t1)-1):]

                # debug: total material
                mtot = self.modelSetup['Vsol']*self.modelSetup['y0'][0] + sum([L**2*(m+1)*self.physicalProperties['rhoc'] for m,L in enumerate(self.modelSetup['y0'][1:])])
                print 'Initial Material ',mtot,' Vsol ',self.modelSetup['Vsol']
            
                # solve the first time domain
                y = odeint(self.__plateletsEvolution,self.modelSetup['y0'],t1,args=(gfuns,),rtol=1e-15,atol=1e-2)
                # write out first part
                self.output['t'] = self.modelSetup['t']
                self.output['c'] = y[:,0]
                self.output['Lm'] = y[:,1:]
                
                # debug: second check
                mtot = self.modelSetup['Vsol']*y[-1,0] + sum([L**2*(m+1)*self.physicalProperties['rhoc'] for m,L in enumerate(y[-1,1:])])
                print 'Before Step ',mtot,' Vsol ',self.modelSetup['Vsol']
                
                # set new initial conditions
                y1 = np.append(y[-1,0]/self.modelSetup['rc'],y[-1,1:]) # concentration reduced by factor 'rc'
                self.modelSetup['Vsol'] = self.modelSetup['Vsol']*self.modelSetup['rc'] # Volume increased by factor 'rc'
                
                # debug: third check
                mtot = self.modelSetup['Vsol']*y1[0] + sum([L**2*(m+1)*self.physicalProperties['rhoc'] for m,L in enumerate(y1[1:])])
                print 'AfterStep ',mtot,' Vsol ',self.modelSetup['Vsol']
                
                # solve the second time domain
                y = odeint(self.__plateletsEvolution,y1,t2,args=(gfuns,),rtol=1e-15,atol=1e-2)
                # debug at the end
                mtot = self.modelSetup['Vsol']*y[-1,0] + sum([L**2*(m+1)*self.physicalProperties['rhoc'] for m,L in enumerate(y[-1,1:])])
                print 'At the end ',mtot,' Vsol ',self.modelSetup['Vsol']
                self.modelSetup['Vsol'] = self.modelSetup['Vsol']/self.modelSetup['rc'] # recover initial value
                
                # update the output
                self.output['c'] = np.append(self.output['c'], y[1:,0])
                self.output['Lm'] = np.append(self.output['Lm'], y[1:,1:],axis = 0)
                
                
        if self.modelSetup['oldModel'] > 0:
            # inverse of eq. constant
            Amfun = lambda m: np.exp((2*self.physicalProperties['h']**3*(-1.5)+4*self.physicalProperties['h']**2*5.7/m)/(self.physicalProperties['kB']*self.modelSetup['T']))

            # find supersaturation value that gives you the ominous Ev from the paper
            Sold = np.exp(1.5*self.physicalProperties['V0']/(self.physicalProperties['kB']*self.modelSetup['T']))            
            
            # calculate forward rates
            km = [2.8e11*m*np.exp(-self.__bm(Sold,m)/(self.physicalProperties['kB']*self.modelSetup['T'])) for m in np.arange(1.,self.modelSetup['m']+1)] # note that this will not work if your cstar is not 1!
            
            # create matrix containing all necessary rates
            K = np.zeros([self.modelSetup['m']+1]*2)
            
            K[0,:] = [-np.sum(km)] + [Amfun(m+1)*k for (m,k) in enumerate(km)] # first row of K matrix
            for i in range(1,self.modelSetup['m']+1):
                K[i,0] = km[i-1]
                K[i,i] = -Amfun(i)*km[i-1]
                
            # translate initial conditions to old-style I.C.
            cm0 = np.append(self.modelSetup['y0'][0],
                            self.modelSetup['Nm'][0:]*self.modelSetup['y0'][1:]**2*np.arange(1.,self.modelSetup['m']+1)) 
                            
            
            cm = odeint(self.__oldModel,cm0,self.modelSetup['t'],args=(K,)) # args passes additional arguments to function, notice the weird (LIST,)-structure to convert to tuple
            
            # write into output dict
            self.output['old_cm'] = cm
            
        if self.modelSetup['popModel'] > 0:

            y0 = np.append([ 1.0*self.modelSetup['y0'][0] , 0.0*self.modelSetup['y0'][0] ],self.modelSetup['y0'][1:])
            h = self.physicalProperties['h']

            M = 100 #continuous bins
            N = 25 #discrete side lengths
            
            Ld = np.arange(1,N+1)
            Lt = np.zeros(M)
            
            Lt[0] = N+1 #first element of cont. part is one layer thick
            for i in np.arange(1,M):
                Lt[i] = Lt[i-1] + (1+0.1)**(i-1)
                
            # convert from no. of monolayers to length
            L = np.append(Ld,Lt) #*self.physicalProperties['h']
            
            contSpacing = np.ones(M)
            contIntSpacing = np.ones(M)
            for i in np.arange(1,M-1):
                contSpacing[i] = 1./2. * (Lt[i+1]-Lt[i-1])
                contIntSpacing[i] = Lt[i+1] - Lt[i]
                

            ylength = (N+M-1)*self.modelSetup['m']
            y = odeint(self.__populationEvolution,np.append(y0[:2],np.zeros( 2*ylength ) ),
                       self.modelSetup['t'], args=(M,N,L,contIntSpacing,contSpacing,detAttFuns,detAttFunsWide)
                       , full_output=1, mxstep = 1000000 , rtol = 1E-9 )
                       #, full_output=1, h0 = 1E-18 , hmin=1E-38, mxstep = 10000000 )  
            
            self.output['t'] = self.modelSetup['t'][::1]
            self.output['pop_L'] = L
            self.output['pop_y'] = y[0][::1,:ylength+2]
            self.output['fluxode'] = y[0][::1,ylength+2:]
            #self.output['odeint'] = y[1] # information about the odeint process

            #evaluate fluxes
            latFlux = np.zeros([self.modelSetup['m'],2,N-1]) # attach/detach / layers / side lengths (only over discrete population)
            layerFlux = np.zeros([self.modelSetup['m'],2,N-1])

            # export all the rates
            latRates=[]
            layerRates=[]
            for t,yt in zip(self.output['t'],self.output['pop_y']):
                c = yt[1] #conc.
                latRates.append(np.array([zip(*detAttFuns[m](c,Ld[1:]*h)) 
                                          for m,ym in enumerate(np.split(yt[2:],self.modelSetup['m']))] ))
                layerRates.append(np.array([zip(*detAttFunsWide[m](c,Ld[1:]*h))
                                            for m,ym in enumerate(np.split(yt[2:],self.modelSetup['m']))] ))
                                                     
            self.output['latRates'] = latRates
            self.output['layerRates'] = layerRates

            Vtot = [np.array([]) for i in range(self.modelSetup['m']+2)]
            ym = [np.array([]) for i in range(self.modelSetup['m']+1)]
            dZdt = [np.array([])]

            for t,yi in zip(self.output['t'],self.output['pop_y']):
                ym[0]=np.append(ym[0],yi[1])
                Vtot[0] = np.append(Vtot[0],yi[0])
                Vtot[1] = np.append(Vtot[1],yi[1])
                #dZdt = np.append(dZdt,self.__populationEvolution(yi,t,M,N,L,contIntSpacing,contSpacing,detAttFuns,detAttFunsWide))
                #introduce grid spacing for all the 
                dL = L[1:]-L[:-1]
                for j,yj in enumerate(np.split(yi[2:],self.modelSetup['m'])):

                    ySum = np.sum(yj*L[1:]*dL) * h if j>0 else ( np.sum(yj*L[1:]*dL) + yi[1]*L[0] ) * h
                    Vtot[j+2] = np.append(Vtot[j+2],np.sum( np.ceil( (j+1.) * Ld[1:]**2 /2) * yj[:N-1] ) + \
                                              np.sum( np.ceil( (j+1.) * (L[N:]+contIntSpacing/2.)**2 /2.) * yj[N-1:] * contIntSpacing ))

                    if ySum > 0.0:
                        ym[j+1]=np.append(ym[j+1],ySum/np.sum(yj*dL)) if j>0 else np.append(ym[j+1],ySum/( np.sum(yj*dL) + yi[1] ) )
                    else:
                        ym[j+1]=np.append(ym[j+1], h*(j+1.))                                 

            self.output['c'] = ym[0]
            self.output['Lm'] = ym[1:]
            self.output['Vtot'] = Vtot
            #self.output['dZdt'] = np.split(dZdt,len(self.output['t']))
            
            
    # simly returns the tuple (Dm,Im)
    def rates(self,c,m,Ls):
        """
        Calculate growth and dissolution rates as a function of concentration and NPL size.
        """
        # unpack stuff into local variables
        sigma, kappa, h, V0, cstar, kB = [self.physicalProperties[prop] for prop in ['sigma', 'kappa', 'h', 'V0', 'cstar', 'kB']]
        T = self.modelSetup['T']
        
        if type(Ls) != np.ndarray:
            Ls = np.array([Ls])

        Deltamu = lambda m,L: 2.*sigma*V0*(1.0/m/h + 1.0/L) # + 4*V0/h*kappa/(m*L)
        #ceqs = [min(cmax, np.exp(Deltamu(m,l)/(kB*T))*cstar) for l in Ls] #cstar(m,L)
        ceqs = np.exp(Deltamu(m,Ls)/(kB*T))*cstar  #cstar(m,L)

        # attachment rate
        if c > cstar:
            Im0 = self.params[0] * np.exp(-self.__bm(c,m)/(kB*T)) * (c/cstar)**2
            Ims = [ Im0 for i in ceqs]
        # pin c to cstar
        elif c < cstar: 
            c = 2.*cstar - c
            Im0 = self.params[0] * np.exp(-self.__bm(c,m)/(kB*T)) * (c/cstar)**2
            Ims = [ -Im0 for i in ceqs]
        else:
             Ims = [ 0.0 for i in ceqs]
        
        # detachment rate
        #Dms = [ min(self.params[0] * np.exp(-self.__bm(ceq,m)/(kB*T)) * (ceq/cstar)**2 , 1E12) for ceq in ceqs]
        Dms = [ self.params[0] * np.exp(-self.__bm(ceq,m)/(kB*T)) * (ceq/cstar)**2  for ceq in ceqs]
        
        # return rates as detachment, attachment tuple (order is such that np.diff(output) = g)
        # We need to take into account that the side length cannot become negative

        #case L = m*h might be wrong for the population equation, but Dm is luckily not used there for L=m*h
        #return [(Dm, Im)  if L > m*h else (np.min([Im,Dm]), Im) if L == m*h else (0.,0.) for (Dm,Im,L) in zip(Dms,Ims,Ls)] # growth and dissolution
        return [(Dm, Im)  if L >= m*h else (0.,0.) for (Dm,Im,L) in zip(Dms,Ims,Ls)] # growth and dissolution 

    # returns the growth rates on the wide facet
    def orthoRates(self,c,m,Ls):
        sigma, kappa, h, V0, cstar, kB = [self.physicalProperties[prop] for prop in ['sigma', 'kappa', 'h', 'V0', 'cstar', 'kB']]
        T = self.modelSetup['T']
        
        if type(Ls) != np.ndarray:
            Ls = np.array([Ls])
            
        Deltamu = lambda L: 4.*sigma*V0/L #very important!! the solubility is orientation-dependent!
        ceqs = np.exp(Deltamu(Ls)/(kB*T))*cstar  #cstar(m,L)
        #side length is facet width
        Lm = np.array([ int(Li/h) for Li in Ls ])

        #Attachment - factor 0.5 because growth into two instead of 4 directions
        if c > cstar:
            ImL = [ self.params[0] * np.exp(-self.__bm(c,Li)/(kB*T)) * (c/cstar)**2 for Li in Lm ]
        elif c < cstar:
            c =2.*cstar -c
            ImL = [ - self.params[0] * np.exp(-self.__bm(c,Li)/(kB*T)) * (c/cstar)**2 for Li in Lm ]
        else:
             ImL = [ 0.0 for Li in Lm ]

        #Detachment
        DmL = [ self.params[0] * np.exp(-self.__bm(ceq,Li)/(kB*T)) * (ceq/cstar)**2  for ceq,Li in zip(ceqs,Lm) ]

        #return [(Dm, Im)  if L >= m*h else (0.,0.) for (Dm,Im,L) in zip(DmL,ImL,Ls)]
        return zip(DmL,ImL)
        #return [(Dm, Im)  if L >= m*h else (0.,0.) for (Dm,Im,L) in zip(DmL,ImL,Ls)] 
    

    def __plateletsEvolution(self,y,t,gfuns):
        """
        Calculate r.h.s. of ODEs for new model. 
        """
        # the double underscore makes this method hidden (or private? what's the difference in python?)
        c = y[0]
        Lm = y[1:]
        
        # growth rate
        gm = np.array([g(c,Lm[i]) for i,g in enumerate(gfuns)])
        
        # concentration
        dc = -2*self.physicalProperties['rhoc']/self.modelSetup['Vsol']*np.sum(self.modelSetup['Nm']*np.arange(1,self.modelSetup['m']+1)*gm*Lm)
        
        # growth
        dLm = gm
        
        # return output
        return np.append(dc,dLm)    

    
    #returns an array with the incremental changes to y: y' = f(y,t,.....)
    def __populationEvolution(self,Z,t,M,N,L,contIntSpacing,contSpacing,detAttFuns,detAttFunsWide):

        # unpack constants (inefficient to do this at every point in time!?)
        sigma, kappa, h, V0, cstar, kB = [self.physicalProperties[prop] for prop in ['sigma', 'kappa', 'h', 'V0', 'cstar', 'kB']]
        T = self.modelSetup['T']        

        dividel = len(Z[2:])/2
        Zmspart = np.split(Z[2:dividel+2],self.modelSetup['m'])

        Zms = [np.append(Z[1],Zm) for Zm in Zmspart] # the first two elements could be anything as it isn't really used (code uses Y[0] instead)

        # calculate attachment and detachment rates for all m and L
        Ims = []
        Dms = []
        for i in range(self.modelSetup['m']):
            Dm, Im = zip(*detAttFuns[i](Z[1],L*h)) # converts the list of tuples into two tuples
            Dms = np.append(Dms,np.array(Dm))
            Ims = np.append(Ims,np.array(Im))

        ImL = []
        DmL = []
        for i in range(self.modelSetup['m']):
            Dm, Im = zip(*detAttFunsWide[i](Z[1],L*h))
            DmL = np.append(DmL,np.array(Dm))
            ImL = np.append(ImL,np.array(Im))

        # returns (dZdt,dFdt)
        dZmdtAll = popRates(Ims,Dms,ImL,DmL,contSpacing,contIntSpacing,M,N,self.modelSetup['m'],L,Zms,
                                sigma, kappa, h, V0, cstar, kB)
        
        # translate solution into a single vector containing dZdt for all m
        dZ00dt = - self.params[1] * Z[0]
        dZ0dt = [ dZ[0] for dZ in dZmdtAll[0] ]     
        dZmdt = [ dZ[1:] for dZ in dZmdtAll[0] ]
        dFmdt = [ dF[1:] for dF in dZmdtAll[1] ]
        
        dZdt = np.array([])
        for i in range(self.modelSetup['m']):
            dZdt = np.append(dZdt,dZmdt[i])
        for i in range(self.modelSetup['m']):
            dZdt = np.append(dZdt,dFmdt[i])

        return np.append([dZ00dt,np.sum(dZ0dt)-dZ00dt],dZdt)

        
    def __oldModel(self,N,t,K):
        """
        Calculate r.h.s. of ODEs for new model. Depending on setup models a first or second order process
        """
        
        if self.modelSetup['oldModel']==1:
            # use first-order model
            dN = np.dot(K,N)
            
        elif self.modelSetup['oldModel']==2:
            # use second-order model
            dN = [K[0,0]*N[0]**2+np.sum([K[0,i]*np.sqrt(N[i]) for i in range(1,self.modelSetup['m']+1)])] + \
                 [K[i,0]*N[0]**2+K[i,i]*np.sqrt(N[i]) for i in range(1,self.modelSetup['m']+1)] #  elements of K already have right sign, that's why all is plus

        # return output
        return np.squeeze(dN)
    
    
    def __bm(self,c,m):
        """ 
        Helper function to calculate barrier. Unpacks all variables and then calls static barrier function
        """
        
        # unpack physical properties
        sigma, kappa, h, V0, cstar, kB = [self.physicalProperties[prop] for prop in ['sigma', 'kappa', 'h', 'V0', 'cstar', 'kB']]
        
        T = self.modelSetup['T']
        
        return self.calcBarrier(c,cstar,m,sigma,kappa,h,V0,kB,T)
            
    
    @staticmethod
    def calcBarrier(c,cstar,m,sigma,kappa,h,V0,kB,T):
        """ 
        Thickness- and concentration-dependent barrier. This is a static method so that it can be called without problems also
        from outside (e.g. plotting functions).
        """
        
        # c/cstar <= 1 -> Delta mu <= 0.0 -> The nucleation barrier is Inf ("system is undersaturated")
        if c > cstar:
            # Evaluate Delta mu from concentration
            Dmu = kB*T*np.log(c/cstar)
    
            #critical area from setting wide and narrow facet regime equal
            critA = m**2/2./sigma**2 * (kappa+2*h*sigma) * (kappa+sigma*h-np.sqrt(kappa*(kappa+2*h*sigma)))

            if (critA < ((kappa+2*h*sigma)/(Dmu/V0*h))**2):
                bW = -h*(Dmu/V0)*critA + (4*h*sigma+2*kappa)*np.sqrt(critA) # narrow facet DeltaG
                return bW
            else:
                return  (kappa+2*sigma*h)**2/(Dmu/V0*h) # wide facet barrier
        else:
            return np.inf


    # quick switch between standard scenarios
    def scenarios(self,sce='default'):

        # unpack variables
        sigma, kappa, h, V0, cstar, kB = [self.physicalProperties[prop] for prop in ['sigma', 'kappa', 'h', 'V0', 'cstar', 'kB']]
        T = self.modelSetup['T']

        # Critical side length
        Lcrit = lambda m,c: max(m*h, 2*V0*(m*h*sigma+2*kappa)/( m*h*kB*T*np.log(c/cstar) - 2*sigma*V0 ) )
        # Formation energy
        deltaG = lambda m,L,c: -kB*T*np.log(c/cstar)/V0*m*h*L**2 + sigma*( 2*L**2 + 4*m*h*L ) #+ kappa*( 4*m*h + 8*L )
        # Cube formation energy
        dGcube = lambda m,c: -kB*T*np.log(c/cstar)/V0*(m*h)**3 + sigma*6*(m*h)**2 # + kappa*12*m*h
        # Specific large platelet formation energy
        dGNPL = lambda m,c: -kB*T*np.log(c/cstar) + 2*V0*sigma/(h*m)
        lengths = [h*m for m in np.arange(6)+1]

        if sce == 'tests':
            print 'dGNPL',[ np.exp(-dGNPL(m,10.)/(kB*T)) for m in np.arange(6)+1 ]
            print 'dGcube',[ np.exp(-dGcube(m,10.)/(kB*T)) for m in np.arange(6)+1 ]
            print 'dGmax', [ np.exp(-deltaG(m,Lcrit(m,10.),10.)/(kB*T)) for m in np.arange(6)+1 ]
        
        if sce == 'default':
            #evaluate distribution
            distr = [ np.exp(-dGNPL(m,10.)/(kB*T)) for m in np.arange(6)+1 ]
            distr = distr / np.sum(distr)
            #print distr
            self.modelSetup['t'] = np.append(0,np.logspace(-4,4,100))
            self.modelSetup['y0'] = np.append(10.,lengths)
            self.modelSetup['Nm'] = distr
            self.modelSetup['oldModel'] = 0
            self.modelSetup['twoStep'] = 0
            self.modelSetup['tSplit'] = 0.0
        elif sce == 'initial3':
            #evaluate distribution
            distr = [ np.exp(-dGNPL(m,1.)/(kB*T)) for m in np.arange(6)+1 ]
            distr = distr / np.sum(distr)
            self.modelSetup['t'] = np.append(0,np.logspace(-4,4,100))
            self.modelSetup['y0'] = np.append(1.,lengths)
            self.modelSetup['y0'][3] = 1000.
            self.modelSetup['Nm'] = distr
            self.modelSetup['oldModel'] = 0
            self.modelSetup['twoStep'] = 0
            self.modelSetup['tSplit'] = 0.0
        elif sce == 'tSplit':
            distr = [ np.exp(-dGNPL(m,10)/(kB*T)) for m in np.arange(6)+1 ]
            distr = distr / np.sum(distr)
            self.modelSetup['t'] = np.append(0,np.logspace(-5,3,100))
            self.modelSetup['y0'] = np.append(10.,lengths)
            self.modelSetup['Nm'] = distr
            self.modelSetup['oldModel'] = 0
            self.modelSetup['twoStep'] = 1
            self.modelSetup['tSplit'] = 1E-3
        elif sce == 'oldModel':
            self.modelSetup['t'] = np.append(0,np.logspace(-4,4,100))
            self.modelSetup['y0'] = np.append(10.,np.zeros(6))
            self.modelSetup['oldModel'] = 1
            self.modelSetup['twoStep'] = 0
            self.modelSetup['tSplit'] = 0.0
#        elif sce == 'BoltzmanSeedDist':
            
