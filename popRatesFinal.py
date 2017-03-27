# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 18:29:33 2016

@author: Dave, Flo
"""
import numpy as np
import pdb

def popRatesStar(args):
    return popRates(*args) # unpack that thing (*args) = (arg1,arg2,...)?


#contIntSpacing is \Delta n_{m,m+1}, and contSpacing is \Delta n_{n-1,n+1}
#Im/Dm: attachement/detachment; M: Number of bins in the cont. part; N: Maximum side length (in layers) of discrete part
#Y: Cubic seed distr.; Zm: NPL concentrations; m: NPL thickness; L: side length of each size
#rest are material params
def popRates(Ims,Dms,ILs,DLs,contSpacing,contIntSpacing,M,N,mmax,L,Zf,sigma, kappa, h, V0, cstar, kB):

    #splitting one could do it with dicts and loops, but no - we need another layer to account for the removal of material by wide facet growth
    ln = len(L)
    Zf.append(np.zeros(ln)) # additional thickness with conc = 0
    Z = np.array(Zf)
    Im = np.split(Ims,mmax) # split rates for each thickness
    Im.append(np.zeros(ln)) # attach another set of lat. growth rates -> smae procedure for Dm, IL, and DL
    Dm = np.split(Dms,mmax)
    Dm.append(np.zeros(ln))
    IL = np.split(ILs,mmax)
    IL.append(np.zeros(ln))
    DL = np.split(DLs,mmax)
    DL.append(np.zeros(ln))
    
    #create the dZdt array
    dZdt = np.zeros(Z.shape)
    dFdt = np.zeros(Z.shape)

    #first treat all the cases from m=2 to m=mmax-1
    for m,Zi,Zp,Zm,Ii,Di,Il,Ilm,Dl,Dlp in zip(np.arange(2,mmax+1),Z[1:-1],Z[2:],Z[:-2],Im[1:-1],Dm[1:-1],IL[1:-1],IL[:-2],DL[1:-1],DL[2:]): 

        m = int(m)

        Zd = Zi[0:N] #Discrete
        Zdm = Zm[0:N]
        Zdp = Zp[0:N]

        Zt = Zi[N:]  #continuous
        Ztm = Zm[N:]
        Ztp = Zp[N:]
    
        Gt =  Ii[N:] - Di[N:]
        Ht =  Ii[N:] + Di[N:]       
    
        # CC70 coefficients
        # A at intermediate points, notation i = i+1/2
        # Vetter2013: A_{m+1/2}
        Aint = -1./2. * (Gt[1:M]+Gt[0:M-1]) + 1./2. * (Ht[1:M]-Ht[0:M-1]) / contIntSpacing[0:M-1]
    
        # B at intermeidate points, notation i = i+1/2
        # Vetter2013: B_{m+1/2}
        Bint = 1./4. * (Ht[1:M] + Ht[0:M-1]) 
        
        # w,W+,W- at intermediate points, notation i = i+1/2
        w = Aint / Bint * contIntSpacing[0:M-1]

        #Wplus = np.array([ 1./(1.-wi/2.) if abs(wi) < 1E-12 else wi / (1.-np.exp(-wi)) for wi in w ])
        #Wminus = np.array([ 1./(1.+wi/2.) if abs(wi) < 1E-12 else -wi / (1.-np.exp(wi)) for wi in w])
        Wplus = w / (1.-np.exp(-w))
        Wminus = -w / (1.-np.exp(w))
        
        # CC70 coefficients
        a = 1./contSpacing[1:M-1] * Bint[0:M-2] / contIntSpacing[0:M-2] * Wminus[0:M-2]
        # a = np.insert(a,0,0.) #this inserts a 0 at the first position -> indexing must not be adjusted!
        
        b = 1./contSpacing[1:M-1] * ( Bint[1:M-1] / contIntSpacing[1:M-1] * Wminus[1:M-1] \
                                          +  Bint[0:M-2] / contIntSpacing[0:M-2] * Wplus[0:M-2] )
        # b = np.insert(b,0,0.)
        
        c = 1./contSpacing[1:M-1] * Bint[1:M-1] / contIntSpacing[1:M-1] * Wplus[1:M-1]
        # c = np.insert(c,0,0.)
        
        dZd_dt = np.zeros(N)        
        dZt_dt = np.zeros(M)   
        dFd_dt = np.zeros(N)
        dFt_dt = np.zeros(M)
        
        # part with discrete number of layers
        dZd_dt[1:m-1] = 0. # crystals for which m>L don't exist
        dZd_dt[m-1] = Zdm[m-1]*Ilm[m-1] - (Ii[m-1] + Dl[m-1])*Zd[m-1] + Di[m]*Zd[m] #formation of the cubes; they "cannot" grow in thickness
        dZd_dt[m:N-1] = Ii[m-1:N-2]*Zd[m-1:N-2] + Di[m+1:N]*Zd[m+1:N] \
            + Ilm[m:N-1]*Zdm[m:N-1] + Dlp[m:N-1]*Zdp[m:N-1] \
            - ( Ii[m:N-1]+Di[m:N-1]+Il[m:N-1]+Dl[m:N-1] )*Zd[m:N-1] #meaning that the last discrete element here is Zd[N-2]
        #flux
        dFd_dt[m:] = - Dlp[m:N]*Zdp[m:] + Il[m:N]*Zd[m:]

        # transition
        dZd_dt[N-1] = Ii[N-2]*Zd[N-2] + Di[N]*Zt[0]\
            + Ilm[N-1]*Zdm[N-1] + Dlp[N-1]*Zdp[N-1] \
            - (Ii[N-1] + Di[N-1] + Il[N-1] + Dl[N-1])*Zd[N-1] #Element taken from the cont. part  Lt[0] is exactly 1, so the integration should be fine 
        
        dZt_dt[0] = 1./contSpacing[0]*Bint[0] / contIntSpacing[0]*(Wplus[0]*Zt[1] - Wminus[0]*Zt[0]) \
            + contIntSpacing[0]*(Ilm[N]*Ztm[0] + Dlp[N]*Ztp[0] - (Il[N] + Dl[N])*Zt[0]) \
            + (Ii[N-1]*Zd[N-1] - Di[N]*Zt[0]) #discrete part
        
        # continuous part
        # dZt_dt[1:M-2] = a[1:M-2] * Zt[0:M-3] - b[1:M-2] * Zt[1:M-2] + c[1:M-2] * Zt[2:M-1] #due to adjusted a, b, c
        dZt_dt[1:M-1] = a[0:M-2]*Zt[0:M-2] - b[0:M-2]*Zt[1:M-1] + c[0:M-2]*Zt[2:M] \
            + contIntSpacing[1:M-1]*(Ilm[N+1:-1]*Ztm[1:M-1]+Dlp[N+1:-1]*Ztp[1:M-1] - (Il[N+1:-1] + Dl[N+1:-1])*Zt[1:M-1])
        
        #flux
        dFt_dt = contIntSpacing*( - Dlp[N:]*Ztp + Il[N:]*Zt)

        # out flux condition
        dZt_dt[M-1] = 0.0
        # no flux condition
        #dZt_dt[M-1] = 1/contSpacing[M-1] * Bint[M-2]/contIntSpacing[M-2] * ( Wminus[M-2]*Zt[M-2] - Wplus[M-2]*Zt[M-1] ) \
        #    + contIntSpacing[M-1] * (Ilm[-1]*Ztm[M-1] + Dlp[-1]*Ztp[M-1] - (Il[-1] + Dl[-1]) * Zt[M-1])

        dZd_dt[0] = -  np.sum( dZd_dt[1:]*np.ceil(L[1:N]**2 * m/2.) ) \
            - np.sum( dZt_dt*np.ceil((L[N:] + contIntSpacing/2.)**2 * m/2. )*contIntSpacing)

        dZdt[m-1]= np.append(dZd_dt,dZt_dt)
        dFdt[m-1]= np.append(dFd_dt,dFt_dt)

    # include the material loss from the last element Variables -> everything that reaches m > mmax is considered as loss
    dLoss = np.sum(Il[m:N-1]*Zd[m:N-1] * np.ceil( L[m:N-1]**2 * (m+1.)/2. )) \
        + np.sum(contIntSpacing * Il[N:] * Zt* np.ceil( (L[N:]+contIntSpacing/2.)**2 * (m+1.)/2. ) )

    #if dLoss > 1E-2:
    #    print dLoss, Il[m:m+3],Il[N:N+3]
    
    dZdt[m-1][0] = dZdt[m-1][0] - dLoss

    # m = 1 must be treated separately because it is the thinnest population
    m = 1

    #Zdm/Ztm do not exist
    Zd = Z[0][:N] #Discrete
    Zdp = Z[1][:N]
    
    Zt = Z[0][N:]  #continuous
    Ztp = Z[1][N:]
    
    Ii = Im[0]
    Di = Dm[0]

    Gt =  Im[0][N:] - Dm[0][N:]
    Ht =  Im[0][N:] + Dm[0][N:]      
    
    # CC70 coefficients
    # A at intermediate points, notation i = i+1/2
    # Vetter2013: A_{m+1/2}
    Aint = -1./2. * (Gt[1:M]+Gt[0:M-1]) + 1./2. * (Ht[1:M]-Ht[0:M-1]) / contIntSpacing[0:M-1]
    
    # B at intermeidate points, notation i = i+1/2
    # Vetter2013: B_{m+1/2}
    Bint = 1./4. * (Ht[1:M] + Ht[0:M-1]) 
    
    # w,W+,W- at intermediate points, notation i = i+1/2
    w = Aint / Bint * contIntSpacing[0:M-1]
    
    #Wplus = np.array([ 1./(1.-wi/2.) if abs(wi) < 1E-12 else wi / (1.-np.exp(-wi)) for wi in w ])
    #Wminus = np.array([ 1./(1.+wi/2.) if abs(wi) < 1E-12 else -wi / (1.-np.exp(wi)) for wi in w])
    Wplus = w / (1.-np.exp(-w))
    Wminus = -w / (1.-np.exp(w))

    # CC70 coefficients
    a = 1./contSpacing[1:M-1] * Bint[0:M-2] / contIntSpacing[0:M-2] * Wminus[0:M-2]
    # a = np.insert(a,0,0.) #this inserts a 0 at the first position -> indexing must not be adjusted!
    
    b = 1./contSpacing[1:M-1] * ( Bint[1:M-1] / contIntSpacing[1:M-1] * Wminus[1:M-1] \
                                      +  Bint[0:M-2] / contIntSpacing[0:M-2] * Wplus[0:M-2] )
    # b = np.insert(b,0,0.)
    
    c = 1./contSpacing[1:M-1] * Bint[1:M-1] / contIntSpacing[1:M-1] * Wplus[1:M-1]
    # c = np.insert(c,0,0.)
    
    dZd_dt = np.zeros(N)        
    dZt_dt = np.zeros(M)   
    dFd_dt = np.zeros(N)
    dFt_dt = np.zeros(M)

    # part with discrete number of layers
    # dZd_dt[0] will be determined differently because it is the monomer conc.
    #smallest platelets grow from monomers
    dZd_dt[1] = Ii[0]*Zd[0] + Di[2]*Zd[2] \
        - (Ii[1]+Di[1]+Il[1]) * Zd[1] # debugging: multiply by Zd[0]
    # larger ones just normal
    dZd_dt[2:N-1] = Ii[1:N-2] * Zd[1:N-2] + Di[3:N] * Zd[3:N] \
        +  Dlp[2:N-1]*Zdp[2:N-1] \
        - (Ii[2:N-1]+Di[2:N-1]+Il[2:N-1]) * Zd[2:N-1] 

    # flux
    dFd_dt[m:] = - Dlp[m:N]*Zdp[m:] + Il[m:N]*Zd[m:]

    # transition
    dZd_dt[N-1] = Ii[N-2] * Zd[N-2] + Di[N] * Zt[0]\
        +  Dlp[N-1]*Zdp[N-1] \
        - (Ii[N-1] + Di[N-1] + Il[N-1]) * Zd[N-1]
    
    dZt_dt[0] = 1./contSpacing[0] * Bint[0] / contIntSpacing[0] * (Wplus[0] * Zt[1] - Wminus[0] * Zt[0]) \
            + contIntSpacing[0] * (Dlp[N]*Ztp[0] - Il[N]*Zt[0]) \
            + (Ii[N-1] * Zd[N-1] - Di[N] * Zt[0])

    # continuous part
    dZt_dt[1:M-1] = a[0:M-2] * Zt[0:M-2] - b[0:M-2] * Zt[1:M-1] + c[0:M-2] * Zt[2:M] \
        + contIntSpacing[1:M-1] * (Dlp[N+1:-1]*Ztp[1:M-1] - Il[N+1:-1]*Zt[1:M-1])

    #flux
    dFt_dt = contIntSpacing * ( - Dlp[N:]*Ztp + Il[N:]*Zt)
    
    # out flux condition
    dZt_dt[M-1] = 0.0
    # no flux condition
    # dZt_dt[M-1] = 1/contSpacing[M-1] * Bint[M-2]/contIntSpacing[M-2] * ( Wminus[M-2]*Zt[M-2] - Wplus[M-2]*Zt[M-1] )
    #dZt_dt[M-1] = 1/contSpacing[M-1] * Bint[M-2]/contIntSpacing[M-2] * ( Wminus[M-2]*Zt[M-2] - Wplus[M-2]*Zt[M-1] ) \
    #        + contIntSpacing[M-1] * (Dlp[-1]*Ztp[M-1] - Il[-1]*Zt[M-1])
    
    dZd_dt[0] = -  np.sum (dZd_dt[1:] * np.ceil( L[1:N]**2 * m/2. )) \
        - np.sum(dZt_dt * np.ceil((L[N:] + contIntSpacing/2.)**2 * m/2.) * contIntSpacing)
    
    dZdt[0] = np.append(dZd_dt,dZt_dt)
    dFdt[0]= np.append(dFd_dt,dFt_dt)

    return (dZdt,dFdt) #dZdt

