#creates plots from the simulations

import sys
import operator
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

sns.set(font_scale = 0.9, style = 'white')
sns.set_style("ticks", {"xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 0.0, "ytick.minor.size": 0.0, 'xtick.direction': u'in', 'ytick.direction': u'in'})

h = 3.1
tmin = 1E-4
tmax = 1E15

color5 = ['steelblue','forestgreen','crimson','mediumpurple','darkorange']

def main():
    savestring = sys.argv[1]
    normalPlots(savestring)
    #contourPlots(savestring)

def normalPlots(savestr='data.npy'):
    
    print savestr,' 1D Plots'

    data = np.load(savestr).item()
    
    # evaluate xticks
    nt = 3 # number of ticks
    xt = [10** (i*np.floor(np.log10(tmax/tmin)/nt))  * tmin for i in range(int(np.floor(np.log10(tmax/tmin)))) if 10** (i*np.floor(np.log10(tmax/tmin)/nt)) <= tmax/tmin ]
    print xt
    m = len(data['Vtot'])-2

    # Sizes in cm
    inch=2.54/1.

    # plotting
    fig,(ax0,ax1) = plt.subplots(2,1,figsize=(6./inch,10./inch), sharex = True) #for some strange reason this figsize does not work
    fig.set_size_inches([6./inch,10./inch])
    #print 6./inch,fig.get_size_inches()

    # compute min/max indices of the time plots
    imin = np.argmin(np.abs(data['t']-tmin*10.))
    imax = np.argmin(np.abs(data['t']-tmax/10.))
    print imin, imax

    # compute total loss
    loss = 1-np.sum(data['Vtot'],axis=0)
    print 'loss',loss[imax]

    #set yticks
    yt0 = np.arange(0,1.3,0.4)
    #yt0 = [0.0 , 0.4 , 0.8 , 1.2 ]

    #plot1

    #ax0.set_title('Monomer Concentration &  Material Loss')

    # platelet yield
    for i,y in enumerate(data['Vtot'][2:7]):
        ax0.plot(data['t'],y,color5[i],lw=1.0)
    #ax0.plot(data['t'],np.transpose(data['Vtot'][2:7]),lw=1.0)
    ax0.set_xlim([tmin,tmax])
    ax0.set_ylim([0,1.2])
    ax0.set_xscale('log')#,linthreshx = Z.modelSetup['t'][1])                                                                                          
    #ax0.set_xticks(xt)
    ax0.set_ylabel(r'$\mathrm{m}_i / \mathrm{c}_0$')
    ax0.set_xticks(xt)
    ax0.set_yticks(yt0)
    # label lines
    #for i in range(m):
    #for i in range(m-2):
    #    tlb,ylb = max(zip(data['t'][imin:imax],data['Vtot'][i+2][imin:imax]),key = operator.itemgetter(1)) # finds the coordinates of the functions maximum                                                               
        #if ylb > 0.0:
        #    #plt.text(tlb,ylb,'m = {}'.format(i+1), color = plt.gca().lines[-1-(m-1-i)].get_color(),
        #    ax0.text(tlb,ylb+0.05,'m = {}'.format(i+1), color = ax0.lines[-1-(m-3-i)].get_color(),fontsize=7.5,
        #             horizontalalignment = 'center', verticalalignment = 'bottom')
    
    #conc.
    ax0.plot(data['t'],data['c'],lw=1.5, color = 'k')
    #tlb,ylb = max(zip(data['t'][imin:imax],data['c'][imin:imax]),key = operator.itemgetter(1))
    #ax0.text(tlb/5.,ylb+0.05,'monomers', color = ax0.lines[-1].get_color(),fontsize=7.5,
    #         horizontalalignment = 'left', verticalalignment = 'bottom')

    
    ax0.text(1E-10,ax0.get_ylim()[1],'a',fontsize=9,weight='bold',
             horizontalalignment = 'right', verticalalignment = 'top')

    #correct parts of L_m with extremely small V_tot
    for i,V in enumerate(data['Vtot'][2:]):
        for j,Vi in enumerate(V):
            if Vi < 1E-6:
                data['Lm'][i][j] = (i+1.)*h
    # print length
    #for i,y in enumerate(data['Lm'][:-2]):
    #    ax1.plot(data['t'],y,color5[i],lw=1.0)

    # conc
    dL = data['pop_L'][1:]-data['pop_L'][:-1]
    conc = [ [] for i in range(m) ]
    for y in data['pop_y']:
        for i,yi in enumerate(np.split(y[2:],m)):
            conc[i] = np.append(conc[i],np.sum(yi*dL))
            #conc[i] = np.append(conc[i],yi[i]) 
    # print conc
    for i,y in enumerate(conc[:-1]):
        ax1.plot(data['t'],y,color5[i],lw=1.0)

    #ax1.plot(data['t'],np.transpose(data['Lm'][:-2]),lw=1.0)
    #ax1.set_ylim([0, np.amax( [ dat[imin:imax] for dat in conc[:-2] ] ) ]) # there must be a better way!
    ymax = np.amax( [ dat[imin:imax] for dat in conc[:-2] ] ) 
    ax1.set_ylim([ymax/1E6 , 10*ymax])
    ax1.set_xlim([tmin,tmax])
    ax1.set_xscale('log')#,linthreshx = Z.modelSetup['t'][1])
    ax1.set_xlabel(r'$\tau$')
    #ax1.set_ylabel(r'$\bar L \, \left(\, %d \,\mathrm{\AA} \,\right)$' % max10)
    #ax1.set_ylabel(r'$\bar L \, \left(\mathrm{\AA} \,\right)$')
    ax1.set_ylabel(r'$\mathrm{c}/\mathrm{c}_0$') 
    ax1.set_yscale('log')
    # yticks
    yt1 =  10**np.arange( np.ceil(np.log10(ax1.get_ylim()[0])) , np.ceil(np.log10(ax1.get_ylim()[1])) , 2)
    print ax1.get_ylim(),yt1
    ax1.set_yticks(yt1)
    ax1.set_xticks(xt)

    ax1.text(1E-10,ax1.get_ylim()[1],'b',fontsize=9,weight='bold',
             horizontalalignment = 'right', verticalalignment = 'top')
    
    #for i in range(m-2):
    #    tlb,ylb = max(zip(data['t'][imin:imax],data['Lm'][i][imin:imax]/max10),key = operator.itemgetter(1)) 
    #    #tlb,ylb = max(zip(data['t'][imin:imax],data['Lm'][i][imin:imax]),key = operator.itemgetter(1))
    #    if ylb > 0.0:
    #        ax1.text(tlb,ylb+0.05,'m = {}'.format(i+1), color = ax1.lines[-1-(m-3-i)].get_color(),fontsize=7.5,
    #                 horizontalalignment = 'center', verticalalignment = 'bottom')

    #ax1.text(1E-8,ax1.get_ylim()[1],'a',fontsize=9,weight='bold',
    #         horizontalalignment = 'right', verticalalignment = 'top')
    
    #plt.tight_layout()
    fig.subplots_adjust(left=0.25, bottom=0.15, right=0.95, top=0.98,
                        wspace=0.6, hspace=None)
    
    savestrFig = 'Evolution_'+savestr + '.pdf'
    plt.savefig(savestrFig)
    

def contourPlots(savestr='data.npy'):

    print savestr,' 2D Plots'

    #get data ready
    data = np.load(savestr).item()
    m = len(data['Vtot'])-2
    print m


    #time ticks
    nt = 3 # number of ticks
    xt = [10** (i*np.floor(np.log10(tmax/tmin)/nt))  * tmin for i in range(int(np.floor(np.log10(tmax/tmin)))) if 10** (i*np.floor(np.log10(tmax/tmin)/nt)) <= tmax/tmin ]
    
    # prepare the population arrays for each thickness
    PD=[np.array([]) for i in range(m)]
    for yi in data['pop_y']:
        yy = np.split(yi[2:],m)
        for (j,yj) in enumerate(yy):
            PD[j] = np.append(PD[j],yj)
    
    # X and Y coordinates mesh
    X,Y =np.meshgrid(data['pop_L'][1:],data['t'])
    
    #array of plotable Z-coordinates
    Z = [np.split(PD[i],len(data['t'])) for i in range(len(PD))]

    # evaluate global zmax and zmin
    zmaxi = []
    for i in range(m-1):
        # evaluate the maximum in z floored in log10 space
        zmaxi.append(1E0 * 10**(np.floor(np.log10(np.amax(Z[i])))))
        if np.isnan(zmaxi[i]):
            print 'zmax is NaN'
            raise SystemExit
        
    zmin = 1E-4 * np.amin(zmaxi)
    zmax = 1E-0 * np.amax(zmaxi)

    # cm in inch
    inch=2.54/1.
    #fig = plt.figure(figsize=(10,10))
    fig,axes = plt.subplots(1,m-1,figsize=(17./inch,4.5/inch), sharey=True )
    for i,ax in enumerate(axes):
        #create im array of plots

        # makes nicer plots
        Z[i] = np.clip(Z[i],zmin,10.*zmax)

        #im.append(ax[i].pcolor(X,Y,Z[i], vmin = 0.0, vmax = zmax , cmap=matplotlib.cm.RdBu))
        #im.append(ax[i].pcolor(X,Y,Z[i], norm=LogNorm(vmin=zmin, vmax=zmax) , cmap=matplotlib.cm.RdBu))
        im = ax.pcolor(X,Y,Z[i], norm=LogNorm(vmin=zmin, vmax=zmax) , cmap=matplotlib.cm.YlOrRd)
        if i==0:
            ax.set_ylabel(r'$\tau$')
        ax.set_xlabel(r'$L$ ($\mathrm{\AA}$)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.set_ylim([np.amin(Y),np.amax(Y)])
        ax.set_ylim([tmin,tmax])
        ax.set_yticks(xt)
        #ax[i].set_xlim([ i+2., np.amax(X) ])
        ax.set_xlim([ i+2., 2E3 ]) 
        #ax.set_title(r'$m$ = '+str(i+1)+' ML',y=1.1)
        ax.set_title(r'$m$ = '+str(i+1),y=1.1)
        """
        #create array of scale bars
        #is an object containing information about the coords.?
        div.append(make_axes_locatable(ax[i]))
        #defines the axis where the scale bar should be added; size: how much of im[i] is taken; pad: offset 
        caxi.append( div[i].append_axes("bottom", size="5%", pad=.8))
        #adds the scale bar
        cbar.append( plt.colorbar(im[i], cax = caxi[i],orientation = 'horizontal',ticks= [zmin , zmax ]))
        #cbar[i].ax.set_xticklabels([0.0,r'10$^{%i}$' % np.log10(zmax) ])
        cbar[i].ax.set_xticklabels([r'<10$^{%i}$' % np.log10(zmin) ,r'10$^{%i}$' % np.log10(zmax) ])
        """        
        
    # make single scale bar
    fig.subplots_adjust(left=0.08, bottom=0.3, right=0.87, top=0.8,
                        wspace=0.18, hspace=None) #adds space to the figure for the scale bar
    cbar_ax = fig.add_axes([0.89, 0.3, 0.01, 0.5]) #specifies the scale bar axes
    cbar = fig.colorbar(im, cax = cbar_ax, ticks= [zmin , zmax ])
    cbar.ax.set_xticklabels([r'$ < 10^{%i}$' % np.log10(zmin) ,r'$> 10^{%i}$' % np.log10(zmax) ])
    cbar.set_label(r'$\mathrm{c}/\mathrm{c}_0$')

    #plt.tight_layout()

    savestrFig = 'Population_'+savestr + '.png'
    plt.savefig(savestrFig, dpi = 600)
    #plt.show()

if __name__ == "__main__": main()

