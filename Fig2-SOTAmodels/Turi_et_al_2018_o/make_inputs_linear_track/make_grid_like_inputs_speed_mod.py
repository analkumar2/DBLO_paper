#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:08:09 2018

@author: spiros
"""

import os, sys
import numpy as np
from gridfield import gridfield
#import matplotlib.pyplot as plt
#import matplotlib as mpl

#from matplotlib.colors import LinearSegmentedColormap
#
## Adopted from https://github.com/BIDS/colormap/blob/master/parula.py
#cm_data = np.loadtxt('parula_like_colormap.txt') 
#parula_map = LinearSegmentedColormap.from_list('parula', cm_data)




# Choose coordinates +1 for the value
# e.g., if you want 100-->myx = 101
# Apart from the case that one dimension is one


###############################################################################
####################### P A T H   C O N S T R U C T I O N  ####################
###############################################################################

myx = 201 # Choose +1 the points you want (apart from 1)
myy = 1

# Place field coordinations; all to all combinations
x_array = range(0,myx,5)
#x_array = [100]
y_array = [1]

my_run=int(sys.argv[1])

print my_run

maindir1 = 'runs_produced_by_python_speed_pos/'
maindir2 = 'runs_produced_by_python_speed_neg/'

dirname1 = maindir1+'run_'+str(my_run)
dirname2 = maindir2+'run_'+str(my_run)

os.system('mkdir -p ' + dirname1)
os.system('mkdir -p ' + dirname2)

np.random.seed(my_run)


nx  = 1
ny  = 1

p0  = [nx, ny]
k   = 1

zold = 0

totlength =0
if myy == 1:
    npoints = (myx-1)*myy
else:
    npoints = (myx-1)*(myy-1)


path = []
for i in xrange(1, npoints+1):
    if (ny > myy):
        break
    
    if ((i >= 70) and (i <=95)) or ((i >= 105) and (i <=130)): 
        # random time at each point
        print 'Enters/Leaves reward zone'
        mu = 140-np.abs(100-i)
        sigma = 2
        # random time at each point
        z = np.round(mu + np.random.randn(1).item()*sigma)      
    elif (i > 95) and (i < 105):
        print 'In reward zone'
        mu = 500-np.abs(100-i)
        sigma = 5
        # random time at each point
        z = np.round(mu + np.random.randn(1).item()*sigma)
    else:
        # random time at each point
        mu    = 70
        sigma = 2
        # random time at each point
        z = np.round(mu + np.random.randn(1).item()*sigma)
        while (z<mu-20) or (z>mu+20): 
            z = np.round(mu + np.random.randn(1).item()*sigma)

    
    time_at_the_same_point=z
    path += [p0]*int(time_at_the_same_point)
    
    nx = nx + k
    p0 = [nx, ny]
    
    if (nx > myx-1) and (ny % 2) != 0:
        ny=ny+1
        nx=myx-1
        p0 = [nx, ny]
        k=-1

    
    if (nx < 1) and (ny % 2)==0:
        ny=ny+1
        nx=1
        p0 = [nx, ny]
        k=1
        
    zold += time_at_the_same_point

# save the path
path = np.array(path)
filename = dirname1+'/path.txt'
np.savetxt(filename, path, fmt='%.0f', delimiter=' ')

print 'Done with the path'

###############################################################################
####################### G R I D    L I K E    I N P U T S  ####################
###############################################################################



ndend       = 8 # Number of dendrites 
theta_freq  = 8 # in Hz
theta_phase = 0
my_field    = 0


for xxx in x_array:
    for yyy in y_array:

        my_field += 1
                
        folder1 = dirname1+'/place_field_'+str(my_field)
        folder2 = dirname2+'/place_field_'+str(my_field)
        os.system('mkdir -p '+str(folder1))
        os.system('mkdir -p '+str(folder2))
        
        d  = np.zeros((ndend, myx ,myy)) # d is the x,y point of the grid field of dend ni
        dd = np.zeros((myx, myy))
        

        angle = 0.0
        lambda_var = 3.0       
        for ni in xrange(ndend):
            lambda_var += 0.5
            angle += 0.4
            for x in xrange(myx):
                for y in xrange(myy):  #to d einai to shmeio x,y tou grid field tou dend ni
                    d[ni, x, y] = gridfield(angle, lambda_var, xxx, yyy, x, y)

        for ni in xrange(ndend):
            dd += d[ni,:,:]

        
##        cmap = mpl.cm.plasma
#        cmap = parula_map
#        
#        fig, axes = plt.subplots(nrows=4, ncols=2, dpi=150)
#        ni = 0
#        for ax in axes.flat:
#            im = ax.imshow(d[ni,:,:].T, origin='lower',cmap=cmap, aspect='50', vmin=0, vmax=1)
#            ax.tick_params( axis='y',which='both',right='off',left='off',labelleft='off')
#            ax.set_xlabel('Position (cm)')
#            ax.set_xticks(range(0, 201, 50))
#            ax.set_xticklabels([str(x) for x in range(0, 201,50)])
##            ax.set_title('Grid-like Input '+str(ni+1))
#            ni += 1
#            
##        cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
#        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
##        cbar = fig.colorbar(im, cax=cb_ax)
#        cbar.ax.set_ylabel('Normalised firing rate [Hz]')
#        cbar.set_ticks([x/10.0 for x in range(0,11,2)])
#        cbar.set_ticklabels([str(x/10.0) for x in range(0,11,2)])
#
##        plt.tight_layout()
#        plt.savefig('TheoreticalGridCell.eps', format='eps', dpi=1200)
#        plt.savefig('TheoreticalGridCell.png', format='png', dpi=1200)
#        
#        
#        
#        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
#        im = plt.imshow(dd.T/np.max(dd), origin='lower',cmap=cmap, aspect='50', vmin=0, vmax=1)
#        ax.tick_params( axis='y',which='both',right='off',left='off',labelleft='off')
#        ax.set_xlabel('Position (cm)')
#        ax.set_xticks(range(0, 201, 50))
#        ax.set_xticklabels([str(x) for x in range(0, 201, 25)])
#        ax.set_title('Place-like Cell')
##        cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
#        cbar = plt.colorbar(im)
#        cbar.ax.set_ylabel('Normalised firing rate [Hz]')    
#        plt.savefig('TheoreticalPlaceCell.eps', format='eps', dpi=1200)
#        plt.savefig('TheoreticalPlaceCell.png', format='png', dpi=1200)
#        
        
        
        
        for ni in xrange(ndend):
            spikes1 = []
            spikes2 = []         
            for i in xrange(len(path)): # i represent the time in ms
                current_loc = path[i, :]-1
                

                prob = d[ni, current_loc[0], current_loc[1]]
                prob *= (np.sin(2.0*np.pi*theta_freq*i/1000.0 + theta_phase)+1.0)/2.0
                
                r_cnt  = 100 # reward center location
                lag1   = 5 # 2 seconds in bins
                
                # Positive modulation filter
                pp     = 1 - 0.25*np.exp(-0.05*(current_loc[0] - (r_cnt+lag1) )**2)

                # negative modulation filter
                pn     = 1 + 0.25*np.exp(-0.05*(current_loc[0] -r_cnt-1 )**2)
                
                p_pos  = prob * pp
                p_neg  = prob * pn


                rnd_num = np.random.rand(1).item()

                if (p_pos > 0.7) and (rnd_num < prob / 2.0):
                    
                    # spikes is a vector with the locations/timespots where there is a spike
                    spikes1.append(i)  

                if (p_neg > 0.7) and (rnd_num < prob / 2.0):
                    
                    # spikes is a vector with the locations/timespots where there is a spike
                    spikes2.append(i)  

            
            spikes1 = np.array(spikes1).reshape(-1,1)
            spikes2 = np.array(spikes2).reshape(-1,1)
            filename1 = folder1+'/s'+str(ni)+'.txt'
            filename2 = folder2+'/s'+str(ni)+'.txt'
            np.savetxt(filename1, spikes1, fmt='%.0d', delimiter=' ')
            np.savetxt(filename2, spikes2, fmt='%.0d', delimiter=' ')
       
        print 'Done with Grid field ' +str(my_field)





