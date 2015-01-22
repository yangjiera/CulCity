from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from pylab import *
from scipy.stats.distributions import norm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sys

from scipy.misc import imread
import matplotlib.cbook as cbook
#from mpl_toolkits.basemap import Basemap


def data_visulize_temporal_scatter(boundingbox, u_category, data, plotNo, plotIndex):
    datetimes = [x[0] for x in data]
    plt.subplot(1, plotNo, plotIndex, title = u_category)
    
    plt.ylim( (0, 24) )
    #datetimes = dates.date2num(datetimes)
    #plt.plot_date(datetimes, Y, alpha=0.015)
    plt.plot([dt.date() for dt in datetimes],[dt.hour + dt.minute/60. for dt in datetimes],',')
    return 0

def data_visulize_temporal_kde(boundingbox, u_category, data, plotNo, plotIndex):
    datetimes = [x[0] for x in data]
    datatimes_format = np.array([dt.hour + dt.minute/60. for dt in datetimes])
    #print datatimes_format[0:10]
    
    plt.subplot(1, plotNo, plotIndex, title = u_category)
    
    x_grid = np.linspace(0, 24, 5000)
    '''grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.01, 0.5, 20)},
                    cv=20) # 20-fold cross-validation
    grid.fit(datatimes_format[:, None])
    print grid.best_params_
    
    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(x_grid[:, None]))'''
    kde = KernelDensity(kernel='gaussian', bandwidth=0.12).fit(datatimes_format[:, np.newaxis])
    pdf = np.exp(kde.score_samples(x_grid[:, np.newaxis]))
    plt.xlim( (0, 24) )
    #plt.ylim( (0, ) )
    s_pdf = sum(pdf)
    plt.plot(x_grid,pdf)
    #plt.hist(datatimes_format, bins=12)
    return 0

def data_visulize_geo_scatter(boundingbox, u_category, data, plotNo, plotIndex):
    X = np.array([x[0] for x in data])
    Y = np.array([x[1] for x in data])
    del data
    # set the figure axes
    plt.subplot(1, plotNo, plotIndex, title = str(u_category)+': '+str(len(X)))
    plt.xlim( (boundingbox[2], boundingbox[3]) )
    plt.ylim( (boundingbox[0], boundingbox[1]) )
    plt.scatter(X, Y, s=1, alpha=0.25, zorder=1)

    datafile = cbook.get_sample_data("/Users/jyang3/Projects/www'websci/Code/vlz/back_am.jpg")
    img = imread(datafile)
    plt.imshow(img, zorder=0, extent=[boundingbox[2], boundingbox[3], boundingbox[0], boundingbox[1]])

    return 0

def data_visulize_geo_kde(boundingbox, u_category, data, plotNo, plotIndex):
    data2fit = np.array([[x[0], x[1]] for x in data])
    del data
    #print data2fit[0:10]
    #print boundingbox
    
    # set the figure axes
    plt.subplot(1, plotNo, plotIndex, title = str(u_category)+': '+str(len(data2fit)))
    plt.xlim( (boundingbox[2], boundingbox[3]) )
    plt.ylim( (boundingbox[0], boundingbox[1]) )
    #ax = fig.add_subplot(1, 2, 1, projection='3d')
    #ax.set_xlim(boundingbox[2], boundingbox[3]);
    #ax.set_ylim(boundingbox[0], boundingbox[1]);
    
    # cv search for the best bandwidth for KDE
    '''grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=1E-4),
                    {'bandwidth': np.linspace(0.001, 0.02, 20)},
                    cv=20) # 20-fold cross-validation
    grid.fit(data2fit)
    print 'best bandwidth in cross-validation: '+str(grid.best_params_)
    kde = grid.best_estimator_'''
    kde = KernelDensity(kernel='gaussian', bandwidth=0.015, rtol=1E-4).fit(data2fit)
    
    # plot KDE estimated pdf against the data grid
    phi_m = np.linspace(boundingbox[2], boundingbox[3], 100)
    phi_p = np.linspace(boundingbox[0], boundingbox[1], 100)
    X,Y = meshgrid(phi_m, phi_p)
    #print X
    #print Y
    positions = np.vstack([X.ravel(), Y.ravel()])
    #print len(positions[0])
    #print positions[0][0:20]
    
    data_grid = np.array([[positions[0][k], positions[1][k]] for k in xrange(len(positions[0]))])
    #print data_grid[0:10]
    #sys.exit(1)
    pdf_log = kde.score_samples(data_grid)
    #print pdf_log
    pdf = np.exp(pdf_log)
    #print sum(pdf)
    #sys.exit(1)
    pdf_format = []
    for i in xrange(len(X)):
        pdf_silce = []
        for k in xrange(len(X[0])):
            pdf_silce.append(pdf[i*len(X[0])+k])
        pdf_format.append(pdf_silce)
    
    #print pdf_format
    #p = ax.plot_surface(X, Y, pdf_format, rstride=4, cstride=4, linewidth=0)
    plt.contourf(X, Y, pdf_format, cmap=plt.cm.Reds)
    #m = Basemap(projection='cyl', llcrnrlat=boundingbox[0], urcrnrlat=boundingbox[1], llcrnrlon=boundingbox[2],urcrnrlon=boundingbox[3], resolution='c')
    #m.drawcoastlines()
    #m.drawcountries()
    #plt.show()
    return 0


'''
# Example of 3D plotting
'''
'''alpha = 0.7
phi_ext = 2 * np.pi * 0.5
def flux_qubit_potential(phi_m, phi_p):
    return 2 + alpha - 2 * cos(phi_p)*cos(phi_m) - alpha * cos(phi_ext - 2*phi_p)
phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T
fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)
plt.show()'''