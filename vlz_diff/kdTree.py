# Copyleft 2008 Sturla Molden
# University of Oslo

#import psyco
#psyco.full()

import numpy
import copy


import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.colors import Normalize
from pylab import *
from scipy.stats.distributions import norm
from mpl_toolkits.mplot3d.axes3d import Axes3D

def kdtree( data, leafsize):
    """
    build a kd-tree for O(n log n) nearest neighbour search

    input:
        data:       2D ndarray, shape =(ndim,ndata), preferentially C order
        leafsize:   max. number of data points to leave in a leaf

    output:
        kd-tree:    list of tuples
    """

    leaves = []

    ndim = data.shape[0]
    ndata = data.shape[1]
    #print ndim
    #print ndata

    # find bounding hyper-rectangle
    hrect = numpy.zeros((2,data.shape[0]))
    hrect[0,:] = data.min(axis=1)
    hrect[1,:] = data.max(axis=1)

    # create root of kd-tree
    idx = numpy.argsort(data[0,:], kind='mergesort')
    data[:,:] = data[:,idx]
    splitval = data[0,ndata/2]

    left_hrect = hrect.copy()
    right_hrect = hrect.copy()
    left_hrect[1, 0] = splitval
    right_hrect[0, 0] = splitval

    tree = [(None, None, left_hrect, right_hrect, None, None)]

    stack = [(data[:,:ndata/2], idx[:ndata/2], 1, 0, True),
             (data[:,ndata/2:], idx[ndata/2:], 1, 0, False)]

    # recursively split data in halves using hyper-rectangles:
    while stack:

        # pop data off stack
        data, didx, depth, parent, leftbranch = stack.pop()
        ndata = data.shape[1]
        nodeptr = len(tree)

        # update parent node

        _didx, _data, _left_hrect, _right_hrect, left, right = tree[parent]

        tree[parent] = (_didx, _data, _left_hrect, _right_hrect, nodeptr, right) if leftbranch \
            else (_didx, _data, _left_hrect, _right_hrect, left, nodeptr)

        # insert node in kd-tree

        # leaf node?
        if ndata <= leafsize:
            _didx = didx.copy()
            _data = data.copy()
            leaf = (_didx, _data, None, None, 0, 0)
            #leaf = (_data)
            tree.append(leaf)
            leaves.append(_didx)

        # not a leaf, split the data in two      
        else:
            splitdim = depth % ndim
            idx = numpy.argsort(data[splitdim,:], kind='mergesort')
            data[:,:] = data[:,idx]
            didx = didx[idx]
            nodeptr = len(tree)
            stack.append((data[:,:ndata/2], didx[:ndata/2], depth+1, nodeptr, True))
            stack.append((data[:,ndata/2:], didx[ndata/2:], depth+1, nodeptr, False))
            splitval = data[splitdim,ndata/2]
            if leftbranch:
                left_hrect = _left_hrect.copy()
                right_hrect = _left_hrect.copy()
            else:
                left_hrect = _right_hrect.copy()
                right_hrect = _right_hrect.copy()
            left_hrect[1, splitdim] = splitval
            right_hrect[0, splitdim] = splitval

            #print data
            # append node to tree
            tree.append((None, None, left_hrect, right_hrect, None, None))

    return tree,leaves

def color_grid(data_plt):
    print data_plt
    x_ll = min([x[0] for x in data_plt])
    y_ll = min([x[1] for x in data_plt])
    x_ur = max([x[0] for x in data_plt])
    y_ur = max([x[1] for x in data_plt])

    return [x_ll, y_ll, x_ur, y_ur]
    
    phi_m = np.linspace(x_ll, x_ur, 20)
    phi_p = np.linspace(y_ll, y_ur, 20)
    X,Y = meshgrid(phi_m, phi_p)
    positions = np.vstack([X.ravel(), Y.ravel()])
    data_grid = np.array([[positions[0][k], positions[1][k]] for k in xrange(len(positions[0]))])

    data_plt_rec.extend(data_grid)
    color_plt_rec.extend([pos for x in data_grid])

    return data_plt_rec, color_plt_rec

def overlap(r1, r2):
    return range_overlap(r1.left, r1.right, r2.left, r2.right) and range_overlap(r1.bottom, r1.top, r2.bottom, r2.top)
def range_overlap(a_min, a_max, b_min, b_max):
    return not ((a_min >= b_max) or (b_min >= a_max))

if __name__ == '__main__':
    data = [[3,2], [-3,-2],[-2,-1],[-1,-1],[1,1],[2,1]]
    print data
    data_temp = [x for x in data]
    tree,leaves = kdtree(numpy.transpose(numpy.array(data_temp)),3)
    print leaves
    
    color_plt = []
    data_plt_rec = []
    color_plt_rec = []
    lens = []
    recs = []
    for region in leaves:
        data_plt = []
        #print region
        pos = 0
        #print data
        for dot_index in region:
            data_plt.append(data[dot_index])
            pos = 1
        for dot_index in region:
            color_plt.append(pos)
        lens.append(len(region))
        rec = color_grid(data_plt)
        recs.append(rec)
    print rec
    plt.subplot(111)
    plt.axis('scaled')
    rectangle = plt.Rectangle((rec[0], rec[1]), rec[2]-rec[0], rec[3]-rec[1], fc='r')
    plt.gca().add_patch(rectangle)
    plt.xlim([-4, 4])
    plt.ylim([-3, 3])
    plt.show()
	