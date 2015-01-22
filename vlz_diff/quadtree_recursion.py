# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
import resource, sys
#resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
#resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
#print sys.getrecursionlimit()
sys.setrecursionlimit(10000)
#print sys.getrecursionlimit()
from matplotlib import pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
#from astroML.plotting import setup_text_plots
#setup_text_plots(fontsize=8, usetex=True)


# We'll create a QuadTree class which will recursively subdivide the
# space into quadrants
leaves_data = []
leaves_idx = []
quads = []
iterations = 0
class QuadTree:
    """Simple Quad-tree class"""

    # class initialization function
    def __init__(self, data, Idx, mins, maxs, leafsize, depth=-1):
        self.data = np.asarray(data)
        global leaves_data
        global leaves_idx
        global quads
        global iterations
        # data should be two-dimensional
        assert self.data.shape[1] == 2

        if mins is None:
            mins = data.min(0)
        if maxs is None:
            maxs = data.max(0)

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.sizes = self.maxs - self.mins

        self.children = []
        #self.leaves = []

        mids = 0.5 * (self.mins + self.maxs)
        xmin, ymin = self.mins
        xmax, ymax = self.maxs
        xmid, ymid = mids

        if len(data) > leafsize:
            iterations += 1
            # split the data into four quadrants
            #print data[0:10]
            index1 = (data[:, 0] < mids[0]) & (data[:, 1] < mids[1])
            data_q1 = data[index1]
            Idx_q1 = [Idx[i] for i in xrange(len(index1)) if index1[i]]

            index2 = (data[:, 0] < mids[0]) & (data[:, 1] >= mids[1])
            data_q2 = data[index2]
            Idx_q2 = [Idx[i] for i in xrange(len(index2)) if index2[i]]

            index3 = (data[:, 0] >= mids[0]) & (data[:, 1] < mids[1])
            data_q3 = data[index3]
            Idx_q3 = [Idx[i] for i in xrange(len(index3)) if index3[i]]

            index4 = (data[:, 0] >= mids[0]) & (data[:, 1] >= mids[1])
            data_q4 = data[index4]
            Idx_q4 = [Idx[i] for i in xrange(len(index4)) if index4[i]]

            # recursively build a quad tree on each quadrant which has data
            if len(data_q1) > leafsize:
                self.children.append(QuadTree(data_q1, Idx_q1, 
                                              [xmin, ymin], [xmid, ymid], leafsize))
            else:
                if len(data_q1)>0:
                    leaves_data.append(data_q1)
                    leaves_idx.append(Idx_q1)
                    quads.append([xmin, ymin, xmid-xmin, ymid-ymin])
            if len(data_q2) > leafsize:
                self.children.append(QuadTree(data_q2, Idx_q2, 
                                              [xmin, ymid], [xmid, ymax], leafsize))
            else:
                if len(data_q2)>0:
                    leaves_data.append(data_q2)
                    leaves_idx.append(Idx_q2)
                    quads.append([xmin, ymid, xmid-xmin, ymax-ymid])
            if len(data_q3) > leafsize:
                self.children.append(QuadTree(data_q3, Idx_q3, 
                                              [xmid, ymin], [xmax, ymid], leafsize))
            else:
                if len(data_q3)>0:
                    leaves_data.append(data_q3)
                    leaves_idx.append(Idx_q3)
                    quads.append([xmid, ymin, xmax-xmid, ymid-ymin])
            if len(data_q4) > leafsize:
                self.children.append(QuadTree(data_q4, Idx_q4, 
                                              [xmid, ymid], [xmax, ymax], leafsize))
            else:
                if len(data_q4)>0:
                    leaves_data.append(data_q4)
                    leaves_idx.append(Idx_q4)
                    quads.append([xmid, ymid, xmax-xmid, ymax-ymid])

            #print leaves_idx

    def draw_rectangle(self, ax, depth):
        """Recursively plot a visualization of the quad tree region"""
        if depth is None or depth == 0:
            rect = plt.Rectangle(self.mins, *self.sizes, zorder=2,
                                 ec='#000000', fc='none')
            ax.add_patch(rect)
        if depth is None or depth > 0:
            for child in self.children:
                child.draw_rectangle(ax, depth - 1)


def draw_grid(ax, xlim, ylim, Nx, Ny, **kwargs):
    """ draw a background grid for the quad tree"""
    for x in np.linspace(xlim[0], xlim[1], Nx):
        ax.plot([x, x], ylim, **kwargs)
    for y in np.linspace(ylim[0], ylim[1], Ny):
        ax.plot(xlim, [y, y], **kwargs)

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

"""#------------------------------------------------------------
# Create a set of structured random points in two dimensions
np.random.seed(0)

X = np.random.random((30, 2)) * 2 - 1
X[:, 1] *= 0.1
X[:, 1] += X[:, 0] ** 2

#------------------------------------------------------------
# Use our Quad Tree class to recursively divide the space
mins = (-1.1, -0.1)
maxs = (1.1, 1.1)
Idx = [i for i in xrange(len(X))]
#print X
#print mins
#print maxs
QT = QuadTree(X, Idx, mins, maxs, 2, depth=3)
#print leaves_data
#print leaves_idx
print quads

fig = plt.figure(figsize=(5, 5))
plt.xlim( (-1.1, 1.1) )
plt.ylim( (-0.1, 1.1) )
i = 1
cmap = get_cmap(30)
for rec in quads:
        rectangle = plt.Rectangle((rec[0], rec[1]), rec[2], rec[3], color=cmap(i))# s=1, c=np.array(color_plt)
        plt.gca().add_patch(rectangle)
        i += 1
plt.scatter([xx[0] for xx in X], [xx[1] for xx in X])
plt.show()
sys.exit(1)

#------------------------------------------------------------
# Plot four different levels of the quad tree
fig = plt.figure(figsize=(5, 5))
fig.subplots_adjust(wspace=0.1, hspace=0.15,
                    left=0.1, right=0.9,
                    bottom=0.05, top=0.9)

for level in range(1, 5):
    ax = fig.add_subplot(2, 2, level, xticks=[], yticks=[])
    ax.scatter(X[:, 0], X[:, 1])
    QT.draw_rectangle(ax, depth=level - 1)

    Nlines = 1 + 2 ** (level - 1)
    #draw_grid(ax, (mins[0], maxs[0]), (mins[1], maxs[1]), Nlines, Nlines, linewidth=1, color='#CCCCCC', zorder=0)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.15, 1.15)
    ax.set_title('level %i' % level)

# suptitle() adds a title to the entire figure
fig.suptitle('Quad-tree Example')
plt.show()"""