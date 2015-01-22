'''
A visualization tool to examine the activities of different user class in big cities.
Usage: python visualize 
        -u role/gender/age 
        -c Amsterdam/London/Paris/Rome
        -a geo/temporal
        -m scatter/kde
'''

import MySQLdb
import sys
import numpy as np
import getopt
from scipy.misc import comb

from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.colors import Normalize
from pylab import *
from scipy.stats.distributions import norm
from mpl_toolkits.mplot3d.axes3d import Axes3D

from kdTree import *

# connect to database, which has the following main tables: 
# tweets of users sent in 3 weeks, user twitter profiles, and user derived profiles, venue profile
con = None
con = MySQLdb.connect(host='127.0.0.1', db='Culture', user='root', passwd='')
cur = con.cursor()

# define user categories according to the following fields 
u_role = ['FOREIGN_TOURIST', 'LOCAL_TOURIST', 'RESIDENT']
u_gender = ['Male', 'Female']
u_age = [(16,30),(31,45),(46,200)]

# define the location of four cities
loc_amsterdam = [52.299175, 52.427505, 4.739402, 4.989898]
loc_london = [51.4513133054, 51.5538081151, -0.3240025797, 0.0730833169]
loc_paris = [48.815573, 48.902145, 2.2609345347, 2.4228856827]
loc_rome = [41.8200201386, 41.9744022955, 12.3945787041, 12.5703003967]
name_cities = ['Amsterdam', 'London', 'Paris', 'Rome']
loc_cities = [loc_amsterdam, loc_london, loc_paris, loc_rome]

def color_grid(data_plt_rec, color_plt_rec, data_plt, pos):
    x_ll = min([x[0] for x in data_plt])
    y_ll = min([x[1] for x in data_plt])
    x_ur = max([x[0] for x in data_plt])
    y_ur = max([x[1] for x in data_plt])

    print [x_ll, y_ll, x_ur, y_ur]
    
    phi_m = np.linspace(x_ll, x_ur, 20)
    phi_p = np.linspace(y_ll, y_ur, 20)
    X,Y = meshgrid(phi_m, phi_p)
    positions = np.vstack([X.ravel(), Y.ravel()])
    data_grid = np.array([[positions[0][k], positions[1][k]] for k in xrange(len(positions[0]))])

    data_plt_rec.extend(data_grid)
    color_plt_rec.extend([pos for x in data_grid])

    return data_plt_rec, color_plt_rec

def vlzdif_uclass_city(loc_city, u_category, dc_from, dc_to, plotNo, plotIndex, axis):
    if axis=='geo':
        data_visulize_geo(loc_city, u_category, dc_from, dc_to,  plotNo, plotIndex)
 
    if axis=='temporal':
        data_visulize_temporal(loc_city, u_category, dc_from, dc_to, plotNo, plotIndex)

    return 0

def data_visulize_geo(boundingbox, u_category, dc_from, dc_to,  plotNo, plotIndex):
    data_from = [[x[0],x[1]] for x in dc_from]
    data_to = [[x[0],x[1]] for x in dc_to]
    del dc_from
    del dc_to

    data = data_from+data_to
    class_index = [1 for x in data_from]+[-1 for y in data_to]
    del data_from
    del data_to

    #print class_index
    #sys.exit(1)
    tree, leaves = kdtree(np.transpose(data), 2500)

    plt.subplot(1, plotNo, plotIndex, title = str(u_category)+': '+str(len(data)))
    plt.xlim( (boundingbox[2], boundingbox[3]) )
    plt.ylim( (boundingbox[0], boundingbox[1]) )

    data_plt = []
    color_plt = []
    data_plt_rec = []
    color_plt_rec = []
    #print 'No. regions: '+str(len(leaves))
    lens = []
    for region in leaves:
        pos = 0
        for dot_index in region:
            data_plt.append(data[dot_index])
            pos += class_index[dot_index]
        for dot_index in region:
            color_plt.append(pos)
        lens.append(len(region))

    print lens
        #data_plt_rec, color_plt_rec = color_grid(data_plt_rec, color_plt_rec, data_plt, pos)
    #print color_plt
    cmap=plt.cm.bwr
    mm = math.fabs(max(math.fabs(max(color_plt)),math.fabs(min(color_plt))))
    norm = Normalize(vmin=-math.fabs(max(math.fabs(max(color_plt)),math.fabs(min(color_plt)))), vmax=math.fabs(max(math.fabs(max(color_plt)),math.fabs(min(color_plt))))) 
    plt.scatter([x[0] for x in data_plt], [x[1] for x in data_plt],  c=cmap(norm(color_plt)), s=10)# s=1, c=np.array(color_plt)
    #plt.scatter([x[0] for x in data_plt_rec], [x[1] for x in data_plt_rec],  c=cmap(norm(color_plt_rec)), s=10)# s=1, c=np.array(color_plt)

    return 0

def data_visulize_temporal(boundingbox, u_category, dc_from, dc_to,  plotNo, plotIndex):
    plt.subplot(1, plotNo, plotIndex, title = u_category)
    plt.xlim( (0, 24) )

    x_grid = np.linspace(0, 24, 5000)

    datetimes_from = [x[0] for x in dc_from]
    datatimes_format_from = np.array([dt.hour + dt.minute/60. for dt in datetimes_from])
    kde_from = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(datatimes_format_from[:, np.newaxis])
    pdf_from = np.exp(kde_from.score_samples(x_grid[:, np.newaxis]))

    datetimes_to = [x[0] for x in dc_to]
    datatimes_format_to = np.array([dt.hour + dt.minute/60. for dt in datetimes_to])
    kde_to = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(datatimes_format_to[:, np.newaxis])
    pdf_to = np.exp(kde_to.score_samples(x_grid[:, np.newaxis]))
    
    plt.plot(x_grid,pdf_from-pdf_to)
    return 0

def usage():
    print "Usage: python visualize [-option] arg \n"+\
        "  -u role/gender/age              #select user categorization criteria\n"+\
        "  -c Amsterdam/London/Paris/Rome  #select city \n"+\
        "  -a geo/temporal                 #select the dimension for visualization\n"
    return 0

if __name__ == '__main__':
    u_cat = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "u:c:a:")
        #print opts
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    for o, a in opts:
        if o == '-u':
            u_cat = a
            if a == 'role':
                u_class = u_role
            if a == 'gender':
                u_class = u_gender
            if a == 'age':
                u_class = u_age
        if o == '-c':
            name_city = a
        if o == '-a':
            axis = a
    
    # for each city, each user class, visualize the geographical distribution

    fig = plt.figure(figsize=(16, 5))
    i = name_cities.index(name_city)
    loc_city = loc_cities[i]
    #fig.suptitle('User activities in ' + name_city, fontsize=20)
    print 'Results of: ' + name_city
    data_each_category = []
    for u_category in u_class:
        # get all activities locations inside the city, performed by the specified category of users
        if u_class==u_role:
            print '..' + u_category
            if axis=='geo':
                qry = "select longitude, latitude from inst_posts as T, inst_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.uclass='" + u_category + "' and l_city='"+name_city+"'"
            if axis=='temporal':
                qry = "select T.createdAt from inst_posts as T, inst_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.uclass='" + u_category + "' and l_city='"+name_city+"'"
        if u_class==u_gender:
            print '..' + u_category
            if axis=='geo':
                qry = "select longitude, latitude from inst_posts as T, inst_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.g_gender='" + u_category + "' and l_city='"+name_city+"'"
            if axis=='temporal':
                qry = "select T.createdAt from inst_posts as T, inst_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.g_gender='" + u_category + "' and l_city='"+name_city+"'"
        if u_class == u_age:
            print '..age in (' + str(u_category[0])+','+str(u_category[1])+')'
            if axis=='geo':
                qry = "select longitude, latitude from inst_posts as T, inst_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.g_age>=" + str(u_category[0]) + " and U.g_age<="+str(u_category[1]) + " and l_city='"+name_city+"'"
            if axis=='temporal':
                qry = "select T.createdAt from inst_posts as T, inst_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.g_age>=" + str(u_category[0]) + " and U.g_age<="+str(u_category[1]) + " and l_city='"+name_city+"'"
        cur.execute(qry)
        tweets_info = cur.fetchall()
        print '...No. Tweets: ' + str(len(tweets_info))
        data_each_category.append(tweets_info)
    
    exist_diff_sets = []
    plotIndex = 0
    for dc_from in data_each_category:
        name_from = u_class[data_each_category.index(dc_from)]
        for dc_to in data_each_category:
            name_to = u_class[data_each_category.index(dc_to)]
            if dc_from != dc_to:
                this_diff_set = set()
                this_diff_set.add(name_from)
                this_diff_set.add(name_to)
                if this_diff_set in exist_diff_sets:
                    continue
                exist_diff_sets.append(this_diff_set)
                print str(name_from)+'_'+str(name_to)
                vlzdif_uclass_city(loc_city, str(name_from)+'_'+str(name_to), dc_from, dc_to, int(round(comb(len(u_class),2))), plotIndex, axis)
                plotIndex += 1

    #plt.show()
    plt.savefig('plt/inst_'+name_city+'_'+u_cat+'_'+axis)