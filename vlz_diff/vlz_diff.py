'''
A visualization tool to examine the activities of different user class in big cities.
Usage: python visualize 
        -u role/gender/age 
        -c Amsterdam/London/Paris/Rome
        -a geo/temporal
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
#from mpl_toolkits.basemap import Basemap

from quadtree_heat import *
from carea import *

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

def color_grid(data_plt):
    x_ll = min([x[0] for x in data_plt])
    y_ll = min([x[1] for x in data_plt])
    x_ur = max([x[0] for x in data_plt])
    y_ur = max([x[1] for x in data_plt])

    return [x_ll, y_ll, x_ur-x_ll, y_ur-y_ll]
    
    phi_m = np.linspace(x_ll, x_ur, 20)
    phi_p = np.linspace(y_ll, y_ur, 20)
    X,Y = meshgrid(phi_m, phi_p)
    positions = np.vstack([X.ravel(), Y.ravel()])
    data_grid = np.array([[positions[0][k], positions[1][k]] for k in xrange(len(positions[0]))])

    data_plt_rec.extend(data_grid)
    color_plt_rec.extend([pos for x in data_grid])

    return data_plt_rec, color_plt_rec

def check_overlap(recs):
    for bb1 in recs:
        for bb2 in recs:
            if bb1 == bb2:
                continue
            p1 = Point(bb1[0], bb1[1])
            p2 = Point(bb1[0]+bb1[2], bb1[1]+bb1[3])
            r1 = Rect(p1,p2)
            p3 = Point(bb2[0], bb2[1])
            p4 = Point(bb2[0]+bb2[2], bb2[1]+bb2[3])
            r2 = Rect(p3,p4)
            if overlap(r1, r2):
                print '[ERROR]: rectangle areas overlapped!!'
                sys.exit(1)

def vlzdif_uclass_city(loc_city, u_category, dc_from, dc_to, plotNo, plotIndex, axis, dc_third):
    if axis=='geo':
        data_visulize_geo(loc_city, u_category, dc_from, dc_to, dc_third, plotNo, plotIndex)
 
    if axis=='temporal':
        data_visulize_temporal(loc_city, u_category, dc_from, dc_to, dc_third, plotNo, plotIndex)

    return 0

def data_visulize_geo(boundingbox, u_category, dc_from, dc_to, dc_third, plotNo, plotIndex):
    data_from = [[x[0],x[1]] for x in dc_from]
    data_to = [[x[0],x[1]] for x in dc_to]
    data_third = [[x[0],x[1]] for x in dc_third]
    del dc_from
    del dc_to
    del dc_third

    class_pos = float(1)/len(data_from)
    class_neg = float(1)/len(data_to)
    data = data_from+data_to
    class_index = [class_pos for x in data_from]+[-class_neg for y in data_to]
    del data_from
    del data_to

    global leaves_data
    global leaves_idx
    global quads
    global iterations
    #leaves_data = []
    #leaves_idx = []
    #quads = []
    depth = 3
    data_temp = [x for x in data]+data_third
    Idx = [i for i in xrange(len(data_temp))]
    QT = QuadTree(np.array(data_temp), Idx, (boundingbox[2],boundingbox[0]), (boundingbox[3],boundingbox[1]), depth)

    #print iterations

    plt.subplot(1, plotNo, plotIndex, title = str(u_category)+': '+str(len(data)))
    plt.xlim( (boundingbox[2], boundingbox[3]) )
    plt.ylim( (boundingbox[0], boundingbox[1]) )
    
    #print leaves_idx
    color_plt = []
    #data_plt_rec = []
    #color_plt_rec = []
    #print 'No. regions: '+str(len(leaves))
    lens = []
    recs = []
    for region in leaves_idx:
        data_plt = []
        pos = 0
        for dot_index in region:
            if dot_index>=len(data):
                continue
            data_plt.append(data[dot_index])
            pos += class_index[dot_index]
        if len(data_plt) == 0:
            pos = 0
        color_plt.append(pos)
        lens.append(len(region))

    print lens
    print color_plt
    cmap=plt.cm.jet
    #mm = math.fabs(max(math.fabs(max(color_plt)),math.fabs(min(color_plt))))
    #norm = Normalize(vmin=-math.fabs(max(math.fabs(max(color_plt)),math.fabs(min(color_plt)))), vmax=math.fabs(max(math.fabs(max(color_plt)),math.fabs(min(color_plt))))) 
    norm = Normalize(vmin=min(color_plt), vmax=max(color_plt)) 
    
    #plt.scatter([x[0] for x in data_plt], [x[1] for x in data_plt],  c=cmap(norm(color_plt)), s=10)# s=1, c=np.array(color_plt)
    #plt.scatter([x[0] for x in data_plt_rec], [x[1] for x in data_plt_rec],  c=cmap(norm(color_plt_rec)), s=10)# s=1, c=np.array(color_plt)

    '''#check_overlap(quads)
    depth  = depth**3
    colorplot_format = []
    for i in xrange(depth):
        color_silce = []
        for k in xrange(depth):
            color_silce.append(color_plt[i*depth+k])
        colorplot_format.append(color_silce)
    colorplot_format = np.array(colorplot_format)
    heatmap = plt.pcolor(colorplot_format)

    for y in range(colorplot_format.shape[0]):
        for x in range(colorplot_format.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.4f' % colorplot_format[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

    plt.colorbar(heatmap)'''


    for rec in quads:
        this_color = cmap(norm(color_plt[quads.index(rec)]))
        #print rec
        #print this_color
        rectangle = plt.Rectangle((rec[0], rec[1]), rec[2], rec[3], color=this_color)# s=1, c=np.array(color_plt)
        x = rec[0]+float(rec[2])/2
        y = rec[1]+float(rec[3])/2
        c = norm(color_plt[quads.index(rec)])-0.5
        text = plt.text(rec[0]+float(rec[2])/2, rec[1]+float(rec[3])/2, '%.1f' % float(norm(color_plt[quads.index(rec)])),
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
        plt.gca().add_patch(rectangle)
        #plt.gca().add_patch(text)
    #plt.colorbar(cmap)
    
    return 0

#old implementation using kdtree
"""def data_visulize_geo(boundingbox, u_category, dc_from, dc_to, dc_third, plotNo, plotIndex):
    data_from = [[x[0],x[1]] for x in dc_from]
    data_to = [[x[0],x[1]] for x in dc_to]
    data_third = [[x[0],x[1]] for x in dc_third]
    del dc_from
    del dc_to
    del dc_third

    class_pos = float(1)/len(data_from)
    class_neg = float(1)/len(data_to)
    data = data_from+data_to
    class_index = [class_pos for x in data_from]+[-class_neg for y in data_to]
    del data_from
    del data_to

    data_temp = [x for x in data]+data_third
    tree, leaves = kdtree(np.transpose(data_temp), 1000)

    plt.subplot(1, plotNo, plotIndex, title = str(u_category)+': '+str(len(data)))
    plt.xlim( (boundingbox[2], boundingbox[3]) )
    plt.ylim( (boundingbox[0], boundingbox[1]) )

    
    color_plt = []
    #data_plt_rec = []
    #color_plt_rec = []
    #print 'No. regions: '+str(len(leaves))
    lens = []
    recs = []
    for region in leaves:
        data_plt = []
        pos = 0
        for dot_index in region:
            if dot_index>=len(data):
                continue
            data_plt.append(data[dot_index])
            pos += class_index[dot_index]
        if len(data_plt) == 0:
            continue
        color_plt.append(pos)
        lens.append(len(region))
        rec = color_grid(data_plt)
        recs.append(rec)
    #print lens
    #print color_plt
    cmap=plt.cm.bwr
    mm = math.fabs(max(math.fabs(max(color_plt)),math.fabs(min(color_plt))))
    norm = Normalize(vmin=-math.fabs(max(math.fabs(max(color_plt)),math.fabs(min(color_plt)))), vmax=math.fabs(max(math.fabs(max(color_plt)),math.fabs(min(color_plt))))) 
    #plt.scatter([x[0] for x in data_plt], [x[1] for x in data_plt],  c=cmap(norm(color_plt)), s=10)# s=1, c=np.array(color_plt)
    #plt.scatter([x[0] for x in data_plt_rec], [x[1] for x in data_plt_rec],  c=cmap(norm(color_plt_rec)), s=10)# s=1, c=np.array(color_plt)

    check_overlap(recs)

    for rec in recs:
        this_color = cmap(norm(color_plt[recs.index(rec)]))
        #print rec
        #print this_color
        rectangle = plt.Rectangle((rec[0], rec[1]), rec[2], rec[3], color=this_color)# s=1, c=np.array(color_plt)
        plt.gca().add_patch(rectangle)
    return 0"""

def data_visulize_temporal(boundingbox, u_category, dc_from, dc_to, dc_third, plotNo, plotIndex):
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
                qry = "select longitude, latitude from twitter_tweets as T, twitter_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.uclass='" + u_category + "' and l_city='"+name_city+"'"
            if axis=='temporal':
                qry = "select T.createdAt from twitter_tweets as T, twitter_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.uclass='" + u_category + "' and l_city='"+name_city+"'"
        if u_class==u_gender:
            print '..' + u_category
            if axis=='geo':
                qry = "select longitude, latitude from twitter_tweets as T, twitter_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.g_gender='" + u_category + "' and l_city='"+name_city+"'"
            if axis=='temporal':
                qry = "select T.createdAt from twitter_tweets as T, twitter_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.g_gender='" + u_category + "' and l_city='"+name_city+"'"
        if u_class == u_age:
            print '..age in (' + str(u_category[0])+','+str(u_category[1])+')'
            if axis=='geo':
                qry = "select longitude, latitude from twitter_tweets as T, twitter_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.g_age>=" + str(u_category[0]) + " and U.g_age<="+str(u_category[1])+ " and l_city='"+name_city+"'"
            if axis=='temporal':
                qry = "select T.createdAt from twitter_tweets as T, twitter_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and " + \
                "T.userId=U.id and U.g_age>=" + str(u_category[0]) + " and U.g_age<="+str(u_category[1])+ " and l_city='"+name_city+"'"

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
                for dc_third in data_each_category:
                    if dc_third!=dc_from and dc_third!=dc_to:
                        break
                vlzdif_uclass_city(loc_city, str(name_from)+'_'+str(name_to), dc_from, dc_to, int(round(comb(len(u_class),2))), plotIndex, axis, dc_third)
                plotIndex += 1
    #plt.subplot(1, int(round(comb(len(u_class),2)))+1, plotIndex)
    #plt.colorbar(cmap=plt.cm.jet)
    plt.show()
    #plt.savefig('plt/twit_'+name_city+'_'+u_cat+'_'+axis)