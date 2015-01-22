import sys
sys.path.append(sys.path[0]+'/../stats/')
from GiniCoef import *

import MySQLdb
import numpy as np
import getopt
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib import dates
from pylab import *
from scipy.stats.distributions import norm
from mpl_toolkits.mplot3d.axes3d import Axes3D


# connect to database, which has the following main tables: 
con = None
con = MySQLdb.connect(host='127.0.0.1', db='Culture', user='root', passwd='')
cur = con.cursor()

# define the location of four cities
loc_amsterdam = [52.299175, 52.427505, 4.739402, 4.989898]
loc_london = [51.4513133054, 51.5538081151, -0.3240025797, 0.0730833169]
loc_paris = [48.815573, 48.902145, 2.2609345347, 2.4228856827]
loc_rome = [41.8200201386, 41.9744022955, 12.3945787041, 12.5703003967]
name_cities = ['Amsterdam', 'London', 'Paris', 'Rome']
loc_cities = [loc_amsterdam, loc_london, loc_paris, loc_rome]


def get_pdf_temporal(data, boundingbox, plotIndex, table):
    datetimes = [x[2] for x in data]
    datatimes_format = np.array([dt.hour + dt.minute/60. for dt in datetimes])

    x_grid = np.linspace(0, 24, 5000)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=0.12).fit(datatimes_format[:, np.newaxis])
    pdf = np.exp(kde.score_samples(x_grid[:, np.newaxis]))

    plt.subplot(1, 2, plotIndex, title = 'Temporal distribution')
    plt.xlim( (0, 24) )
    plt.plot(x_grid,pdf)

    return pdf

def get_pdf_geo(data, boundingbox, plotIndex, table):
    data2fit = np.array([[x[0], x[1]] for x in data])
    del data
    
    kde = KernelDensity(kernel='gaussian', bandwidth=0.015, rtol=1E-4).fit(data2fit)
    
    phi_m = np.linspace(boundingbox[2], boundingbox[3], 100)
    phi_p = np.linspace(boundingbox[0], boundingbox[1], 100)
    X,Y = meshgrid(phi_m, phi_p)
    positions = np.vstack([X.ravel(), Y.ravel()])
    data_grid = np.array([[positions[0][k], positions[1][k]] for k in xrange(len(positions[0]))])
    
    pdf_log = kde.score_samples(data_grid)
    pdf = np.exp(pdf_log)
    #s_pdf = sum(pdf)
    #return [float(x)/s_pdf for x in pdf]

    plt.subplot(1, 2, plotIndex, title = 'Spatial distribution')
    plt.xlim( (boundingbox[2], boundingbox[3]) )
    plt.ylim( (boundingbox[0], boundingbox[1]) )
    pdf_format = []
    for i in xrange(len(X)):
        pdf_silce = []
        for k in xrange(len(X[0])):
            pdf_silce.append(pdf[i*len(X[0])+k])
        pdf_format.append(pdf_silce)
    plt.contourf(X, Y, pdf_format, cmap=plt.cm.Reds)

    return pdf

def entropy2(s):
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def get_venue_root_categories(data):
    venueIds = [x[3] for x in data]
    root_categories = []
    for vid in venueIds:
        print vid
        cur.execute("select root_category from fsq_venues where name='"+str(vid)+"'")
        print cur.fetchone()
        #select name from fsq_category where id=

def get_stats_plot(qry, loc_city, table):
    stats_vector = []#No.posts, No.users, Gini.Space, Gini.Temporal, Gini.ActivityType
    
    #cur.execute("select longitude, latitude, T.createdAt, T.venueId "+qry)
    cur.execute("select longitude, latitude, createdAt "+qry)
    post_results = cur.fetchall()
    
    #No.Posts
    stats_vector.append(len(post_results))
    #No.Users
    cur.execute("select distinct userId "+qry)
    user_results = cur.fetchall()
    stats_vector.append(len(user_results))
    #Gini.Space
    '''print post_results[0:10]
    print loc_city
    sys.exit(1)'''
    plotIndex = 0
    pdf_space = get_pdf_geo(post_results, loc_city, plotIndex, table)
    gini_space = GRLC(pdf_space)[1]
    stats_vector.append(gini_space)
    #Gini.Temporal
    plotIndex += 1
    pdf_spatial = get_pdf_temporal(post_results, loc_city, plotIndex, table)
    gini_spatial = GRLC(pdf_spatial)[1]
    stats_vector.append(gini_spatial)
    '''#Gini.ActivityType
    venues_results = get_venue_root_categories(post_results)
    etp_venues = entropy2(venues_results)
    stats_vector.append(etp_venues)'''
    
    return stats_vector

def print_plot_stats(loc_city, name_city, table):
    if table=='twitter':
        qry_base = "from twitter_tweets " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + "))"
    if table=='instgram':
        qry_base = "from inst_posts " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + "))"
    

    fig = plt.figure(figsize=(16, 5))
    result_tier = str(name_city)+'>>'+str(table)+': '
    result_tier += str(get_stats_plot(qry_base, loc_city, table))
    print result_tier
    plt.savefig('sm/'+name_city+'_'+table)
    
    return 0

if __name__ == '__main__':
    #axes = ['geo', 'temporal']
    tables = ['twitter', 'instgram']
    for name_city in name_cities:
        i = name_cities.index(name_city)
        loc_city = loc_cities[i]

        print 'Results of: ' + name_city
        
        
        for table in tables:        
            print_plot_stats(loc_city, name_city, table)
            #plotIndex += 1
        