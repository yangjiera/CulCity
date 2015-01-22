'''
ToDo:
    role 
'''

import MySQLdb
import sys
import numpy as np
import getopt
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib import dates
from pylab import *
from scipy.stats.distributions import norm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from GiniCoef import *

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

# define user categories according to the following fields 
u_role = ['FOREIGN_TOURIST', 'LOCAL_TOURIST', 'RESIDENT']
u_gender = ['Male', 'Female']
u_age = [(16,30),(31,45),(46,200)]
u_classes = [u_role, u_gender, u_age]

def get_sub_qry(u_category, u_class, name_city):
    sub_qry = ""
    if u_class==u_role:
        sub_qry = "U.uclass='" + u_category + "' and l_city='"+name_city+"'"
    if u_class==u_gender:
        sub_qry = "U.g_gender='" + u_category + "' and l_city='"+name_city+"'"
    if u_class==u_age:
        sub_qry = "U.g_age>=" + str(u_category[0]) + " and U.g_age<="+str(u_category[1]) + " and l_city='"+name_city+"'"
        
    return sub_qry

def get_pdf_temporal(data, boundingbox):
    datetimes = [x[2] for x in data]
    datatimes_format = np.array([dt.hour + dt.minute/60. for dt in datetimes])

    x_grid = np.linspace(0, 24, 5000)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=0.12).fit(datatimes_format[:, np.newaxis])
    pdf = np.exp(kde.score_samples(x_grid[:, np.newaxis]))
    
    return pdf

def get_pdf_geo(data, boundingbox):
    data2fit = np.array([[x[0], x[1]] for x in data])
    del data
    #print data2fit.shape
    
    try:
        kde = KernelDensity(kernel='gaussian', bandwidth=0.015, rtol=1E-4).fit(data2fit)
    except:
        print 'invalid data2fit'
        return -1 
    
    phi_m = np.linspace(boundingbox[2], boundingbox[3], 100)
    phi_p = np.linspace(boundingbox[0], boundingbox[1], 100)
    X,Y = meshgrid(phi_m, phi_p)
    positions = np.vstack([X.ravel(), Y.ravel()])
    data_grid = np.array([[positions[0][k], positions[1][k]] for k in xrange(len(positions[0]))])
    
    pdf_log = kde.score_samples(data_grid)
    pdf = np.exp(pdf_log)
    #s_pdf = sum(pdf)
    #return [float(x)/s_pdf for x in pdf]
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

def get_stats(qry, loc_city):
    stats_vector = []#No.posts, No.users, Gini.Space, Gini.Temporal, Gini.ActivityType
    
    #cur.execute("select longitude, latitude, T.createdAt, T.venueId "+qry)
    cur.execute("select longitude, latitude, T.createdAt "+qry)
    post_results = cur.fetchall()
    
    if len(post_results) == 0:
        return [0, 0, None, None]

    #No.Posts
    stats_vector.append(len(post_results))
    #No.Users
    cur.execute("select distinct T.userId "+qry)
    user_results = cur.fetchall()
    stats_vector.append(len(user_results))
    #Gini.Space
    '''print post_results[0:10]
    print loc_city
    sys.exit(1)'''
    pdf_space = get_pdf_geo(post_results, loc_city)
    gini_space = GRLC(pdf_space)[1]
    stats_vector.append(gini_space)
    #Gini.Temporal
    pdf_spatial = get_pdf_temporal(post_results, loc_city)
    gini_spatial = GRLC(pdf_spatial)[1]
    stats_vector.append(gini_spatial)
    '''#Gini.ActivityType
    venues_results = get_venue_root_categories(post_results)
    etp_venues = entropy2(venues_results)
    stats_vector.append(etp_venues)'''
    
    return stats_vector

def print_stats(loc_city, u_class_tier1, u_class_tier2, u_class_tier3, name_city, f):
    qry_base1 = "from inst_posts as T, inst_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and T.userId=U.id and "
                            
    for u_category_tier1 in u_class_tier1:
        '''qry_base2 = qry_base1+get_sub_qry(u_category_tier1, u_class_tier1, name_city)+" and "
        for u_category_tier2 in u_class_tier2:
            qry_base3 = qry_base2+get_sub_qry(u_category_tier2, u_class_tier2, name_city)+" and "
            for u_category_tier3 in u_class_tier3:
                qry = qry_base3+get_sub_qry(u_category_tier3, u_class_tier3, name_city)
                result_tier3 = str(u_category_tier1)+'>>'+str(u_category_tier2)+'>>'+str(u_category_tier3)+': '
                result_tier3 += str(get_stats(qry, loc_city))
                f.write(result_tier3+'\n')
            qry = qry_base2+get_sub_qry(u_category_tier2, u_class_tier2, name_city)
            result_tier2 = str(u_category_tier1)+'>>'+str(u_category_tier2)+': '
            result_tier2 += str(get_stats(qry, loc_city))
            f.write(result_tier2+'\n')'''
        qry = qry_base1+get_sub_qry(u_category_tier1, u_class_tier1, name_city)
        result_tier1 = str(u_category_tier1)+': '
        result_tier1 += str(get_stats(qry, loc_city))
        f.write(result_tier1+'\n')
        
    return 0

if __name__ == '__main__':
    #axes = ['geo', 'temporal']
    #axis = sys.argv[1]
    f = open('result_inst_level1.txt', 'w')
    for name_city in name_cities:
        i = name_cities.index(name_city)
        loc_city = loc_cities[i]

        f.write('Results of: ' + name_city+'\n')
        plotIndex = 0
        for u_class_tier1 in u_classes:
            for u_class_tier2 in u_classes:
                if u_class_tier2 == u_class_tier1:
                    continue
                for u_class_tier3 in u_classes:
                    if u_class_tier3 == u_class_tier2 or u_class_tier3 == u_class_tier1:
                        continue
                    print_stats(loc_city, u_class_tier1, u_class_tier2, u_class_tier3, name_city, f)

    f.close()