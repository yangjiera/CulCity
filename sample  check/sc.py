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

import random
import os
import pickle

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



def print_plot_stats(loc_city, name_city, table):
    if table=='twitter':
        qry_all = "select distinct userId from twitter_tweets " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + "))"
        qry_sample = "select distinct U.id from twitter_tweets as T, twitter_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and "+\
                "T.userId=U.id"
    
    if table=='instgram':
        qry_all = "select distinct userId from inst_posts " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + "))"
        qry_sample = "select distinct U.id from inst_posts as T, inst_profiles as U " + \
                "where ((`latitude` between " + str(loc_city[0]) + " and " + str(loc_city[1]) + ") and (`longitude` between " + str(loc_city[2]) + " and " + str(loc_city[3]) + ")) and "+\
                "T.userId=U.id"
    
    cur.execute(qry_all)
    users_all = cur.fetchall()
    users_all = [x[0] for x in users_all]

    cur.execute(qry_sample)
    users_sample = cur.fetchall()
    users_sample = [x[0] for x in users_sample]

    print 'all users: '+str(len(users_all))
    dumpfile(users_all, table+'_users_all')
    print 'sample users: '+str(len(users_sample))
    dumpfile(users_sample, table+'_users_sample')

    random.shuffle(users_all)
    '''quater_random = users_all[0:300]
    dumpfile(quater_random, table+'_quater_random')

    random_diff = list(set(quater_random).difference(set(users_sample)))
    dumpfile(random_diff, table+'_random_diff')'''

    users_extra_sample = []
    i = 0
    for this_u in users_all:
        if this_u not in users_sample:
            i += 1
            users_extra_sample.append(this_u)
            if i==300:
                break

    dumpfile(users_extra_sample, table+'_users_extra_sample')

    return 0

def dumpfile(data, filename):
    f = open('Rome/'+filename+'.pik', 'w')
    pickle.dump(data, f)
    f.close()

    f = open('Rome/'+filename+'.csv', 'w')
    for d in data:
        f.write(str(d)+"\n")
    f.close()
    return 0

if __name__ == '__main__':
    #axes = ['geo', 'temporal']
    tables = ['twitter', 'instgram']
    for name_city in name_cities:
        i = name_cities.index(name_city)
        loc_city = loc_cities[i]

        if name_city!='Rome':
            continue

        print 'Results of: ' + name_city
        
        for table in tables:
            print_plot_stats(loc_city, name_city, table)
            #plotIndex += 1
        