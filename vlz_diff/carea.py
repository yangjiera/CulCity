import numpy as np
import math
import os
import sys
import string
import re
import urllib
import csv

def get_id(urltext):
    id_regx = re.compile('id=.+')
    ids = id_regx.findall(urltext)
    id = ids[0][3:]
    return id

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
class Rect(object):
    def __init__(self, p1, p2):
        self.left   = min(p1.x, p2.x)
        self.right  = max(p1.x, p2.x)
        self.bottom = min(p1.y, p2.y)
        self.top    = max(p1.y, p2.y)
def overlap(r1, r2):
    return range_overlap(r1.left, r1.right, r2.left, r2.right) and range_overlap(r1.bottom, r1.top, r2.bottom, r2.top)
def range_overlap(a_min, a_max, b_min, b_max):
    return not ((a_min >= b_max) or (b_min >= a_max))

def center_dist(bb1, bb2):
    c1 = [float(bb1[2]+bb1[0])/2, float(bb1[3]+bb1[1])/2]
    c2 = [float(bb2[2]+bb2[0])/2, float(bb2[3]+bb2[1])/2]
    dif_c = np.divide(np.array(c1) - np.array(c2), np.array([bb1[2]-bb1[0], bb1[3]-bb1[1]]))
    return np.linalg.norm(dif_c)

'''def center_dist2(bb1, bb2):
    c1 = [float(bb1[2]+bb1[0])/2, float(bb1[3]+bb1[1])/2]
    c2 = [float(bb2[2]+bb2[0])/2, float(bb2[3]+bb2[1])/2]
    dist_vec = np.array(c1)-np.array(c2)
    dist_vec[0] = float(dist_vec[0])/c1
    return np.linalg.norm()'''
def area_overlapping_jaccard(bb1, bb2):
    ao = area_overlapping(bb1, bb2)
    size1 = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    size2 = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    return ao/(size1+size2-ao)

def area_overlapping(bb1, bb2):
    p1 = Point(bb1[0], bb1[1])
    p2 = Point(bb1[2], bb1[3])
    r1 = Rect(p1,p2)
    p3 = Point(bb2[0], bb2[1])
    p4 = Point(bb2[2], bb2[3])
    r2 = Rect(p3,p4)
    
    if overlap(r1, r2):
        if r1.left>r2.left and r1.left<r2.right:
            if r1.right>r2.right:
                length = r2.right - r1.left
            else:
                length = r1.right - r1.left
        else:
            if r2.right>r1.right:
                length = r1.right - r2.left
            else:
                length = r2.right - r2.left
        if r1.bottom>r2.bottom and r1.bottom<r2.top:
            if r1.top>r2.top:
                width = r2.top - r1.bottom
            else:
                width = r1.top - r1.bottom
        else:
            if r2.top>r1.top:
                width = r1.top - r2.bottom
            else:
                width = r2.top - r2.bottom
        return length*width
    else:
        return -1
    
