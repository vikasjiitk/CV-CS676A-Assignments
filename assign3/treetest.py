import random
import glob
import cv2
import math
from sklearn.cluster import KMeans
from multiprocessing import Queue
from collections import defaultdict
import numpy as np
from Queue import *
numclusters = 4
numpoints = 1000
maxlevel = 5
maxval = 10000000
maxleafno=(numclusters)**maxlevel
source_dir = '../../data/assign3/dataset/'
kres_dir = '../../data/assign3/adataset/'
query_dir= '../../data/assign3/query/'
res_dir = '../../data/assign3/rquery/'
dfileList = glob.glob(source_dir + '/*.jpg')
dscore=[]
dnorm = [1 for i in range(len(dfileList))]
qnorm = 1
qfileList = glob.glob(query_dir + '/*.jpg')
node_entropy=[0 for i in range(maxleafno)]
invfilepoint=[[] for i in range(maxleafno)]
leafnodes=0
leafde=[0 for i in range(maxleafno)]
leafre=[0 for i in range(maxleafno)]

class node():
	def __init__(self, cen):
		self.center=cen
		# print len(cen)
		self.child=[]
		# self.filenumList=[]
		self.leafnum=-1
	def chil(self,lis):
		for x in lis:
			self.child.append(x)

def sift_space(fileList):
	image_points = []
	no_images = len(fileList)
	no_sift = 300
	X = np.zeros((no_images*no_sift,128))
	i = 0

	# sift = cv2.xfeatures2d.SIFT_create(nfeatures=no_sift)
	sift = cv2.SIFT(nfeatures=no_sift)

	no_im = 0
	for fil in fileList:
		# print fil
		im = cv2.imread(fil);
		gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		(kps, descs) = sift.detectAndCompute(gray, None)
		# print len(kps)
		no_siftn = min(300, len(kps))
		# print no_sift
		image_points.append([i,i+no_siftn])
		no_im += 1
		X[i:i+no_siftn] = descs[0:no_siftn]
		i += no_siftn
		img = im
		cv2.drawKeypoints(im,kps[0:no_siftn],img)
		cv2.imwrite(kres_dir+fil[28:-4]+'2.jpg',img)
	# print X[0:i]
	# print image_points
	return [X[0:i],image_points]

	# if(numclusters>len(dat)):
	# 	return [[dat], [dat[0]]]

def cluster(dat):
	kmean=KMeans(init='k-means++', n_clusters=numclusters, n_init=10)
	y=kmean.fit_predict(dat)
	partition=[[] for i in range(numclusters)]
	for i in range(len(dat)):
		partition[y[i]].append(dat[i])
	return [partition,kmean.cluster_centers_.tolist()]

def dist(ip1, ip2):
	su=0
	for i in range(len(ip1)):
		su = (ip1[i]-ip2[i])**2
	return math.sqrt(su)

def buildtree(nod, point,level):
	global leafnodes
	if(level==maxlevel):
		nod.leafnum=leafnodes
		leafnodes+=1	 # print "num leaf node %d"%leafnodes # print "number of points in leaf %d"%(len(point))
		return
	[nx,ccenter]=cluster(point)
	children=[]
	for i in range(len(ccenter)):
		children.append(node(ccenter[i]))
		if(len(nx[i]) > numclusters):
			buildtree(children[i],nx[i],level+1)
		else:
			children[i].leafnum=leafnodes
			leafde[leafnodes]=len(nx[i])
			leafnodes+=1	# print "num leaf node %d"%leafnodes # print "number of points in leaf %d"%(len(nx[i]))
	nod.chil(children)

def invfile(filenum):
	global leafnodes
	[start, end] = d_image_points[filenum]
	if start==end:
		return
	Dleaf={}
	for i in range(start,end):
		nod=yn
		ip=dX[i]
		while(nod.leafnum==-1):
			val=maxval
			for j in range(len(nod.child)):
				if(dist(ip,nod.child[j].center)< val):
					index=j
					val=dist(ip,nod.child[j].center)
			nod=nod.child[index]
		leafno=nod.leafnum
		leafre[leafno]+=1
		invfilepoint[leafno].append(filenum)
		if(leafno in Dleaf):
			Dleaf[leafno] += 1
		else:
			Dleaf[leafno] = 1
	dscore.append(Dleaf)

# X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(numpoints)])
[dX,d_image_points] = sift_space(dfileList)
y=dX
# print dX
yn = node([])
buildtree(yn,y,0)
for i in range(len(d_image_points)):
	invfile(i)
for i in range(leafnodes):
	# print leafre[i]
	# print leafde[i]
	print invfilepoint[i]