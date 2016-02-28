import random
import glob
import cv2
from sklearn.cluster import KMeans
from multiprocessing import Queue
import numpy as np
from Queue import *
numclusters = 4
numpoints = 0
maxlevel = 10
maxval=10000000
source_dir = 'dataset/'
fileList = glob.glob(source_dir + '/*.jpg')
image_points = []
def sift_space():
	no_images = len(fileList)
	no_sift = 50
	numpoints = no_images*no_sift
	X = np.zeros((numpoints,128))
	i = 0
	# sift = cv2.xfeatures2d.SIFT_create()
	sift = cv2.SIFT()
	no_im = 0
	for fil in fileList:
		# print fil
		im = cv2.imread(fil);
		gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		(kps, descs) = sift.detectAndCompute(gray, None)
		# print len(kps)
		no_sift = min(50, len(kps))
		# print no_sift
		# global image_points
		image_points.append([i,i+no_sift])
		no_im += 1
		X[i:i+no_sift] = descs[0:no_sift]
		i += no_sift
		img = im
		cv2.drawKeypoints(im,kps[0:no_sift],img)
		cv2.imwrite('a'+fil[:-4]+'2.jpg',img)
	# print X[0:i]
	# print image_points
	return [X,i]

def cluster(dat):
	if(numclusters>len(dat)):
		return [[dat], [dat[0]]]
	kmean=KMeans(init='k-means++', n_clusters=numclusters, n_init=10)
	y=kmean.fit_predict(dat)
	# print kmean
	partition=[[] for i in range(numclusters)]
	for i in range(len(dat)):
		partition[y[i]].append(dat[i])
	# for x in partition:
	# 	print x
	cluscenter=[]
	for i in range(numclusters):
		temp=partition[i][0]
		for j in range(1,len(partition[i])):
			temp+= partition[i][j]
		# temp=temp-partition[i][0]
		cluscenter.append([temp/len(partition[i]),len(partition[i])])
		# cluscenter.append(['center',len(partition[i])])
		# print len(partition[i])
	return [partition,cluscenter]

def dist(ip1, ip2):
	su=0
	# print len(ip1)
	# print len(ip2)
	for i in range(len(ip1)):
		su = abs(ip1[i]-ip2[i])
	return su

def invfile(filenum):
	[start, end] = image_points[filenum]
	for i in range(start,end):
		ip=X[i]
		val=maxval
		for j in range(len(paramc[0][0])):
			if(dist(ip,paramc[0][0][j][0])< val):
				cennum=j
				val=dist(ip,paramc[0][0][j][0])
		if(paramc[0][0][cennum][1]<numclusters):
			paramc[0][0][cennum][2].append(i)
			break
		for j in range(1,maxlevel):
			val=maxval
			for k in range(len(paramc[j][cennum])):
				# print paramc[j][cennum][i][0]
				if(dist(ip,paramc[j][cennum][k][0])< val):
					cennumnew=k
					val=dist(ip,paramc[j][cennum][k][0])
			if(paramc[j][cennum][cennumnew][1]<numclusters):
				paramc[j][cennum][cennumnew][2].append(i)
				break
			if(j==maxlevel-1):
				paramc[j][cennum][cennumnew][2].append(i)
			cennum = cennumnew

q= Queue()
# X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(numpoints)])
[X,i] = sift_space()
y=X[0:i]
q.put(y)
q.put(1)
j=0
centers=[]
paramc=[]
while not(q.empty()):
	# print 'hi'
	elem=q.get()
	if type(elem) is int:
		j+=1
		q.put(j)
		if(j> maxlevel):
			for i in range(len(centers)):
				centers[i].append([])
			while not(q.empty()):
				elem=q.get()
			# 	if type(elem) is list:
			# 		elem.append([])
		paramc.append(centers)
		# print "\nupdated paramc\n"
		# print paramc
		centers=[]
	else:
		[nX,ncenters] = cluster(elem)
		# print j
		for i in range(len(nX)):
			x=nX[i]
			if(len(x) > numclusters):
				q.put(x)
			else:
				ncenters[i].append([])
		centers.append(ncenters)
		# print ncenters
for i in range(len(image_points)):
	invfile(i)
# print type(x)
