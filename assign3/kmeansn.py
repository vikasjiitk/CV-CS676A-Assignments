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


def leaders(xs):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])

def sift_space(fileList):
	image_points = []
	no_images = len(fileList)
	no_sift = 300
	numpoints = no_images*no_sift
	X = np.zeros((numpoints,128))
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
		no_sift = min(300, len(kps))
		# print no_sift
		image_points.append([i,i+no_sift])
		no_im += 1
		X[i:i+no_sift] = descs[0:no_sift]
		i += no_sift
		img = im
		cv2.drawKeypoints(im,kps[0:no_sift],img)
		cv2.imwrite(kres_dir+fil[28:-4]+'2.jpg',img)
	# print X[0:i]
	# print image_points
	return [X[0:i],image_points]

def cluster(dat):
	# print len(dat)
	# print len(dat[0])
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
		cluscenter.append(temp/len(partition[i]))
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

def buildtree(nod, point,level):
	global leafnodes
	if(level==maxlevel):
		nod.leafnum=leafnodes
		leafnodes+=1
		return
	[nx,ccenter]=cluster(point)
	children=[]
	for i in range(len(ccenter)):
		# print len(ccenter[i])
		children.append(node(ccenter[i]))
		if(len(nx[i]) > numclusters):
			buildtree(children[i],nx[i],level+1)
		else:
			children[i].leafnum=leafnodes
			leafnodes+=1
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
		val=maxval
		while(nod.leafnum==-1):
			for i in range(len(nod.child)):
				if(dist(ip,nod.child[i].center)< val):
					index=i
					val=dist(ip,nod.child[i].center)
			nod=nod.child[index]
		leafno=nod.leafnum
		invfilepoint[leafno].append(filenum)
		if(leafno in Dleaf):
			Dleaf[leafno] += 1
		else:
			Dleaf[leafno] = 1
	dscore.append(Dleaf)

def invfilequery(filenum,image_points,X):
	global leafnodes
	leaf=[]
	Qleaf = {}
	[start, end] = image_points[filenum]
	for i in range(start,end):
		nod=yn
		ip=dX[i]
		val=maxval
		while(nod.leafnum==-1):
			for i in range(len(nod.child)):
				if(dist(ip,nod.child[i].center)< val):
					index=i
					val=dist(ip,nod.child[i].center)
			nod=nod.child[index]
		leafno=nod.leafnum
		leaf.append(leafno)
		if(leafno in Qleaf):
			Qleaf[leafno] += 1
		else:
			Qleaf[leafno] = 1
	return [leaf,Qleaf]

# X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(numpoints)])
[dX,d_image_points] = sift_space(dfileList)
y=dX
# print dX
yn = node([])
buildtree(yn,y,0)
for i in range(len(d_image_points)):
	invfile(i)
# for i in range(leafnodes):
# 	print invfilepoint[i]

N = len(dfileList)
for i in range(maxleafno):
	# print 'hello'
	dimages = leaders(invfilepoint[i])
	Ni = len(dimages)+1
	if Ni!=0:
		node_entropy[i] = math.log(float(N)/Ni)+1

for i in range(len(dfileList)):
	Dict = dscore[i]
	print Dict
	norm = 0
	for j in Dict.keys():
		norm += (Dict[j]*node_entropy[j])**2
	dnorm[i] = norm
	print dnorm[i]

for fil in qfileList:
	# print "I am here"
	arg = []
	arg.append(fil)
	Score_Dict = {}
	[qX,q_image_points] = sift_space(arg)
	[qleafs,qdict] = invfilequery(0,q_image_points,qX)
	print qdict
	qleafs = leaders(qleafs)
	qnorm = 0
	for i in qdict.keys():
		qnorm += (qdict[i]*node_entropy[i])**2

	for i in qleafs:
		leafnode = i[0]
		#N = len(dfileList)
		dimages = leaders(invfilepoint[leafnode])
		qi = node_entropy[i[0]]*i[1]/qnorm
		for j in dimages:
			di = node_entropy[j[0]]*j[1]/dnorm[j[0]]
			if j[0] in Score_Dict:
				Score_Dict[j[0]] += qi*di
			else:
				Score_Dict[j[0]] = qi*di
	rscore = 0
	for i in Score_Dict.keys():
		print Score_Dict[i]
		if(rscore < Score_Dict[i]):
			rimage = i
			rscore = Score_Dict[i]

	img2 = cv2.imread(fil)
	img1 = cv2.imread(dfileList[rimage])

	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	print h1,h2,w1,w2
	nWidth = w1+w2
	nHeight = max(h1, h2)
	hdif = abs(h1-h2)/2
	newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
	newimg[hdif:hdif+h2, :w2] = img2
	newimg[:h1, w2:w1+w2] = img1
	# print "hi"
	cv2.imwrite(res_dir+fil[26:-4]+'_result.jpg',newimg)
