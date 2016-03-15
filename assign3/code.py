import random
import glob
import cv2
import math
import sys
from sklearn.cluster import KMeans
from multiprocessing import Queue
from collections import defaultdict
import numpy as np
from Queue import *
numclusters = int(sys.argv[2])
numpoints = 1000
maxlevel = int(sys.argv[1])
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
	def __init__(self):
		self.child=[]
		self.model=None
		self.leafnum=-1
	def chil(self,lis):
		for x in lis:
			self.child.append(x)

def leaders(xs):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])


#Sift Space compute for the file on file list and returns the sift point and the partition of what belongs to whoch file number
def sift_space(fileList):
	image_points = []
	no_images = len(fileList)
	no_sift = 300
	X = np.zeros((no_images*no_sift,128))
	i = 0
	#Uncomment the next line for CV 3.x and comment the one after that
	# sift = cv2.xfeatures2d.SIFT_create(nfeatures=no_sift)
	sift = cv2.SIFT(nfeatures=no_sift)

	no_im = 0
	for fil in fileList:
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
	return [X[0:i],image_points]

#cluster returns the dat clustered with the kmean function
def cluster(dat):
	kmean=KMeans(init='k-means++', n_clusters=numclusters, n_init=10)
	y=kmean.fit_predict(dat)
	partition=[[] for i in range(numclusters)]
	for i in range(len(dat)):
		partition[y[i]].append(dat[i])
	return [partition,kmean]


#build the hierarichical kmeans tree
def buildtree(nod, point,level):
	global leafnodes
	if(level==maxlevel):
		nod.leafnum=leafnodes
		leafnodes+=1
		return
	[nx,model]=cluster(point)
	nod.model=model
	children=[]
	for i in range(len(nx)):
		children.append(node())
		if(len(nx[i]) > numclusters):
			buildtree(children[i],nx[i],level+1)
		else:
			children[i].leafnum=leafnodes
			leafnodes+=1
	nod.chil(children)

#this add the inverse file pointer at the leaf
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
			index=nod.model.predict(ip)
			nod=nod.child[index]
		leafno=nod.leafnum
		invfilepoint[leafno].append(filenum)
		if(leafno in Dleaf):
			Dleaf[leafno] += 1
		else:
			Dleaf[leafno] = 1
	dscore.append(Dleaf)


#this returns the leaf for the query file and the distribution in leafs
def invfilequery(image_points,X):
	global leafnodes
	leaf=[]
	Qleaf = {}
	[start, end] = image_points[0]
	for i in range(start,end):
		nod=yn
		ip=X[i]
		while(nod.leafnum==-1):
			index=nod.model.predict(ip)
			nod=nod.child[index]
		leafno=nod.leafnum
		leaf.append(leafno)
		if(leafno in Qleaf):
			Qleaf[leafno] += 1
		else:
			Qleaf[leafno] = 1
	return [leaf,Qleaf]

[dX,d_image_points] = sift_space(dfileList)
y=dX
yn = node()
buildtree(yn,y,0)

for i in range(len(d_image_points)):
	invfile(i)


N = len(dfileList)
for i in range(maxleafno):
	dimages = leaders(invfilepoint[i])
	Ni = len(dimages)+1
	if Ni!=0:
		node_entropy[i] = math.log(float(N)/Ni)+1

for i in range(len(dfileList)):
	Dict = dscore[i]
	norm = 0
	for j in Dict.keys():
		norm += (Dict[j]*node_entropy[j])**2
	dnorm[i] = math.sqrt(norm)

for fil in qfileList:
	arg = []
	arg.append(fil)
	Score_Dict = {}
	[qX,q_image_points] = sift_space(arg)
	[qleafs,qdict] = invfilequery(q_image_points,qX)
	qleafs = leaders(qleafs)
	qnorm = 0
	for i in qdict.keys():
		qnorm += (qdict[i]*node_entropy[i])**2
	qnorm  = math.sqrt(qnorm)
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
		if(rscore < Score_Dict[i]):
			rimage = i
			rscore = Score_Dict[i]

	img2 = cv2.imread(fil)
	img1 = cv2.imread(dfileList[rimage])

	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	nWidth = w1+w2
	nHeight = max(h1, h2)
	hdif = abs(h1-h2)/2
	newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
	newimg[hdif:hdif+h2, :w2] = img2
	newimg[:h1, w2:w1+w2] = img1
	cv2.imwrite(res_dir+fil[26:-4]+'_'+str(maxlevel)+'_'+str(numclusters)+'_result.jpg',newimg)
