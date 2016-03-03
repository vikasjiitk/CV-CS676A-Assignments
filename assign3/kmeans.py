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
numpoints = 0
maxlevel = 5
maxval = 10000000
maxleafno=(numclusters)**maxlevel
source_dir = '../../data/assign3/dataset/'
query_dir= '../../data/assign3/query/'
dfileList = glob.glob(source_dir + '/*.jpg')
dscore=[]
dnorm = [1 for i in range(len(dfileList))]
qnorm = 1
qfileList = glob.glob(query_dir + '/*.jpg')
node_entropy=[0 for i in range(maxleafno)]
invfilepoint=[[] for i in range(maxleafno)]
leafnodes=0

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

	# sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.1, edgeThreshold = 8)
	sift = cv2.SIFT(nfeatures=no_sift)

	no_im = 0
	for fil in fileList:
		print fil
		im = cv2.imread(fil);
		gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		(kps, descs) = sift.detectAndCompute(gray, None)
		print len(kps)
		no_sift = min(300, len(kps))
		# print no_sift
		image_points.append([i,i+no_sift])
		no_im += 1
		X[i:i+no_sift] = descs[0:no_sift]
		i += no_sift
		img = im
		cv2.drawKeypoints(im,kps[0:no_sift],img)
		cv2.imwrite('a'+fil[:-4]+'2.jpg',img)
	# print X[0:i]
	# print image_points
	return [X[0:i],image_points]

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
	[start, end] = d_image_points[filenum]
	if start==end:
		return
	Dleaf={}
	for i in range(start,end):
		ip=dX[i]
		val=maxval
		cennum=0
		for j in range(maxlevel):
			count =0
			val=maxval
			for k in range(cennum,cennum+numclusters):
				# print paramc[j][cennum][i][0]
				if(dist(ip,paramc[j][k][0])< val):
					index=k
					valcount = count
					val=dist(ip,paramc[j][k][0])
				if(paramc[j][k][1] >= numclusters):
					count += numclusters
			if(paramc[j][index][1]<numclusters):
				invfilepoint[paramc[j][index][2]].append(filenum) #changed
                		leafno = paramc[j][index][2]
                		if(leafno in Dleaf):
                    			Dleaf[leafno] += 1
                		else:
                    			Dleaf[leafno] = 1
                		break
			if(j==maxlevel-1):
				invfilepoint[paramc[j][index][2]].append(filenum)
				leafno = paramc[j][index][2]
				if(leafno in Dleaf):
					Dleaf[leafno] += 1
				else:
					Dleaf[leafno] = 1
			cennum = valcount
	dscore.append(Dleaf)

def invfilequery(filenum,image_points,X):
	leaf=[]
	Qleaf = {}
	[start, end] = image_points[filenum]
	for i in range(start,end):
		ip=X[i]
		val=maxval
		cennum=0
		for j in range(maxlevel):
			count =0
			val=maxval
			for k in range(cennum,cennum+numclusters):
				# print paramc[j][cennum][i][0]
				if(dist(ip,paramc[j][k][0])< val):
					index=k
					valcount = count
					val=dist(ip,paramc[j][k][0])
				if(paramc[j][k][1] >= numclusters):
					count += numclusters
			if(paramc[j][index][1]<numclusters):
				leaf.append(paramc[j][index][2])
				leafno = paramc[j][index][2]
				if(leafno in Qleaf):
					Qleaf[leafno] += 1
				else:
					Qleaf[leafno] = 1
				break
			if(j==maxlevel-1):
				leaf.append(paramc[j][index][2])
				leafno = paramc[j][index][2]
				if(leafno in Qleaf):
					Qleaf[leafno] += 1
				else:
					Qleaf[leafno] = 1
			cennum = valcount
	return [leaf,Qleaf]

q= Queue()
# X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(numpoints)])
[dX,d_image_points] = sift_space(dfileList)
y=dX
q.put(y)
q.put(1)
j=0
centers=[]
paramc=[]
while not(q.empty()):
	print 'hi'
	elem=q.get()
	if type(elem) is int:
		# print "new %d"%(j)
		j+=1
		q.put(j)
		if(j== maxlevel):
			for i in range(len(centers)):
				centers[i].append(leafnodes)
				leafnodes+=1
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
			elif j<maxlevel-1:
				ncenters[i].append(leafnodes)
				leafnodes+=1
			centers.append(ncenters[i])
		# print "centers"
		# print ncenters
for i in range(len(d_image_points)):
	print "invfile"
	invfile(i)

N = len(dfileList)

for i in range(maxleafno):
	print 'hello'
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
		# print Score_Dict[i]
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
	cv2.imwrite("r"+fil[:-4]+'_result.jpg',newimg)
