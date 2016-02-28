import sys
import cv2
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
filename1= sys.argv[1]            # image 1
filename2 = sys.argv[2]           # image 2
# thres_factor  = float(sys.argv[3])      # factor of threshold to be taken; threshold = thres_factor*avg(f)
thres_factor = 2
# descriptor_size = int(sys.argv[4])      # size of descriptor window
descriptor_size = 32
# H_size = int(sys.argv[5])               # size of window for which H matix to be build
H_size = 7
# max_range = int(sys.argv[6])            # range over which f is check for maximum
max_range = 15

############### Parameters to tune
# thres_factor  (parameter)
# padding
# H Matrix - H_size 5
# lamda- maximum max_range - max_range 15
# descriptor size - descriptor_size (parameters)
##################
def direction(ix,iy):                # for finding direction of gradient and classify into bins
	if(ix!=0):
		m = iy*1.0/ix
	if (iy>0 and ix>0):
		if(m <= 1.0):
			return 0
		else:
			return 1
	elif (iy>0 and ix<0):
		if(m <= -1.0):
			return 2
		else:
			return 3
	elif (iy<0 and ix<0):
		if(m <= 1.0):
			return 4
		else:
			return 5
	elif (iy<0 and ix>0):
		if(m <= -1.0):
			return 6
		else:
			return 7
	elif(ix == 0 and iy > 0):
		return 1
	elif(ix == 0 and iy < 0):
		return 4
	elif(ix > 0 and iy == 0):
		return 0
	elif(ix < 0 and iy == 0):
		return 3
	else:
		return -1
def ip(filename):
	img = cv2.imread(filename)
	height, width, channels = img.shape
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ix = [[0 for i in range(max(width,height))] for j in range(max(width,height))]
	iy = [[0 for i in range(max(width,height))] for j in range(max(width,height))]
	direc = [[0 for i in range(max(width,height))] for j in range(max(width,height))]
	f = [[0 for i in range(max(width,height))] for j in range(max(width,height))]
	im = gray_img
	for i in range(height):                             # Finding gradient for each pixel
		for j in range(width):
			if(j == 0):
				ix[i][j]=int(gray_img[i][1])-int(gray_img[i][0])
			elif (j == width-1):
				ix[i][j]=int(gray_img[i][width-1])-int(gray_img[i][width-2])
			else:
				ix[i][j]=(int(gray_img[i][j+1])-int(gray_img[i][j-1]))*1.0/2
			if(i == 0):
				iy[i][j]=int(gray_img[1][j])-int(gray_img[0][j])
			elif (i == height-1):
				iy[i][j]=int(gray_img[height-1][j])-int(gray_img[height-2][j])
			else:
				iy[i][j]=(int(gray_img[i+1][j])-int(gray_img[i-1][j]))*1.0/2
			direc[i][j]=direction(ix[i][j],iy[i][j])

	for i in range(height):                              # Calculating H matrix for each pixel
		for j in range(width):
			val1=0
			val2=0
			val3=0

			for k in range(max(0,i-H_size),min(height,i+H_size)):
				for l in range(max(0,j-H_size),min(width,j+H_size)):
					val1+=ix[k][l]**2
					val2+=ix[k][l]*iy[k][l]
					val3+=iy[k][l]**2
			if(val1+val3==0):                               # calculating harris descriptor for each pixel using H matrix
				f[i][j]=0
			else:
				f[i][j]=(val1*val3-val2**2)/(val1+val3)
			f[i][j]/=(H_size**2)

	su = 0
	for x in range(len(f)):                                   # setting threshold for harris operator
		su += sum(f[x])
	threshold = thres_factor*float(su)/len(f)/len(f[0])
	for i in range(height):                                  # selecting points above threshold
		for j in range(width):
			if(f[i][j]>threshold):
				im[i][j]=f[i][j]
			else:
				im[i][j]=0
	im2=im

	points=[]                                     # taking local maxima points
	for i in range(5,height-5):
		for j in range(5,width-5):
			if(im[i][j]==0):
				continue
			flag=0
			for k in range(max(0,i-max_range),min(height,i+max_range)):
				for l in range(max(0,j-max_range),min(width,j+max_range)):
					if(im[i][j]<im[k][l]):
						flag=1
						break
			if(flag==1):
				im2[i][j]=0
			else:
				points.append([i,j])
	ip=[]
	for i in range(len(points)):                    # creating descriptor for each point
		desc = []
		k1 = max(0,points[i][0]-descriptor_size)
		k2 = min(height,points[i][0]+descriptor_size)
		l1 = max(0,points[i][1]-descriptor_size)
		l2 = min(width,points[i][1]+descriptor_size)
		patchx = np.linspace(k1,k2,5)               # dividing into 16 patches
		patchy = np.linspace(l1,l2,5)
		for k in range(4):
			for l in range(4):
				temp=[0,0,0,0,0,0,0,0]
				for px in range(int(patchx[k]),int(patchx[k+1])):
					for py in range(int(patchy[l]), int(patchy[l+1])):
						if(direc[px][py]!=-1):
							temp[direc[px][py]]=temp[direc[px][py]]+1
				for hist in temp:
					desc.append(hist)
		ip.append([desc,points[i]])
	return [img,im2,ip]               # return original image, interest point image and list of interest points

def intersection(ip1, ip2):                            # finding match of interest points for 2 images
	neigh = NearestNeighbors(n_neighbors = 2, metric = 'l2')              # using KNN
	tr_data = []
	tr_points = []
	for i in ip2:
		tr_data.append(i[0])
		tr_points.append(i[1])
	ts_data = []
	ts_points = []
	for i in ip1:
		ts_data.append(i[0])
		ts_points.append(i[1])
	neigh.fit(tr_data)
	results = neigh.kneighbors(ts_data, 2, return_distance = True)
	match = []
	for i in range(len(results[0])):
		IP2 = tr_points[results[1][i][0]]
		IP1 = ts_points[i]
		match.append([IP1, IP2, results[0][i][0]/results[0][i][1]])
	return match                            # return list of matches along with corresponding ratio of best match to second best match

# MAIN PROGRAM
[img1,ip1image,ip1]=ip(filename1)          # find interest points of image1
[img2,ip2image,ip2]=ip(filename2)          # find interest points of image 2

matches = intersection(ip1,ip2)            # find matches between interest points of two images
matches = sorted(matches,key=lambda matches: matches[2])              # sort matches using ratio of best match to second best match


# Creating combined image to visualize
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

nWidth = w1+w2
nHeight = max(h1, h2)
hdif = (h1-h2)/2
newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
newimg[hdif:hdif+h2, :w2] = img2
newimg[:h1, w2:w1+w2] = img1
for i in range(min(len(matches),50)):                # select to 50 best match points
	pt_a = (int(matches[i][1][1]), int(matches[i][1][0]+hdif))
	pt_b = (int(matches[i][0][1]+w2), int(matches[i][0][0]))
	cv2.line(newimg, pt_a, pt_b, (0, 255, 0),1)
	cv2.circle(newimg,(matches[i][1][1],matches[i][1][0]+hdif), 3 , (0,0,255), -1)
	cv2.circle(newimg,(matches[i][0][1]+w2,matches[i][0][0]), 3, (0,0,255), -1)

# saving results
cv2.imwrite(filename1[:-4]+str(thres_factor)+str(H_size)+str(max_range)+str(descriptor_size)+'combine.png',newimg)  # combined image with matches
cv2.imwrite(filename1[:-4]+str(thres_factor)+str(H_size)+str(max_range)+str(descriptor_size)+'interestpoints.png',ip1image)  # interest points of image 1
cv2.imwrite(filename2[:-4]+str(thres_factor)+str(H_size)+str(max_range)+str(descriptor_size)+'interestpoints.png',ip2image)  # interest points of image 2
