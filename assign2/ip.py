import sys
import cv2
import numpy as np
import math
filename1= sys.argv[1]
filename2 = sys.argv[2]
window=4
threshold=1200
# cv2.imshow('image',gray_img)
def direction(ix,iy):
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
	for i in range(height):
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
	# print direc
	for i in range(height):
		for j in range(width):
			val1=0
			val2=0
			val3=0
			# print i,j
			for k in range(max(0,i-window),min(height,i+window)):
				for l in range(max(0,j-window),min(width,j+window)):
					val1+=ix[k][l]**2
					val2+=ix[k][l]*iy[k][l]
					val3+=iy[k][l]**2
			if(val1+val3==0):
				f[i][j]=0
			else:
				f[i][j]=(val1*val3-val2**2)/(val1+val3)
			f[i][j]/=(window**2)
			if(f[i][j]>threshold):
				im[i][j]=f[i][j]
			else:
				im[i][j]=0
			# print f[i][j]

	# print max(max(f))
	# data=np.array([f[x/height][x%width] for x in range(height*width)])
	# hist,bins=np.histogram(data,bins=np.linspace(0,max(max(f)),max(max(f))/20))
	# print(hist)
	im2=im
	# for i in range(1,height-1):
	# 	for j in range(1,width-1):
	# 		im[i][j]=(im[i][j-1]+im[i][j+1]+im[i+1][j-1]+im[i+1][j]
	# +im[i+1][j+1]+im[i-1][j-1]+im[i-1][j+1]+im[i-1][j]+im[i][j])/9
	windo=8
	points=[]
	for i in range(1,height-1):
		for j in range(1,width-1):
			if(im[i][j]==0):
				continue
			flag=0
			for k in range(max(0,i-windo),min(height,i+windo)):
				for l in range(max(0,j-windo),min(width,j+windo)):
					if(im[i][j]<im[k][l]):
						flag=1
						break
			if(flag==1):
				im2[i][j]=0
			else:
				points.append([i,j])
		# 	im2[i][j]=1
	# print len(points)
	ip=[]
	windo2=10
	for i in range(len(points)):
		temp=[0,0,0,0,0,0,0,0]
		for k in range(max(0,points[i][0]-windo2),min(height,points[i][0]+windo2)):
			for l in range(max(0,points[i][1]-windo2),min(width,points[i][1]+windo2)):
				if(direc[k][l]!=-1):
					temp[direc[k][l]]=temp[direc[k][l]]+1
		# for j in range(len(temp)):
		# 	temp[j]=temp[j]*1.0/(windo2**2)/4
		# print temp
		ip.append([temp,points[i]])
	return [im2,ip]

def SSD(hog1, hog2):
	ssd = 0
	for i in range(len(hog1)):
		ssd += (hog1[i] - hog2[i])**2
	return ssd

def intersection(ip1, ip2):
	match = []
	no_ip1 = 0
	for i in ip1:
		fl=0
		lssd = 1280
		no_ip2 = 0
		for j in ip2:
			ssd = SSD(i[0],j[0])
			if (ssd < lssd):
				lssd = ssd
				temp=j
				fl=1
			no_ip2 += 1
		no_ip1 += 1
		if(fl==1):
			match.append([i[1],temp[1]])
	return match


# cv2.imshow('image',im)
[ip1image,ip1]=ip(filename1)
[ip2image,ip2]=ip(filename2)
# for x in ip1:
# 	print x
match = intersection(ip1,ip2)
for x in match:
	print x
# print match
cv2.imwrite(filename1[:-4]+'lam.png',ip1image)
cv2.imwrite(filename2[:-4]+'lam.png',ip2image)
# cv2.waitKey(0)
