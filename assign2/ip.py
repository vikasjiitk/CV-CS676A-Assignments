import sys
import cv2
import numpy as np
import math
filename = sys.argv[1]
img = cv2.imread(filename)
window=4
threshold=8000
height, width, channels = img.shape
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ix = [[0 for i in range(max(width,height))] for j in range(max(width,height))]
iy = [[0 for i in range(max(width,height))] for j in range(max(width,height))]
direc = [[0 for i in range(max(width,height))] for j in range(max(width,height))]
f = [[0 for i in range(max(width,height))] for j in range(max(width,height))]
im = gray_img
# cv2.imshow('image',gray_img)
for i in range(height):
	for j in range(width):
		if(j == 0):
			ix[i][j]=int(gray_img[i][1])-int(gray_img[i][0])
		elif (j == width-1):
			ix[i][j]=int(gray_img[i][width-1])-int(gray_img[i][width-2])
		else:
			ix[i][j]=(int(gray_img[i][j+1])-int(gray_img[i][j-1]))/2
		if(i == 0):
			iy[i][j]=int(gray_img[1][j])-int(gray_img[0][j])
		elif (i == height-1):
			iy[i][j]=int(gray_img[height-1][j])-int(gray_img[height-2][j])
		else:
			iy[i][j]=(int(gray_img[i+1][j])-int(gray_img[i-1][j]))/2
		direc[i][j]=direction(ix[i][j],iy[i][j])
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
		if(f[i][j]>threshold):
			im[i][j]=f[i][j]
		else:
			im[i][j]=0
		# print f[i][j]

# print max(max(f))
# data=np.array([f[x/height][x%width] for x in range(height*width)])
# hist,bins=np.histogram(data,bins=np.linspace(0,27000,400))
# print(hist)
im2=im
# for i in range(1,height-1):
# 	for j in range(1,width-1):
# 		im[i][j]=(im[i][j-1]+im[i][j+1]+im[i+1][j-1]+im[i+1][j]+im[i+1][j+1]+im[i-1][j-1]+im[i-1][j+1]+im[i-1][j]+im[i][j])/9
windo=10
ip=[]
for i in range(1,height-1):
	for j in range(1,width-1):
		temp=[0,0,0,0,0,0,0,0]
		flag=0
		for k in range(max(0,i-windo),min(height,i+windo)):
			for l in range(max(0,j-windo),min(width,j+windo)):
				if(im[i][j]<im[k][l]):
					flag=1
					break
		if(flag==1):
			im2[i][j]=0
		else:

		# 	im2[i][j]=1
# cv2.imshow('image',im)
cv2.imwrite(filename[:-4]+'lam.png',im)
cv2.imwrite(filename[:-4]+'lam1.png',im2)
# cv2.waitKey(0)