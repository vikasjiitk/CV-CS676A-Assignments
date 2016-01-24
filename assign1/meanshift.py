import cv2
import numpy as np
import math
kernel_h=1
img = cv2.imread("./dataset/42049small.jpg")
height, width, channels = img.shape

def dist(x1,y1,x2,y2):
	return 1

def neggradientkernel(x1,y1,x2,y2):
	a=dist(x1,y1,x2,y2)
	return a*math.exp(-(a**2)/2/(kernel_h**2))/math.sqrt(2*math.pi)/(kernel_h**2)

def meanshift(x):
	return 1
	#find mean shift

#shiftedpos = [[dist(j,i) for i in range(width)] for j in range(height) ]
mean = [[[j,i] for i in range(max(width,height))] for j in  range(max(width,height))]
gradp = [[[0,0] for i in range(max(width,height))] for j in range(max(width,height))]
gradp[2][1][0]
for i in range(height):
	for j in range(width):
		val=0
		for k in range(height):
			for l in range(width):
				#print k,l
				grad=neggradientkernel(i,j,k,l)
				val+=grad
				#print i,j
				gradp[i][j][0]=k*grad
				gradp[i][j][1]=l*grad
		gradp[i][j][0] = gradp[i][j][0]/val - i
		gradp[i][j][1] = gradp[i][j][1]/val - j
print gradp