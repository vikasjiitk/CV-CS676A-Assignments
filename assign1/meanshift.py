import cv2
import numpy as np
import math
kernel_h=1
img = cv2.imread("./dataset/42049.jpg")
height, width,channels = img.shape

def dist(x,y):
	return 1

def neggradientkernel(x,y):
	a=dist(x,y)
	return dist*math.exp(-(a**2)/2/(kernel_h**2))/math.sqrt(2*math.pi)/(kernel_h**2)

def meanshift(x):
	return 1
	#find mean shift

#shiftedpos = [[dist(j,i) for i in range(width)] for j in range(height) ]
mean = [[[j,i] for i in range(1,width)] for j in range(1,height) ]

for i in range(1,height):
	for j in range(1,width):
		temp[i,j]=0
		for k in range(1,height):
			for l in range(1,width):
				shiftedpos[i,j]=
print shiftedpos