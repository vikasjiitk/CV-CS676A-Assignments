import cv2
import numpy as np
import math
kernel_h=1

def dist(x,y):
	return 1

def neggradientkernel(x,y):
	a=dist(x,y)
	return dist*math.exp(-(a**2)/2/(kernel_h**2))/math.sqrt(2*math.pi)/(kernel_h**2)


img = LoadImage("./dataset/42049.jpg")
height, width,channels = img.shape
shiftedpos = [ [[j,i] for i in range(width)] for j in range(height) ]
