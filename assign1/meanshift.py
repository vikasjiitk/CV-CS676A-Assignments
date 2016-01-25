import cv2
import numpy as np
import math
kernel_h=1
kernel_window=4*kernel_h
img = cv2.imread("./dataset/42049small.jpg")
height, width, channels = img.shape
#imgLAB = [[[0 for i in range(3)] for j in range(width)] for k in range(height)]
#imgLAB = cv2.cvtColor(img, cv2.CV_BGR2Lab)
imgLAB=img
m=1
S=1
mean = [[[j,i] for i in range(max(width,height))] for j in  range(max(width,height))]
gradp = [[[0,0] for i in range(max(width,height))] for j in range(max(width,height))]

def dist(x1,y1,x2,y2):
	Ds = math.sqrt((x2-x1)**2+(y2-y2)**2)
	Dc = math.sqrt((imgLAB[x1][y1][0]-imgLAB[x2][y2][0])**2 + (imgLAB[x1][y1][1]-imgLAB[x2][y2][1])**2 + (imgLAB[x1][y1][2]-imgLAB[x2][y2][2])**2)
	return math.sqrt(Ds**2 + (m**2)*((Ds/S)**2))


def neggradientkernel(x1,y1,x2,y2):
	a=dist(x1,y1,x2,y2)
	return a*math.exp(-(a**2)/2/(kernel_h**2))/math.sqrt(2*math.pi)/(kernel_h**2)

def assignmode(x,y):
	if check_convergance(gradp[x],gradp[y]):
		return [x,y]
	else
		[i,j]=assignmode(x+gradp[x],y+gradp[y])
		return [i,j];

#shiftedpos = [[dist(j,i) for i in range(width)] for j in range(height) ]

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
for i in range(height):
	for j in range(width):
		print '[%.2f %.2f]'%(gradp[i][j][0], gradp[i][j][1]),
	print "\n"

final = [[[-1,-1] for i in range(max(width,height))] for j in range(max(width,height))]

for i  in range(height):
	for j in range(width):
		if(final[i][j]==-1):
			final[i][j]=assignmode(i,j);