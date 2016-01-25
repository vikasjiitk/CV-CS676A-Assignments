import sys
import cv2
import numpy as np
import math
kernel_h=1
kernel_window=4*kernel_h
kernel_thres=0.01
filename = sys.argv[1]
img = cv2.imread(filename)
height, width, channels = img.shape
#print height
#imgLAB = [[[0 for i in range(3)] for j in range(width)] for k in range(height)]
#imgLAB = cv2.cvtColor(img, cv2.CV_BGR2Lab)
imgLAB=img
m=1
S=1
mean = [[[j,i] for i in range(max(width,height))] for j in  range(max(width,height))]
gradp = [[[0,0] for i in range(max(width,height))] for j in range(max(width,height))]
final = [[[-1,-1] for i in range(max(width,height))] for j in range(max(width,height))]

def dist(x1,y1,x2,y2):
	Ds = math.sqrt((x2-x1)**2+(y2-y2)**2)
	Dc = math.sqrt((int(imgLAB[x1][y1][0])-int(imgLAB[x2][y2][0]))**2 + (int(imgLAB[x1][y1][1])-int(imgLAB[x2][y2][1]))**2 + (int(imgLAB[x1][y1][2])-int(imgLAB[x2][y2][2]))**2)
	return math.sqrt(Dc**2 + (m**2)*((Ds/S)**2))

def check_convergance(gx,gy):
	if abs(gx)<=1 and abs(gy)<=1:
		return 1
	else:
		return 0

def neggradientkernel(x1,y1,x2,y2):
	a=dist(x1,y1,x2,y2)
	#c=distc(x1,y1,x2,y2)
	return a*math.exp(-(a**2)/2/(kernel_h**2))/math.sqrt(8*math.pi)

def assignmode(x,y):
	#print 'assignmode %d %d'%(x,y)
	if check_convergance(gradp[x][y][0],gradp[x][y][1]):
		final[x][y][0]=x
		final[x][y][1]=y
		return [x,y]
	else:
		final[x][y]=assignmode(x+gradp[x][y][0],y+gradp[x][y][1])
		return final[x][y][0],final[x][y][1]

#shiftedpos = [[dist(j,i) for i in range(width)] for j in range(height) ]

for i in range(height):
	for j in range(width):
		val=0
		for k in range(max(0,i-kernel_window),min(height,i+kernel_window)):
			for l in range(max(0,j-kernel_window),min(width,j+kernel_window)):
				#print k,l
				grad=neggradientkernel(i,j,k,l)
				val+=grad
				#print i,j
				gradp[i][j][0]=k*grad
				gradp[i][j][1]=l*grad
		if(val<kernel_thres):
			gradp[i][j][0] = gradp[i][j][1]=0
		else:
			gradp[i][j][0] =int( gradp[i][j][0]/val - i)
			gradp[i][j][1] =int( gradp[i][j][1]/val - j)

for i in range(height):
	for j in range(width):
		print '[%d %d]'%(gradp[i][j][0], gradp[i][j][1]),
	print "\n"

for i  in range(height):
	for j in range(width):
		if(final[i][j][0]==-1):
			final[i][j]=assignmode(i,j)

print "Final Positions"
for i in range(height):
	for j in range(width):
		print '[%d %d]'%(final[i][j][0], final[i][j][1]),
	print "\n"

imgLABComp=img
for i in range(height):
	for j in range(width):
		imgLABComp[i][j][0]=imgLAB[final[i][j][0]][final[i][j][1]][0]
		imgLABComp[i][j][1]=imgLAB[final[i][j][0]][final[i][j][1]][1]
		imgLABComp[i][j][2]=imgLAB[final[i][j][0]][final[i][j][1]][2]
cv2.imshow('image',imgLABComp)
cv2.waitKey(0)
cv2.destroyAllWindows()
