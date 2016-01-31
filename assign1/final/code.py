################
# Mean Shift Based Image Segmentation
# CS676A - Computer Vision Assignment 1
# Shubham Jain  - 13683
# Vikas Jain - 13788
# January 31, 2016
#
# Two kernels are implemented: Gaussian Kernel and Flat Kernel
# Given code runs gaussian kernel with file name and kernel bandwidths passed as command line arguments
#  e.g python meanshift.py filename 10 10
# For Flat kernel comment and uncomment certain lines indicated below and pass filename and kernel bandwidth as command line arguments
#  e.g python meanshift.py filename 30

 ##################
import sys
import cv2
import numpy as np
import math

kernel_window=40   # vicinity in which gradient of each pixel is computed
kernel_thres=1.1
filename = sys.argv[1]

# bandwidth for flat kernel
# uncomment for flat kernel
#flat_kernel_h=int(sys.argv[2])

# bandwidths for gaussian kernel
# comment out below 2 statements for flat kernel
kernel_hs=int(sys.argv[2])
kernel_hc=int(sys.argv[3])

img = cv2.imread(filename)
height, width, channels = img.shape

imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
imgLAB=img

# uncomment for flat kernel
# m=1
# S=20

# comment out below 2 statements for flat kernel
m=1
S=1

lis = [[0,0] for j in  range(width*height)]
modes = []

gradp = [[[0,0] for i in range(max(width,height))] for j in range(max(width,height))]
final = [[[-1,-1] for i in range(max(width,height))] for j in range(max(width,height))]
modes_count = [[[0,[]] for i in range(max(width,height))] for j in range(max(width,height))]

# Colour Distance between two pixels
def distc(x1,y1,x2,y2):
	Dc = math.sqrt((int(imgLAB[x1][y1][0])-int(imgLAB[x2][y2][0]))**2 + (int(imgLAB[x1][y1][1])-int(imgLAB[x2][y2][1]))**2 + (int(imgLAB[x1][y1][2])-int(imgLAB[x2][y2][2]))**2)
	return Dc

# Spacial Distance between two pixels
def dists(x1,y1,x2,y2):
	Ds = math.sqrt((x2-x1)**2+(y2-y1)**2)
	return m*Ds/S

# cumilative distance between two pixels
def dist(x1,y1,x2,y2):
	return math.sqrt((m/S)**2*dists(x1,y1,x2,y2)**2 + distc(x1,y1,x2,y2)**2)

def check_convergance(gx,gy):
	if (math.sqrt(gx**2+gy**2)<1):
		return 0
	else:
		return 1

# Gaussian Kernel
def neggradientkernel(x1,y1,x2,y2):
	Ds=dists(x1,y1,x2,y2)
	Dc=distc(x1,y1,x2,y2)

	return math.exp(-(Ds**2)/2/(kernel_hs**2))*math.exp(-(Dc**2)/2/(kernel_hc**2))/4*math.pi


# Flat Kernel
def flatkernel(x1,y1,x2,y2):
	if ( dist(x1,y1,x2,y2)/flat_kernel_h <= 1):
		return 1
	else:
		return 0

# Procedure for merging modes
def mergemode(tx,ty):
	val=10
	val2=[tx,ty]
	for k in range(max(0,tx-kernel_window),min(height,tx+kernel_window)):
		for l in range(max(0,ty-kernel_window),min(width,ty+kernel_window)):
			temp=dist(tx,ty,final[k][l][0],final[k][l][1])
			if(temp<val and not(final[k][l][0]==tx and final[k][l][1]==ty )):
				val=temp
				val2=final[k][l]
	if(val2 == [tx,ty]):
		return
	modes_count[val2[0]][val2[1]][0]+=modes_count[tx][ty][0]
	modes_count[tx][ty][0]=0
	while len(modes_count[tx][ty][1]):
		temp=modes_count[tx][ty][1].pop()
		final[temp[0]][temp[1]]=val2
		modes_count[val2[0]][val2[1]][1].append(temp)
	modes.remove([tx,ty])
	return

# Procedure for assigning modes
def assignmode(x,y):
	i=0
	lis[i]=[x,y]
	i=i+1
	tx=x
	ty=y
	while(check_convergance(gradp[tx][ty][0],gradp[tx][ty][1])):
		if(i>height and [tx,ty] in lis):
			break
		lis[i]=[tx,ty]
		i=i+1
		[tx,ty]=[tx+gradp[tx][ty][0],ty+gradp[tx][ty][1]]
		if (final[tx][ty][0]!=-1):
			[tx,ty]=final[tx][ty]
			break
	val2=[tx,ty]

	for j in range(i):
		[temx,temy]=lis[j]
		final[temx][temy]=val2
	return

# computing gradient of each pixel
print "Computing gradient"
for i in range(height):
	for j in range(width):
		val=0
		# print i,j
		for k in range(max(0,i-kernel_window),min(height,i+kernel_window)):
			for l in range(max(0,j-kernel_window),min(width,j+kernel_window)):

				# comment out below line for flat kernel
				grad = neggradientkernel(i,j,k,l)

				# uncomment below line for flat kernel
				# grad = flatkernel(i,j,k,l)
				# print grad
				val+=grad

				gradp[i][j][0]+=k*grad
				gradp[i][j][1]+=l*grad
		if(val<kernel_thres):
			gradp[i][j][0] = gradp[i][j][1]=0
		else:
			gradp[i][j][0] =int( gradp[i][j][0]/val - i)
			gradp[i][j][1] =int( gradp[i][j][1]/val - j)

print "Computing Mode Positions"

for i  in range(height):
	for j in range(width):
		if(final[i][j][0]==-1):
			assignmode(i,j)

for i  in range(height):
	for j in range(width):
		if (not (final[i][j] in modes)):
			modes.append(final[i][j])
			modes_count[final[i][j][0]][final[i][j][1]][0] = 1
		else:
			modes_count[final[i][j][0]][final[i][j][1]][0] += 1
		modes_count[final[i][j][0]][final[i][j][1]][1].append([i,j])

print 'Initial number of modes: ' + str(len(modes))
imgLABComp=img

for i in range(height):
	for j in range(width):
		imgLABComp[i][j][0]=imgLAB[final[i][j][0]][final[i][j][1]][0]
		imgLABComp[i][j][1]=imgLAB[final[i][j][0]][final[i][j][1]][1]
		imgLABComp[i][j][2]=imgLAB[final[i][j][0]][final[i][j][1]][2]

# comment out line below for flat kernel
cv2.imwrite(filename[:-4]+'type1'+str(kernel_hs)+'_'+str(kernel_hc)+'.png',imgLABComp)

#uncomment below line for flat kernel
# cv2.imwrite(filename[:-4]+'type1'+str(flat_kernel_h)+'.png',imgLABComp)

for i  in range(height):
	for j in range(width):
		if(modes_count[i][j][0]<20 and modes_count[i][j][0] !=0):
			mergemode(i,j)

print "Computing Final Image"
print 'Final number of modes: ' + str(len(modes))
imgLABComp=img
for i in range(height):
	for j in range(width):
		imgLABComp[i][j][0]=imgLAB[final[i][j][0]][final[i][j][1]][0]
		imgLABComp[i][j][1]=imgLAB[final[i][j][0]][final[i][j][1]][1]
		imgLABComp[i][j][2]=imgLAB[final[i][j][0]][final[i][j][1]][2]

# comment out line below for flat kernel
cv2.imwrite(filename[:-4]+'type2'+str(kernel_hs)+'_'+str(kernel_hc)+'.png',imgLABComp)

#uncomment below line for flat kernel
# cv2.imwrite(filename[:-4]+'type2'+str(flat_kernel_h)+'.png',imgLABComp)
