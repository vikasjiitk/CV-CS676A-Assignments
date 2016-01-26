import sys
import cv2
import numpy as np
import math
kernel_hs=10
kernel_hc=10
kernel_h=5
kernel_window=4*kernel_h
kernel_thres=0.1
filename = sys.argv[1]
img = cv2.imread(filename)
height, width, channels = img.shape
#print height
#imgLAB = [[[0 for i in range(3)] for j in range(width)] for k in range(height)]
imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
imgLAB=img
m=1
S=1
mean = [[[j,i] for i in range(max(width,height))] for j in  range(max(width,height))]
gradp = [[[0,0] for i in range(max(width,height))] for j in range(max(width,height))]
final = [[[-1,-1] for i in range(max(width,height))] for j in range(max(width,height))]

def distc(x1,y1,x2,y2):
	Dc = math.sqrt((int(imgLAB[x1][y1][0])-int(imgLAB[x2][y2][0]))**2 + (int(imgLAB[x1][y1][1])-int(imgLAB[x2][y2][1]))**2 + (int(imgLAB[x1][y1][2])-int(imgLAB[x2][y2][2]))**2)
	return Dc

def dists(x1,y1,x2,y2):
	Ds = math.sqrt((x2-x1)**2+(y2-y1)**2)
	return m*Ds/S

def dist(x1,y1,x2,y2):
	return math.sqrt(dists(x1,y1,x2,y2)**2 + distc(x1,y1,x2,y2)**2)

def check_convergance(gx,gy):
	if abs(gx)<=1 and abs(gy)<=1:
		return 1
	else:
		return 0

def neggradientkernel(x1,y1,x2,y2):
	Ds=dists(x1,y1,x2,y2)
	Dc=distc(x1,y1,x2,y2)
	return 0.01*math.exp(-(Ds**2)/2/(kernel_hs**2))*math.exp(-(Dc**2)/2/(kernel_hc**2))/4*math.pi
	#return a*math.exp(-(a**2)/2/(kernel_h**2))/math.sqrt(2*)/(kernel_h**2)

def assignmode(x,y):
	lis=[]
	lis.append([x,y])
	tx=x
	ty=y
	# if check_convergance(gradp[x][y][0],gradp[x][y][1]):
	# 	final[x][y][0]=x
	# 	final[x][y][1]=y
	# 	return [x,y]
	# else:
	# 	lis=[]
	while(not check_convergance(gradp[tx][ty][0],gradp[tx][ty][1])):
		lis.append([tx,ty])
		[tx,ty]=[tx+gradp[tx][ty][0],ty+gradp[tx][ty][1]]
	while(len(lis)!=0):
		[temx,temy]=lis.pop()
		final[temx][temy]=[tx,ty]
	return

print "Computing gradient"
for i in range(height):
	for j in range(width):
		val=0
		# print i,j
		for k in range(max(0,i-kernel_window),min(height,i+kernel_window)):
			for l in range(max(0,j-kernel_window),min(width,j+kernel_window)):
				# print "k  %d l %d"%(k,l)
				grad=neggradientkernel(i,j,k,l)
				# print grad
				val+=grad
				#print i,j
				gradp[i][j][0]+=k*grad
				gradp[i][j][1]+=l*grad
		if(val<kernel_thres):
			gradp[i][j][0] = gradp[i][j][1]=0
		else:
			gradp[i][j][0] =int( gradp[i][j][0]/val - i)
			gradp[i][j][1] =int( gradp[i][j][1]/val - j)

# for i in range(height):
# 	for j in range(width):
# 		print '[%d %d]'%(gradp[i][j][0], gradp[i][j][1]),
# 	print "\n"
print "Computing Mode Positions"

for i  in range(height):
	for j in range(width):
		if(final[i][j][0]==-1):
			assignmode(i,j)

# print "Final Positions"
# for i in range(height):
# 	for j in range(width):
# 		print '[%d %d]'%(final[i][j][0], final[i][j][1]),
# 	print "\n"
imgLABComp=img
for i in range(height):
	for j in range(width):
		imgLABComp[i][j][0]=imgLAB[final[i][j][0]][final[i][j][1]][0]
		imgLABComp[i][j][1]=imgLAB[final[i][j][0]][final[i][j][1]][1]
		imgLABComp[i][j][2]=imgLAB[final[i][j][0]][final[i][j][1]][2]
# cv2.imshow('image',imgLABComp)
cv2.imwrite('type1.png',imgLABComp)
for i in range(height):
	for j in range(width):
		if(final[i][j][0]==i and final[i][j][1]==j):
			val=10
			val2=[i,j]
			for k in range(max(0,i-1),min(height,i+2)):
				for l in range(max(0,j-1),min(width,j+2)):
					temp=dist(i,j,k,l)
					if(temp<val and not(k==i and l==j )):
						val=temp
						val2=[k,l]
			final[i][j]=val2


print "Computing Final Image"

imgLABComp=img
for i in range(height):
	for j in range(width):
		imgLABComp[i][j][0]=imgLAB[final[i][j][0]][final[i][j][1]][0]
		imgLABComp[i][j][1]=imgLAB[final[i][j][0]][final[i][j][1]][1]
		imgLABComp[i][j][2]=imgLAB[final[i][j][0]][final[i][j][1]][2]
# cv2.imshow('image',imgLABComp)
# cv2.waitKey(0)
cv2.imwrite('type2.png',imgLABComp)
# cv2.destroyAllWindows()
