import sys
import cv2
import numpy as np
import math
kernel_hs=25
kernel_hc=25
flat_kernel_h=35
kernel_h=10
kernel_window=4*kernel_h
kernel_thres=1.1
conv=30
filename = sys.argv[1]
kernel_hs=int(sys.argv[2])
kernel_hc=int(sys.argv[3])
conv=int(sys.argv[4])
print kernel_hs, kernel_hc
img = cv2.imread(filename)
height, width, channels = img.shape
# modecount=0
#print height
#imgLAB = [[[0 for i in range(3)] for j in range(width)] for k in range(height)]
imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
imgLAB=img
m=1
S=20
# li=[[] for i in range(height)]
lis = [[0,0] for j in  range(width*height)]
modes = []#[0,0] for j in  range(width*height)]
# print len(lis)
gradp = [[[0,0] for i in range(max(width,height))] for j in range(max(width,height))]
final = [[[-1,-1] for i in range(max(width,height))] for j in range(max(width,height))]
modes_count = [[[0,[]] for i in range(max(width,height))] for j in range(max(width,height))]

def distc(x1,y1,x2,y2):
	Dc = math.sqrt((int(imgLAB[x1][y1][0])-int(imgLAB[x2][y2][0]))**2 + (int(imgLAB[x1][y1][1])-int(imgLAB[x2][y2][1]))**2 + (int(imgLAB[x1][y1][2])-int(imgLAB[x2][y2][2]))**2)
	return Dc

def dists(x1,y1,x2,y2):
	Ds = math.sqrt((x2-x1)**2+(y2-y1)**2)
	return m*Ds/S

def dist(x1,y1,x2,y2):
	return math.sqrt((m/S)**2*dists(x1,y1,x2,y2)**2 + distc(x1,y1,x2,y2)**2)

def check_convergance(gx,gy):
	if (math.sqrt(gx**2+gy**2)<1):
		return 0
	else:
		return 1

def neggradientkernel(x1,y1,x2,y2):
	Ds=dists(x1,y1,x2,y2)
	Dc=distc(x1,y1,x2,y2)
	# print Ds/kernel_hs,Dc/kernel_hc
	return math.exp(-(Ds**2)/2/(kernel_hs**2))*math.exp(-(Dc**2)/2/(kernel_hc**2))/4*math.pi
	#return math.exp(-(a**2)/2/(kernel_h**2))/math.sqrt(2*)/(kernel_h**2)

def flatkernel(x1,y1,x2,y2):
	if ( dist(x1,y1,x2,y2)/flat_kernel_h <= 1):
		return 1
	else:
		return 0

# def assignmodeb(x,y):
# 	row=[]
# 	row.append(x)
# 	li[x].append(y)
# 	tx=x
# 	ty=y
# 	while(check_convergance(gradp[tx][ty][0],gradp[tx][ty][1])):
# 		if(ty in li[tx]):
# 			break
# 		row.append(tx)
# 		li[tx].append(ty)
# 		# lis[i]=[tx,ty] print i print tx,ty i=i+1
# 		[tx,ty]=[tx+gradp[tx][ty][0],ty+gradp[tx][ty][1]]
# 	for j in row:
# 		x =li[j].pop()
# 		final[j][x]=[tx,ty] # [temx,temy]=lis[j] final[temx][temy]=[tx,ty]
# 		# li[j].clear()
# 	return

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
#	a=set([(x[0]*x[1]) for x in lis])
#	if (len(a)<100):
#		val=100
#		for k in range(max(0,tx-kernel_window),min(height,tx)):
#			for l in range(max(0,ty-kernel_window),min(width,ty)):
#				temp=dist(tx,ty,final[k][l][0],final[k][l][1])
#				if(temp<val and final[k][l][0] != -1 and not(k==tx and l==ty )):
#					val=temp
#					val2=final[k][l]
	for j in range(i):
		[temx,temy]=lis[j]
		final[temx][temy]=val2
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
				#grad=flatkernel(i,j,k,l)
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
# cv2.imwrite('main.png',imgLABComp)
for i in range(height):
	for j in range(width):
		imgLABComp[i][j][0]=imgLAB[final[i][j][0]][final[i][j][1]][0]
		imgLABComp[i][j][1]=imgLAB[final[i][j][0]][final[i][j][1]][1]
		imgLABComp[i][j][2]=imgLAB[final[i][j][0]][final[i][j][1]][2]
cv2.imwrite(filename[:-4]+'type1'+str(kernel_hs)+str(kernel_hc)+str(conv) +'.png',imgLABComp)

for i  in range(height):
	for j in range(width):
		if(modes_count[i][j][0]<conv and modes_count[i][j][0] !=0):
			mergemode(i,j)
#for [p,q] in modes:
#	if (modes_count[p][q] <=40):
#		val=100
#		for k in range(max(0,p-kernel_window),min(height,p)):
#			for l in range(max(0,q-kernel_window),min(width,q)):
#				temp=dist(p,q,final[k][l][0],final[k][l][1])
#				if(temp<val and final[k][l][0] != -1 and not(final[k][l][0]==p and final[k][l][1]==q )):
#					val=temp
#					val2=final[k][l]

# print "Final Positions"
# for i in range(height):
# 	for j in range(width):
# 		print '[%d %d]'%(final[i][j][0], final[i][j][1]),
# 	print "\n"

# for i in range(height):
# 	for j in range(width):
# 		if(final[i][j][0]==i and final[i][j][1]==j):
# 			val=10
# 			val2=[i,j]
# 			for k in range(max(0,i-1),min(height,i+2)):
# 				for l in range(max(0,j-1),min(width,j+2)):
# 					temp=distc(i,j,k,l)
# 					if(temp<val and not(k==i and l==j )):
# 						val=temp
# 						val2=final[k][l]
# 			final[i][j]=val2
# for tx in range(height):
# 	for ty in range(width):
# 		val2=[tx,ty]
# 		val=10
# 		for k in range(max(0,tx-kernel_window),min(height,tx+kernel_window)):
# 			for l in range(max(0,ty-kernel_window),min(width,ty+kernel_window)):
# 				temp=distc(tx,ty,final[k][l][0],final[k][l][1])
# 				if(temp<val and final[k][l][0] != -1 and not(k==tx and l==ty )):
# 					val=temp
# 					val2=final[k][l]
# 		final[tx][ty]=val2

print "Computing Final Image"
print 'Final number of modes: ' + str(len(modes))
imgLABComp=img
for i in range(height):
	for j in range(width):
		imgLABComp[i][j][0]=imgLAB[final[i][j][0]][final[i][j][1]][0]
		imgLABComp[i][j][1]=imgLAB[final[i][j][0]][final[i][j][1]][1]
		imgLABComp[i][j][2]=imgLAB[final[i][j][0]][final[i][j][1]][2]
# cv2.imshow('image',imgLABComp)
# cv2.waitKey(0)
cv2.imwrite(filename[:-4]+'type2'+str(kernel_hs)+str(kernel_hc)+str(conv) +'.png',imgLABComp)
# cv2.destroyAllWindows()
