import sys
import cv2
import numpy as np
import math
filename1= sys.argv[1]
filename2 = sys.argv[2]
window = int(sys.argv[4])
thres_factor  = float(sys.argv[3])
windo = int(sys.argv[5])
windo2 = int(sys.argv[6])
# print thres_factor
# cv2.imshow('image',gray_img)
#threshold
#gradient window
#padding
#weighted summation for hessian matrix
# window size of interest point descriptor
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

			# print f[i][j]
	su = 0
	for x in range(len(f)):
		su += sum(f[x])
	threshold = thres_factor*float(su)/len(f)/len(f[0])
	for i in range(height):
		for j in range(width):
			if(f[i][j]>threshold):
				im[i][j]=f[i][j]
			else:
				im[i][j]=0
	# print max(max(f))
	# data=np.array([f[x/height][x%width] for x in range(height*width)])
	# hist,bins=np.histogram(data,bins=np.linspace(0,max(max(f)),max(max(f))/20))
	# print(hist)
	im2=im
	# for i in range(1,height-1):
	# 	for j in range(1,width-1):
	# 		im[i][j]=(im[i][j-1]+im[i][j+1]+im[i+1][j-1]+im[i+1][j]
	# +im[i+1][j+1]+im[i-1][j-1]+im[i-1][j+1]+im[i-1][j]+im[i][j])/9
	points=[]
	for i in range(5,height-5):
		for j in range(5,width-5):
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
	return [img,im2,ip]


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
		lssd = 1280000
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
			match.append([i[1],temp[1],lssd])
	return match


# cv2.imshow('image',im)
[img1,ip1image,ip1]=ip(filename1)
[img2,ip2image,ip2]=ip(filename2)

matches = intersection(ip1,ip2)
matches = sorted(matches,key=lambda matches: matches[2])
# for x in matches:
# 	print x
# print match
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
# print h1,w1,h2,w2
nWidth = w1+w2
nHeight = max(h1, h2)
hdif = (h1-h2)/2
newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
newimg[hdif:hdif+h2, :w2] = img2
newimg[:h1, w2:w1+w2] = img1
for i in range(min(len(matches),30)):
	pt_a = (int(matches[i][1][1]), int(matches[i][1][0]+hdif))
	pt_b = (int(matches[i][0][1]+w2), int(matches[i][0][0]))
	cv2.line(newimg, pt_a, pt_b, (0, 255, 0),1)
	cv2.circle(newimg,(matches[i][1][1],matches[i][1][0]+hdif), 3 , (0,0,255), -1)
	cv2.circle(newimg,(matches[i][0][1]+w2,matches[i][0][0]), 3, (0,0,255), -1)
cv2.imwrite(filename1[:-4]+str(thres_factor)+str(window)+str(windo)+str(windo2)+'combine.png',newimg)
cv2.imwrite(filename1[:-4]+str(thres_factor)+str(window)+str(windo)+str(windo2)+'lam.png',ip1image)
cv2.imwrite(filename2[:-4]+str(thres_factor)+str(window)+str(windo)+str(windo2)+'lam.png',ip2image)
# cv2.waitKey(0)
