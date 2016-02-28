import cv2
import glob
import numpy as np

source_dir = 'dataset/'
fileList = glob.glob(source_dir + '/*.jpg')

no_images = len(fileList)
no_sift = 50
numpoints = no_images*no_sift
X = np.zeros((numpoints,128))
i = 0
sift = cv2.xfeatures2d.SIFT_create(sigma = 1.0)

for fil in fileList:
	print fil
	im = cv2.imread(fil);
	gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	(kps, descs) = sift.detectAndCompute(gray, None)
	print len(kps)
	no_sift = min(50, len(kps))
	print no_sift
	X[i:i+no_sift] = descs[0:no_sift]
	i += no_sift
	img = im
	cv2.drawKeypoints(im,kps[0:no_sift],img)
	cv2.imwrite('a'+fil[:-4]+'2.jpg',img)
print X[0:i]
