img = cv2.imread("./dataset/42049.jpg")
cv2.CvtColor(img, imgLAB, CV_RGB2Lab)

m=1
S=1

def dist(x1,y1,x2,y2):
	Ds = math.sqrt((x2-x1)**2+(y2-y2)**2)
	Dc = math.sqrt((imgLAB[x1][y1][0]-imgLAB[x2][y2][0])**2 + (imgLAB[x1][y1][1]-imgLAB[x2][y2][1])**2 + (imgLAB[x1][y1][2]-imgLAB[x2][y2][2])**2)
	return math.sqrt(Ds**2 + (m**2)*((Ds/S)**2))
