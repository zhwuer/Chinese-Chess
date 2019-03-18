import cv2

ip = '10.19.46.153'
ip = '192.168.1.100'
cap = cv2.VideoCapture("http://admin:admin@%s:8081/" % ip)
begin = (40, 40)

def adjust():
	while (cap.isOpened()):
		## Capture frame-by-frame
		ret, cv2_im = cap.read()
		cv2.rectangle(cv2_im, begin, (begin[0] + 400, begin[1] + 400), (255,255,255), 2)
		cv2.imshow('frame', cv2_im)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

def pic():
	if (cap.isOpened()):
		for j in range(20):
			cap.read()
		ret, cv2_im = cap.read()
		cv2.imwrite('./Test_Image/Step 22.png', cv2_im)
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	#pic()
	adjust()