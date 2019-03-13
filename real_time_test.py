# Chinese Chess Recognition
import AdjustCameraLocation as ad
import cv2, os, pymysql, traceback
import numpy as np
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# The parament of HoughCircle function need to be adjusted
THRESHOLD = 15
calibrated_point = (ad.begin[0] + 20, ad.begin[1] + 20)
GRID_WIDTH_HORI = 45
GRID_WIDTH_VERTI = 40
target_size = (56, 56)
isRed = True
ip = ad.ip
model = load_model('./h5_file/model.h5')
pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
		'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
dic = {'b_jiang':'Black King', 'b_ju':'Black Rook', 'b_ma':'Black Knight', 'b_pao':'Black Cannon', 'b_shi':'Black Guard', 'b_xiang':'Black Elephant', 'b_zu':'Black Pawn',
		'r_bing':'Red Soldier', 'r_ju':'Red Chariot', 'r_ma':'Red Horse', 'r_pao':'Red Cannon', 'r_shi':'Red Adviser', 'r_shuai':'Red General', 'r_xiang':'Red Minister'}


def PiecePrediction(model, img, target_size, top_n=3):
	x = cv2.resize(img, target_size)
	x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	x = x / 255
	x = np.expand_dims(x, axis=0)
	preds = model.predict_classes(x)
	return pieceTypeList[int(preds)]

def savePath(beginPoint, endPoint, piece):
	global legal_move	# For indicating error movement
	begin = (np.around(abs(beginPoint[0]-calibrated_point[0])/GRID_WIDTH_HORI), np.around(abs(beginPoint[1]-calibrated_point[1])/GRID_WIDTH_VERTI))
	#print(beginPoint, endPoint)
	end = begin
	updown = np.around(abs(beginPoint[1]-endPoint[1])/GRID_WIDTH_VERTI)
	leftright = np.around(abs(beginPoint[0]-endPoint[0])/GRID_WIDTH_HORI)
	#print(updown, leftright)
	cv2.imwrite('./pieces/%d.png' % np.random.randint(10000), piece)
	predict_category = PiecePrediction(model, piece, target_size)
	variety = predict_category.split('_')[-1]
	color = predict_category.split('_')[0]

	# Print the path
	if beginPoint[1] - endPoint[1] > 0:
		end = (end[0], end[1] - updown)
	else:
		end = (end[0], end[1] + updown)

	if beginPoint[0] - endPoint[0] > 0:
		end = (end[0] - leftright, end[1])
	else:
		end = (end[0] + leftright, end[1])
	print('{} moved from point {} to point {}'.format(dic[predict_category], begin, end))

	# Using chinese chess rules to reduce error movement
	if variety in ['ma']:
		if not (updown == 1 and leftright == 2) and not (updown == 2 and leftright == 1):
			legal_move = False
	elif variety in ['xiang']:
		if not (updown == 2 and leftright == 2) and not (updown == 2 and leftright == 2):
			legal_move = False
	elif variety in ['shi']:
		if not (updown == 1 and leftright == 1) and not (updown == 1 and leftright == 1):
			legal_move = False
	elif variety in ['jiang', 'shuai']:
		if not (updown == 1 and leftright == 0) and not (updown == 0 and leftright == 1):
			legal_move = False
	elif variety in ['ju', 'pao']:
		if updown != 0 and leftright != 0:
			legal_move = False
	elif variety in ['bing']:
		if begin[1] < end[1] or (begin[1] >= 5.0 and begin[0] != end[0]):
			legal_move = False
	elif variety in ['zu']:
		if begin[1] > end[1] or (begin[1] <= 4.0 and begin[0] != end[0]):
			legal_move = False

	if isRed:
		if color == 'b':
			legal_move = False
			print('It''s red team''s turn to move')
	else:
		if color == 'r':
			legal_move = False
			print('It''s black team''s turn to move')

	text = str(int(begin[0])) + str(int(begin[1])) + str(int(end[0])) + str(int(end[1]))
	if legal_move:
		sql2 = "INSERT INTO chess(STEP) VALUES (\'%s\')" % text
		cursor.execute(sql2)
		db.commit()
	else:
		print('%s performed a illegal movement!' % dic[predict_category])

def findPoint(point, pointset):
	flag = False
	point_finetune = []
	for i in pointset:
		#point is (y, x)
		v1 = np.array([i[1], i[0]])
		v2 = np.array(point)
		d = np.linalg.norm(v1 - v2)
		if d < 30:
			flag = True
			point_finetune = i
			break
	return flag, point_finetune

def CalculateTrace(pre_img, cur_img):
	x, y, w, h = moveRecognition(cur_img, pre_img)
	# Input loca = [xmin,ymin,xmax,ymax], return all circle center inside all the rectangular to pointSet
	[xmin, ymin, xmax, ymax] = [x, y, x + w, y + h]
	pre_img_gray = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
	cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
	pre_img_circle = cv2.HoughCircles(pre_img_gray,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=20,minRadius=18,maxRadius=18)[0]
	cur_img_circle = cv2.HoughCircles(cur_img_gray,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=20,minRadius=18,maxRadius=18)[0]
	pointSet = []
	beginPoint = []
	endPoint = []
	for j in range(int(np.around((ymax-ymin)/GRID_WIDTH_VERTI))):
		for i in range(int(np.around((xmax-xmin)/GRID_WIDTH_HORI))):
			pointSet.append([ymin + (j+0.5)*GRID_WIDTH_VERTI, xmin + (i+0.5)*GRID_WIDTH_HORI])
	
	for p in pointSet:
		if beginPoint != [] and endPoint != []:		# Already find beginPoint and endPoint, exit 
			break
		flag1, p1 = findPoint(p, pre_img_circle)
		flag2, p2 = findPoint(p, cur_img_circle)
		if len(pre_img_circle)-len(cur_img_circle) == 1:	# 发生了吃子
			if flag1 == True and flag2 == False:
				beginPoint = p1
			elif flag1 == True and flag2 == True:
				pre_piece = pre_img[ int(p1[1]-p1[2]):int(p1[1]+p1[2]), int(p1[0]-p1[2]):int(p1[0]+p1[2]) ]
				cur_piece = cur_img[ int(p2[1]-p2[2]):int(p2[1]+p2[2]), int(p2[0]-p2[2]):int(p2[0]+p2[2]) ]
				if PiecePrediction(model, pre_piece, target_size) != PiecePrediction(model, cur_piece, target_size):
					endPoint = p2
		elif len(pre_img_circle) == len(cur_img_circle):	#没有发生棋子减少情况
			if flag1 == True and flag2 == False:
				beginPoint = p1
			elif flag1 == False and flag2 == True:
				endPoint = p2
	try:
		if beginPoint != [] and endPoint != []:
			piece = pre_img[int(beginPoint[1] - beginPoint[2]):int(beginPoint[1] + beginPoint[2]),
					int(beginPoint[0] - beginPoint[2]):int(beginPoint[0] + beginPoint[2])]
			savePath(beginPoint, endPoint, piece)
	except Exception as e:
		print('There is a bug when running function savePath().')
		print(repr(e))
		traceback.print_exc()

def moveRecognition(current_step, previous_step):
	current_frame_gray = cv2.cvtColor(current_step, cv2.COLOR_BGR2GRAY)
	previous_frame_gray = cv2.cvtColor(previous_step, cv2.COLOR_BGR2GRAY)
	frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
	ret, frame_diff = cv2.threshold(frame_diff, 0, 255, cv2.THRESH_OTSU)
	for i in range(6):
		frame_diff = cv2.medianBlur(frame_diff, 11)
	x, y, w, h = cv2.boundingRect(frame_diff)
	return x, y, w, h

def changeDetection(current_step, previous_step):
	current_frame_gray = cv2.cvtColor(current_step, cv2.COLOR_BGR2GRAY)
	previous_frame_gray = cv2.cvtColor(previous_step, cv2.COLOR_BGR2GRAY)
	frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
	frame_diff = cv2.medianBlur(frame_diff, 5)
	ret, frame_diff = cv2.threshold(frame_diff, 0, 255, cv2.THRESH_OTSU)
	frame_diff = cv2.medianBlur(frame_diff, 5)
	x, y, w, h = cv2.boundingRect(frame_diff)
	return x, y, w, h

def PiecesChangeDetection():
	global legal_move
	previous_step = cv2.imread('./Test_Image/Step %d.png' % step)
	r, current_step = cap.read()
	x, y, w, h = changeDetection(current_step, previous_step)
	if w * h < 2700 or (x == 0 and y == 0 and w == 640 and h == 480):
		return 0
	elif x != 0 and y != 0 and x+w != 640 and y+h != 480:
		try:
			CalculateTrace(previous_step, current_step)
		except Exception as e:
			print('There is a bug when running function CalculateTrace().')
			traceback.print_exc()
			return 0

		if legal_move:
			cv2.imwrite('./Test_Image/Step %d.png' % (step + 1), current_step)
			return 1
		else:
			print('Please rollback to step %d' % step)
			while (True):
				r, current_step = cap.read()
				x, y, w, h = moveRecognition(current_step, previous_step)
				#print(x,y,w,h)
				if w * h < 2700 or (x == 0 and y == 0 and w == 640 and h == 480):
					legal_move = True
					cv2.imwrite('./Test_Image/Step %d.png' % step, current_step)
					break
			return 2
	else:
		return 0

if __name__ == '__main__':
	cap = cv2.VideoCapture("http://admin:admin@%s:8081/" % ip)
	# Initialize mysql
	db = pymysql.connect("localhost", "root", "zhenmafan", "chess")
	cursor = db.cursor()
	cursor.execute("DROP TABLE IF EXISTS chess")
	sql1 = """CREATE TABLE chess (
				Id INT AUTO_INCREMENT,
				STEP CHAR(4) NOT NULL,
				PRIMARY KEY(Id)
			)"""
	cursor.execute(sql1)
	print('SQL Initialized.')
	step = 0
	s = []
	legal_move = True
	if cap.isOpened():
		for j in range(20):
			cap.read()
		ret, current_frame = cap.read()
		cv2.imwrite('./Test_Image/Step %d.png' % step, current_frame)
	else:
		exit('Camera is not open.')

	previous_frame = current_frame
	print('Recognition Initialized.\n')
	try:
		while (cap.isOpened()):
			x, y, w, h = changeDetection(current_frame, previous_frame)
			if (x == 0 and y == 0 and w == 640 and h == 480):
				try:
					num = PiecesChangeDetection()
				except:
					print('There is a bug when running function PiecesChangeDetection().')
				if legal_move:
					if num == 1:
						step += 1
						isRed = bool(1 - isRed)
					elif num == 2:
						print('Rollback successfully!')
					elif num == 0:
						pass

			previous_frame = current_frame.copy()
			ret, current_frame = cap.read()
	except:
		print('Exit')
		cap.release()
		cv2.destroyAllWindows()
		db.close()