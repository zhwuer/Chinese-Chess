# Chinese Chess Recognition
import AdjustCameraLocation as ad
import cv2, os, pymysql, operator, copy
import numpy as np
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ip = ad.ip
pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
		'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
pieceTypeList_with_grid = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu', 'grid',
                'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
label_type = pieceTypeList_with_grid
dic = {'b_jiang':'Black King', 'b_ju':'Black Rook', 'b_ma':'Black Knight', 'b_pao':'Black Cannon', 'b_shi':'Black Guard', 'b_xiang':'Black Elephant', 'b_zu':'Black Pawn',
		'r_bing':'Red Soldier', 'r_ju':'Red Chariot', 'r_ma':'Red Horse', 'r_pao':'Red Cannon', 'r_shi':'Red Adviser', 'r_shuai':'Red General', 'r_xiang':'Red Minister'}


def Initialization():
	global GRID_WIDTH_HORI, GRID_WIDTH_VERTI, begin_point, cap, db, cursor, step, legal_move, model, target_size, isRed

	step = 0		# For recording steps
	legal_move = True	# To decide if the move is legal or illegal
	isRed = True		# To decide which team moves now
	target_size = (56, 56)		# CNN input image size
	model = load_model('./h5_file/new_model_v2.h5')	# Machine Learning Model
	
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
	
	# Initialize grid width
	frame0 = cv2.imread('./Test_Image/Step 0.png', 0)
	img_circle = cv2.HoughCircles(frame0,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=20,minRadius=18,maxRadius=22)[0]
	begin_point = img_circle[np.sum(img_circle, axis=1).tolist().index(min(np.sum(img_circle, axis=1).tolist()))]
	end_point = img_circle[np.sum(img_circle, axis=1).tolist().index(max(np.sum(img_circle, axis=1).tolist()))]
	GRID_WIDTH_HORI = (end_point[0] - begin_point[0])/8
	GRID_WIDTH_VERTI = (end_point[1] - begin_point[1])/9
	print('Recognition Initialized.\n')

def PiecePrediction(model, img, target_size, top_n=3):
	x = cv2.resize(img, target_size)
	x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	x = x / 255
	x = np.expand_dims(x, axis=0)
	preds = model.predict_classes(x)
	return label_type[int(preds)]

def savePath(beginPoint, endPoint, piece):
	global legal_move	# For indicating error movement
	begin = (np.around(abs(beginPoint[0]-begin_point[0])/GRID_WIDTH_HORI), np.around(abs(beginPoint[1]-begin_point[1])/GRID_WIDTH_VERTI))
	#print(beginPoint, endPoint)
	end = begin
	updown = np.around(abs(beginPoint[1]-endPoint[1])/GRID_WIDTH_VERTI)
	leftright = np.around(abs(beginPoint[0]-endPoint[0])/GRID_WIDTH_HORI)
	#print(updown, leftright)
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
		if begin[1] < end[1] or (begin[1] >= 5.0 and begin[0] != end[0]) or (begin[1]-end[1] > 1):
			legal_move = False
	elif variety in ['zu']:
		if begin[1] > end[1] or (begin[1] <= 4.0 and begin[0] != end[0]) or (end[1]-begin[1] > 1):
			legal_move = False

	if isRed:
		if color == 'b':
			legal_move = False
			print('It''s red team''s turn to move')
	else:
		if color == 'r':
			legal_move = False
			print('It''s black team''s turn to move')

	if not legal_move:
		cv2.imwrite('./pieces/%d.png' % np.random.randint(10000), piece)
	text = str(int(begin[0])) + str(int(begin[1])) + str(int(end[0])) + str(int(end[1]))
	return text, predict_category

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

def CalculateTrace(pre_img, cur_img, x, y, w, h):
	# Input loca = [x, y, w, h], return all circle center inside the rectangular to pointSet
	pointSet = []
	beginPoint = []
	endPoint = []
	pre_img_gray = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
	cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
	pre_img_circle = cv2.HoughCircles(pre_img_gray,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=20,minRadius=18,maxRadius=18)[0]
	cur_img_circle = cv2.HoughCircles(cur_img_gray,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=20,minRadius=18,maxRadius=18)[0]
	for j in range(int(np.around(h/GRID_WIDTH_VERTI))):
		for i in range(int(np.around(w/GRID_WIDTH_HORI))):
			pointSet.append([y + (j+0.5)*GRID_WIDTH_VERTI, x + (i+0.5)*GRID_WIDTH_HORI])
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
	if beginPoint != [] and endPoint != []:
		piece = pre_img[int(beginPoint[1] - beginPoint[2]):int(beginPoint[1] + beginPoint[2]),
				int(beginPoint[0] - beginPoint[2]):int(beginPoint[0] + beginPoint[2])]
	else:
		return [], [], []
	return beginPoint, endPoint, piece

def changeDetection(previous_step, current_step, visual = False):
	current_frame_gray = cv2.cvtColor(current_step, cv2.COLOR_BGR2GRAY)
	previous_frame_gray = cv2.cvtColor(previous_step, cv2.COLOR_BGR2GRAY)
	frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
	frame_diff = cv2.medianBlur(frame_diff, 5)
	ret, frame_diff = cv2.threshold(frame_diff, 0, 255, cv2.THRESH_OTSU)
	frame_diff = cv2.medianBlur(frame_diff, 5)
	x, y, w, h = cv2.boundingRect(frame_diff)
	#### For Test ####
	if visual:
		cv2.rectangle(frame_diff, (x, y), (x + w, y + h), (255, 255, 255), 2)
		cv2.imshow('', frame_diff)
		cv2.waitKey(20)
	#### For Test ####
	return x, y, w, h

def compare(img1, img2, x, y, w, h):
	subset = []
	r = 19
	dict1 = {}
	dict2 = {}
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img1_circle = cv2.HoughCircles(img1_gray, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=20, minRadius=18, maxRadius=18)[0]
	for i in img1_circle:
		if x<i[0]<x+w and y<i[1]<y+h:
			subset.append(i)
	for point in subset:
		coordinate = ((point[0] - begin_point[0]) / GRID_WIDTH_HORI, (point[1] - begin_point[1]) / GRID_WIDTH_VERTI)
		piece1 = img1[int(point[1] - r):int(point[1] + r), int(point[0] - r):int(point[0] + r)]
		piece2 = img2[int(point[1] - r):int(point[1] + r), int(point[0] - r):int(point[0] + r)]
		cat1 = PiecePrediction(model, piece1, target_size)
		cat2 = PiecePrediction(model, piece2, target_size)
		dict1[(coordinate[0], coordinate[1])] = cat1
		dict2[(coordinate[0], coordinate[1])] = cat2
	return operator.eq(dict1, dict2)

def PiecesChangeDetection(current_step):
	global legal_move
	previous_step = cv2.imread('./Test_Image/Step %d.png' % step)
	x, y, w, h = changeDetection(previous_step, current_step, False)
	if w * h < 50*50 or x == 0 or y == 0 or x+w == 480 or y+h == 480 or compare(previous_step, current_step, x, y, w, h):	#棋子没有移动
		return 0
	else:
		beginPoint, endPoint, piece = CalculateTrace(previous_step, current_step, x, y, w, h)
		if beginPoint != [] and endPoint != [] and piece != []:
			text, predict_category = savePath(beginPoint, endPoint, piece)
			if legal_move:
				sql2 = "INSERT INTO chess(STEP) VALUES (\'%s\')" % text
				cursor.execute(sql2)
				db.commit()
			else:
				print('%s performed a illegal movement!' % dic[predict_category])
		else:
			return 0
		if legal_move:
			cv2.imwrite('./Test_Image/Step %d.png' % (step + 1), current_step)
			return 1
		else:
			print('Please rollback to step %d' % step)
			while (True):
				r, frame = cap.read()
				frame = frame[0:480, 0:480]
				x, y, w, h = changeDetection(previous_step, frame)
				if x != 0 and y != 0 and x + w != 480 and y + h != 480 and compare(previous_step, frame, x, y, w, h):
					legal_move = True
					cv2.imwrite('./Test_Image/Step %d.png' % step, frame)
					print('Rollback successfully!')
					break
			return 0

if __name__ == '__main__':
	# Initialize camera
	cap = cv2.VideoCapture("http://admin:admin@%s:8081/" % ip)
	# cap = cv2.VideoCapture('test.avi')
	if cap.isOpened():
		for j in range(20):
			cap.read()
		ret, current_frame = cap.read()
		current_frame = current_frame[0:480, 0:480]
		cv2.imwrite('./Test_Image/Step 0.png', current_frame)
	else:
		exit('Camera is not open.')
	print('Camera Initialized.')
	previous_frame = current_frame
	Initialization()
	while (cap.isOpened()):
		x, y, w, h = changeDetection(current_frame, previous_frame)
		if (x == 0 and y == 0 and w == 480 and h == 480):
			num = PiecesChangeDetection(current_frame)
			if num == 1:
				step += 1
				isRed = bool(1 - isRed)
			elif num == 0: 
				pass
		previous_frame = current_frame.copy()
		ret, current_frame = cap.read()
		if not ret:
			break
		current_frame = current_frame[0:480, 0:480]
		cv2.rectangle(current_frame, ad.begin, (ad.begin[0] + 400, ad.begin[1] + 400), (255, 255, 255), 2)
		cv2.imshow('', current_frame)
		cv2.waitKey(1)

	cap.release()
	cv2.destroyAllWindows()
	db.close()