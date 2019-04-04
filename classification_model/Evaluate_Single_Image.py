import cv2, os
import numpy as np
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
target_size = (56, 56)
pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
		'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
pieceTypeList_with_grid = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu', 'grid',
                'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
label_type = pieceTypeList_with_grid
# The things you need to change is in here
weights = '/Users/jartus/Chinese-Chess/Temporary_Model/cnn_mini_v49.h5'
weights = '/Users/jartus/Chinese-Chess/h5_file/cnn_mini2.h5'
file_path = '/Users/jartus/Chinese-Chess/Dataset/finetune/'
file_path = '/Users/jartus/Chinese-Chess/Dataset/finetune/r_bing/'

def evaluate_one(weights, file_path):
	model = load_model(weights)
	x = cv2.imread(file_path)
	x = cv2.resize(x, target_size)
	x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	x = x / 255
	x = np.expand_dims(x, axis=0)
	preds = model.predict_classes(x)
	print(label_type[int(preds)])
def evaluate(weights, file_path):    # Input is a directory, Output is the total number and error number
	model = load_model(weights)
	for i in os.listdir(file_path):
		if i == '.DS_Store':
			continue
		x = cv2.imread(file_path + i)
		x = cv2.resize(x, target_size)
		x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
		x = np.expand_dims(x, axis=0)
		preds = model.predict(x)
		preds = np.around(preds)
		print(label_type[np.where(preds[0] == 1)[0][0]])

evaluate(weights, file_path)
#evaluate_one(weights, file_path)