from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import os, multiprocessing, multiprocessing.pool
NUM_CLASSES = 15
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
root_path = '/Users/jartus/Chinese-Chess/classification_model'
os.chdir(root_path)

# The things you need to change is in here
train_dir = '../Dataset/train'
valid_dir = '../Dataset/valid'
loaded_model_path = '../h5_file/model.h5'
#saved_model_path = '../Temporary_Model/cnn_mini_v{epoch:d}.h5'
saved_model_path = '../Temporary_Model/cnn_mini.h5'
epochs = 10
steps_per_epoch = 1400
validation_steps = 400


def create_cnn_model():
		model = Sequential()
		### Conv layer 1
		model.add(Convolution2D(
				input_shape=(56, 56, 3),
				filters = 32,
				kernel_size=3,
				strides=1,
				padding='same',
				data_format='channels_last',
				activation='relu'
		))
		model.add(MaxPooling2D(
				pool_size=2,
				strides=2,
				padding='same',
				data_format='channels_last',
		))
		### Conv layer 2
		model.add(Convolution2D(32, 3, strides=1, padding='same', data_format='channels_last', activation='relu'))
		model.add(MaxPooling2D(2, 2, padding='same', data_format='channels_last'))

		### Conv layer 3
		model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last', activation='relu'))
		model.add(MaxPooling2D(2, 2, padding='same', data_format='channels_last'))
		model.add(Dropout(0.25))

		### FC
		model.add(Flatten())
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.5))

		model.add(Dense(NUM_CLASSES, activation='softmax', name='output'))

		model.compile(optimizer='Adam',
					  loss='categorical_crossentropy',
					   metrics=['accuracy'])
		return model

model = create_cnn_model()
#model.load_weights(loaded_model_path, by_name=True)
model.summary()
#pool = multiprocessing.Pool(processes=8)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
		rescale=1./255,
		rotation_range= 360,
		#channel_shift_range=10,
		#width_shift_range= 0.1,
		#height_shift_range= 0.1,
		#shear_range= 0.2,
		#zoom_range= (0.8,1.1),
		#fill_mode= 'nearest',
		#pool=pool
		)

# this is the augmentation configuration we will use for testing:
# only rescaling
validation_datagen = ImageDataGenerator(
		rescale=1./255,
		rotation_range= 360,
		#pool=pool
		)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
		train_dir,  # this is the target directory
		target_size=(56, 56),  # all images will be resized to 150x150
		batch_size=50,
		# save_to_dir='temp/train',
		class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
		valid_dir,
		target_size=(56, 56),
		batch_size=50,
		# save_to_dir='temp/valid',
		class_mode='categorical')

checkpointer = ModelCheckpoint(filepath=saved_model_path, verbose=1, save_best_only=True)
callbacks = TensorBoard(
    log_dir='./logs',
	histogram_freq=0,
	write_graph = True,
	write_images = True,
	update_freq = 50
)
model.fit_generator(
		train_generator,
		steps_per_epoch=steps_per_epoch,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=validation_steps,
		callbacks=[checkpointer, callbacks]
)