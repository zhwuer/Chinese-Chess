# Chinese Chess Recognition
To solve this problem, the first step is to identify the beginning point and the end point of the pieces movement. The second step is to crop a square image on the begin/end point and pass it to convolutional neural network (CNN) to do the chess pieces classification.
![Flow Chart](./Sources/Flow%20chart.png)

![Test](https://github.com/zhwuer/Chinese-Chess/blob/master/Sources/Test.gif)
---
## <span style="color:blue">Usage</span>
- [CNN_Classification_Model](./CNN_Classification_Model) contains the codes I used to train the classification model, it is ok to use the .h5 model named [new_model_v2.h5](./h5_file/new_model_v2.h5) in the h5_file.
- [Dataset](./Dataset) contains the dataset I made by taking pictures by Phone and using HoughCircle to roughly extract some chess pieces from the picture.
- The [.h5 file model](./h5_file/new_model_v2.h5) works for those images in the Dataset with nearly 100% accuracy, if it is used to detect other kinds of test image, the accuracy may get lower.
- [Temporary_Model](./Temporary_Model) and [Test_Image](./Test_Image) are two necessary directory used in the codes.
- [AdjustCameraLocation.py](AdjustCameraLocation.py) is used to adjust the camera to maximize the area of the chess board in the picture.
- [real_time_test.py](./real_time_test.py) is the main function of this project.

## <span style="color:blue">Notes</span>
- The training data in Dataset/train lost some images because of some unknown reasons, if you need to re-train you model, you can generate more data by yourself or just move some data from valid to train :).
- The location of the phone need to be right over the chess board, it is very hard to fix it(I used the mobile phone holder like [Tools](./Sources/Tools.png)). So I provide a video named [test.avi](./Sources/test.avi) in the Sources directory. If you want to do the real time test, you need to change the code in line 240, [real_time_test.py](./real_time_test.py).

# Test on iPad
![IMG1](https://github.com/zhwuer/Chinese-Chess/blob/master/Sources/Test_IMG1.png)
![IMG2](https://github.com/zhwuer/Chinese-Chess/blob/master/Sources/Test_IMG2.png)

# Test Video
[Full Video tested in the iPad](https://youtu.be/6aI8yIMQmbc)

## Reference
[1] [https://github.com/itlwei/Chess](https://github.com/itlwei/Chess)<br>
[2] [https://github.com/evanchien/chinese_chess_recognition](https://github.com/evanchien/chinese_chess_recognition)
