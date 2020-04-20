import numpy as np
import skvideo.io
from mtcnn import MTCNN
from lipnet import LipNet
from decoder import Decoder

video_data = skvideo.io.vread("GRID/videos/s1/bbaf2n.mpg")

detector = MTCNN()
cropped_mouth = []
size = (100,50)
result = detector.detect_faces(video_data[0])
keypoints = result[0]['keypoints']

x0,y0 = keypoints['mouth_left']
x1,y1 = keypoints['mouth_right']

center_x, center_y = int((x0+x1)/2), int((y0+y1)/2)

x0,x1 = center_x-size[0]//2, center_x+size[0]//2
y0,y1 = center_y-size[1]//2, center_y+size[1]//2

for image in video_data:
    mouth = image[y0:y1, x0:x1]
    cropped_mouth.append(mouth)

cropped_mouth = np.array(cropped_mouth)



weights_path = "data/res/2018-09-26-02-30/lipnet_065_1.96.hdf5"
lipnet = LipNet(frame_count=75, 
                image_channels=3, 
                image_height=50, 
                image_width=100, 
                max_string=32
                ).compile_model().load_weights(weights_path)
                
x_data = np.swapaxes(cropped_mouth, 1, 2)  # T x W x H x C
x_data = x_data.astype(np.float32) / 255
x_data = x_data[np.newaxis, :]
y_pred = lipnet.predict(x_data)

input_lengths=np.array([75])
DICTIONARY_PATH = 'data/dictionaries/grid.txt'

decoder = create_decoder(DICTIONARY_PATH)   #FIXME
results = decoder.decode(y_pred, input_lengths)
print(results)