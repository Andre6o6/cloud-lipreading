import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow
from IPython.display import clear_output
from mtcnn import MTCNN
from tqdm import tqdm

def detect_face(image, detector, color=(0,155,255), max_count=1):
    result = detector.detect_faces(image)
    for r in result[:max_count]:
        bbox = r['box']
        keypoints = r['keypoints']

        cv2.rectangle(image,
                      (bbox[0], bbox[1]),
                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      color,
                      2)
        cv2.circle(image,(keypoints['left_eye']), 2, color, 2)
        cv2.circle(image,(keypoints['right_eye']), 2, color, 2)
        cv2.circle(image,(keypoints['nose']), 2, color, 2)
        cv2.circle(image,(keypoints['mouth_left']), 2, color, 2)
        cv2.circle(image,(keypoints['mouth_right']), 2, color, 2)
    return image

def process_video(video_path, output_path, fps, max_frames=500):
    detector = MTCNN()
    
    #Input stream of frames
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    
    #Output stream of frames
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    size = (image.shape[1], image.shape[0])
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    t = tqdm(range(max_frames))
    count = 0
    while success:
        image = detect_face(image, detector, max_count=2)
        out.write(image)

        success,image = vidcap.read()
        count += 1
        t.update()
        if count>=max_frames:
            break
    out.release()
    t.close()