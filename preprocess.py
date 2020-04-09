import cv2
import numpy as np
import os

def video_to_frames(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    i = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = "frame{:03}.png".format(i)
        frame_path = os.path.join(frames_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        i += 1
    cap.release()


def dataset_to_frames(root="GRID/videos", out_root="GRID/frames"):
    for subject in os.listdir(root):
        os.mkdir(os.path.join(out_root, subject))

        dir_path = os.path.join(root, subject)
        for file in os.listdir(dir_path):
            filename = file.split('.')[0]
            out_dir_path = os.path.join(out_root, subject, filename)
            os.mkdir(out_dir_path)

            video_path = os.path.join(dir_path, file)
            video_to_frames(video_path, out_dir_path)
        print("Done with subject", subject)