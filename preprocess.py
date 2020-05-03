import argparse
import numpy as np
import os
import skvideo.io
from mtcnn import MTCNN

def resize_factor(detector, image, mouth_size=40):
    result = detector.detect_faces(image)
    keypoints = result[0]['keypoints']
    x0,_ = keypoints['mouth_left']
    x1,_ = keypoints['mouth_right']
    return mouth_size/abs(x1-x0)


def crop_mouth(detector, image, size=(100,50)):
    result = detector.detect_faces(image)
    keypoints = result[0]['keypoints']
    x0,y0 = keypoints['mouth_left']
    x1,y1 = keypoints['mouth_right']
    
    center_x, center_y = int((x0+x1)/2), int((y0+y1)/2)
    x0,x1 = center_x-size[0]//2, center_x+size[0]//2
    y0,y1 = center_y-size[1]//2, center_y+size[1]//2
    return x0,y0,x1,y1


def batch_crop_mouth(detector, images, size=(100,50)):
    f,_,_,c = images.shape
    cropped = np.zeros((f,size[1],size[0],c))
    for i,image in enumerate(images):    
        result = detector.detect_faces(image)
        keypoints = result[0]['keypoints']
        x0,y0 = keypoints['mouth_left']
        x1,y1 = keypoints['mouth_right']
        
        center_x, center_y = int((x0+x1)/2), int((y0+y1)/2)
        x0,x1 = center_x-size[0]//2, center_x+size[0]//2
        y0,y1 = center_y-size[1]//2, center_y+size[1]//2

        cropped[i] = images[i, y0:y1, x0:x1]
    return cropped


def dataset_to_numpy(detector, root="GRID/videos/", out_root="GRID/videos_npy/"):
    for subject in os.listdir(root):
        os.mkdir(os.path.join(out_root, subject))

        dir_path = os.path.join(root, subject)
        for video_file in os.listdir(dir_path):
            filename = video_file.split('.')[0]
            out_path = os.path.join(out_root, subject, filename+'.npy')
            video_path = os.path.join(dir_path, video_file)
            
            #TODO process missing mouth and not video separately
            try:
                video_data = skvideo.io.vread(video_path)
                x0,y0,x1,y1 = crop_mouth(detector, video_data[0])
            except:
                continue
            new_video_data = video_data[:, y0:y1, x0:x1, :]
            np.save(out_path, new_video_data)
        print("Done with subject", subject)


def arg_parse():
    parser = argparse.ArgumentParser(description="Preprocess GRID")
    parser.add_argument(
        "--root",
        dest="root",
        help="Directory where GRID videos are",
        default="GRID/videos/",
        type=str,
    )
    parser.add_argument(
        "--out",
        dest="out_root",
        help="Directory where numpy arrays are to be cached",
        default="GRID/videos_npy/",
        type=str,
    )
    return parser.parse_args()
    

if __name__ == "__main__":
    args = arg_parse()
    try:
        os.mkdir(args.out_root)
    except:
        pass
    detector = MTCNN()
    dataset_to_numpy(detector, args.root, args.out_root)