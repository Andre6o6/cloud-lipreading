import argparse
import numpy as np
import skvideo.io
from mtcnn import MTCNN
from decoder import spellchecked_decoder
from lipnet import LipNet
from preprocess import crop_mouth
from subtitles import render_subtitles

class Predictor:
    def __init__(self, weights_path, dict_path, window=75):
        self.window = window
        self.detector = MTCNN()
        #TODO calculate all those constants for random video
        self.lipnet = LipNet(
            frame_count=window, 
            image_channels=3, 
            image_height=50, 
            image_width=100, 
            max_string=64
        ).compile_model().load_weights(weights_path)
        self.decoder = spellchecked_decoder(dict_path)
        
    def preprocess(self, video_data):
        #Find mouth on first frame and crop accordingly
        x0,y0,x1,y1 = crop_mouth(self.detector, video_data[0])
        video_data = video_data[:, y0:y1, x0:x1, :]
        
        #Swap dims to TxWxHxC, normalize
        video_data = np.swapaxes(video_data, 1, 2)
        video_data = video_data.astype(np.float32) / 255
        return video_data
    
    def predict_subs(video_data):
        cropped = self.preprocess(video_data)
        
        #If we have less frames than input needs, add blank frames
        f,h,w,c = cropped.shape
        if f<self.window:
            cropped = np.concatenate((cropped, np.full((75-f,h,w,c),127)))

        subtitles = []
        input_lengths=np.array([self.window])
        iter_count = cropped.shape[0]//self.window
        #Batch process frames
        for i in range(iter_count):
            x_data = cropped[i*self.window:(i+1)*self.window]
            y_pred = self.lipnet.predict(x_data[np.newaxis, :])   #unsqueeze
            subs = self.decoder.decode(y_pred, input_lengths)
            subtitles.extend(subs)
        
        #Batch remaining frames with some already processed from the end
        if cropped.shape[0]%self.window > 0:
            x_data = cropped[-self.window:]
            y_pred = self.lipnet.predict(x_data[np.newaxis, :])   #unsqueeze
            subs = self.decoder.decode(y_pred, input_lengths)
            subtitles.extend(subs)
        return subtitles


def arg_parse():
    parser = argparse.ArgumentParser(description="Predict subtitles")
    parser.add_argument(
        "--video",
        dest="video_path",
        help="Video file to predict subs for",
        type=str,
    )
    parser.add_argument(
        "--weights",
        dest="weights_path",
        help="Weights of the model",
        default="weights/weights.hdf5",
        type=str,
    )
    parser.add_argument(
        "--dict",
        dest="dict_path",
        help="Spellcheck dictionary to use in decoding",
        default="dictionaries/grid.txt",
        type=str,
    )
    parser.add_argument(
        "--save", 
        help="Save video with subtitle overlay",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = arg_parse()
    video_data = skvideo.io.vread(args.video_path)
    model = Predictor(args.weights_path, args.dict_path)
    subs = model.predict_subs(video_data)
    print("Predicted subtitles:")
    print(subs)
    
    if args.save:
        name, ext = args.video_path.split('.')
        out_path = name+"_subbed."+ext
        render_subtitles(args.video_path, out_path, subs)


if __name__ == "__main__":
    main()