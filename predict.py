import argparse
import numpy as np
import skvideo.io
from mtcnn import MTCNN
from decoder import spellchecked_decoder
from lipnet import LipNet
from preprocess import crop_mouth
from subtitles import render_subtitles

def prepare_batch(data):
    #Swap dims to TxWxHxC, normalize and unsqueeze
    x_data = np.swapaxes(data, 1, 2)
    x_data = x_data.astype(np.float32) / 255
    x_data = x_data[np.newaxis, :]
    return x_data


def get_subs(model, decoder, frames, window=75):
    #If we have less frames than input needs, add blank frames
    f = frames.shape[0]
    if f<75:
        frames = np.concatenate((frames, np.full((75-f,h,w,c),127)))

    subtitles = []
    input_lengths=np.array([window])
    iter_count = frames.shape[0]//window
    #Batch process frames
    for i in range(iter_count):
        x_data = prepare_batch(frames[i*window:(i+1)*window])
        y_pred = model.predict(x_data)
        subs = decoder.decode(y_pred, input_lengths)
        subtitles.extend(subs)
    
    #Batch remaining frames with some already processed from the end
    if frames.shape[0]%window > 0:
        x_data = prepare_batch(frames[-window:])
        y_pred = model.predict(x_data)
        subs = decoder.decode(y_pred, input_lengths)
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
    
    detector = MTCNN()
    video_data = skvideo.io.vread(args.video_path)
    
    #Find mouth on first frame and crop accordingly
    x0,y0,x1,y1 = crop_mouth(detector, video_data[0])
    video_data = video_data[:, y0:y1, x0:x1, :]
    
    #TODO calculate all those constants for random video
    lipnet = LipNet(frame_count=75, 
                    image_channels=3, 
                    image_height=50, 
                    image_width=100, 
                    max_string=64
                    ).compile_model().load_weights(args.weights_path)
    decoder = spellchecked_decoder(args.dict_path)
    
    subs = get_subs(lipnet, decoder, video_data)
    print("Predicted subtitles:")
    print(subs)
    
    if args.save:
        name, ext = args.video_path.split('.')
        out_path = name+"_subbed."+ext
        render_subtitles(args.video_path, out_path, subs)


if __name__ == "__main__":
    main()