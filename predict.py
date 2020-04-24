import numpy as np
import skvideo.io
from mtcnn import MTCNN
from decoder import spellchecked_decoder
from lipnet import LipNet
from preprocess import crop_mouth


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
                    max_string=32
                    ).compile_model().load_weights(args.weights_path)
    #Swap dims to TxWxHxC, normalize and unsqueeze
    x_data = np.swapaxes(video_data, 1, 2)
    x_data = x_data.astype(np.float32) / 255
    x_data = x_data[np.newaxis, :]
    
    y_pred = lipnet.predict(x_data)
    
    #Decode dense prediction
    input_lengths=np.array([75])
    decoder = spellchecked_decoder(args.dict_path)
    results = decoder.decode(y_pred, input_lengths)
    print(results)


if __name__ == "__main__":
    main()