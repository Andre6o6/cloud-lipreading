import os
import random
from batch_generator import BatchGenerator
from label_utils import text_to_labels

def is_npy(path):
    return os.path.isfile(path) and path.split('.')[-1]=="npy"

class Align:
    self.__SILENCE_TOKENS = ['sp', 'sil']
    
    sentence: str
    labels: np.ndarray
    length: int

    def __init__(self, filepath, max_string):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        align = [(int(y[0]) / 1000, int(y[1]) / 1000, y[2]) for y in [x.strip().split(' ') for x in lines]]
        align = [sub for sub in align if sub[2] not in self.__SILENCE_TOKENS]
        
        self.sentence = ' '.join([y[-1] for y in align if y[-1] not in self.__SILENCE_TOKENS])
        unpadded_labels = text_to_labels(sentence)
        self.length = len(unpadded_labels)
        self.labels = np.array(unpadded_labels + ([-1.0] * (max_string - self.length)))
        

#TODO cache?
class DatasetGenerator(object):
    def __init__(self, dataset_path, aligns_path, batch_size, val_ratio, max_string=32):
        self.dataset_path = dataset_path
        self.aligns_path = aligns_path
        self.batch_size = batch_size
        self.max_string = max_string
        self.val_ratio = val_ratio
        
        self.train_generator = None
        self.val_generator = None
        
        self.build_dataset()
        
    def build_dataset(self):
        subjects_videos = self.get_subjects_videos()
        train_videos, val_videos = self.train_val_split(subjects_videos)
        
        train_aligns = self.generate_align_hash(train_videos)
        val_aligns = self.generate_align_hash(val_videos)
        
        self.train_generator = BatchGenerator(train_videos, train_aligns, self.batch_size)
        self.val_generator = BatchGenerator(val_videos, val_aligns, self.batch_size)

    def get_subjects_videos(self):
        subjects_videos = []
        for subjects in os.listdir(self.dataset_path):
            subj_dir = os.path.join(self.dataset_path, subjects)
            subj_videos = [os.path.join(subj_dir, file) for file in os.listdir(subj_dir) if is_npy(file)]
            random.shuffle(subj_videos)
            subjects_videos.append(subj_videos)
        return subjects_videos

    #TODO split by subjects to validate on unseen
    def train_val_split(self, subjects_videos):
        train_videos = []
        val_videos = []
        for subj_videos in subjects_videos:
            split_idx = int(self.val_ratio*len(subj_videos))
            train_videos += subj_videos[split_idx:]
            val_videos += subj_videos[:split_idx]
        return train_videos, val_videos

    def generate_align_hash(self, videos):       
        align_hash = {}
        for path in videos:
            video_name = os.path.basename(path).split('.')[0]
            align_path = os.path.join(self.aligns_path, video_name) + '.align'  #FIXME
            align_hash[video_name] = Align(align_path, self.max_string)
        return align_hash