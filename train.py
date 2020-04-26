import argparse
import datetime
import os
import time

from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

from dataset_generator import DatasetGenerator
from lipnet import LipNet

def train(run_name, dataset_path, aligns_path):
    lipnet = LipNet().compile_model()
    datagen = DatasetGenerator(dataset_path, aligns_path)
    callbacks = create_callbacks(run_name)
    
    start_time = time.time()
    
    lipnet.model.fit_generator(
        generator = datagen.train_generator,
        validation_data = datagen.val_generator,
        epochs = 1,
        verbose = 1,
        shuffle = True,
        max_queue_size = 5,
        workers = 2,
        callbacks=callbacks,
        use_multiprocessing = True
    )
    elapsed_time = time.time() - start_time
    print('\nTraining completed in: {}'.format(datetime.timedelta(seconds=elapsed_time)))
    
    
def create_callbacks(run_name, log_root='log/', checkpoint_root='results/'):
    create_dir(log_root)
    
    run_log_dir = os.path.join(log_root, run_name)
    create_dir(run_log_dir)
    
    tensorboard = TensorBoard(log_dir=run_log_dir)
    csv_log = os.path.join(run_log_dir, 'training.csv')
    csv_logger = CSVLogger(csv_log, separator=',', append=True)
    
    create_dir(checkpoint_root)
    
    checkpoint_dir = os.path.join(checkpoint_root, run_name)
    create_dir(checkpoint_dir)
    checkpoint_template = os.path.join(checkpoint_dir, "lipnet_{epoch:03d}_{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(checkpoint_template, monitor='val_loss', save_weights_only=True, mode='auto', period=1, verbose=1)


def create_dir(path):
    try:
        os.mkdir(path)
    except:
        pass

        
def arg_parse():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--dataset_path",
        help="Dataset root directory",
        default="GRID/videos_npy/",
        type=str,
    )
    parser.add_argument(
        "--aligns_path",
        help="Directory containing all align files",
        default="GRID/align/",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    run_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    train(run_name, args.dataset_path, args.aligns_path)