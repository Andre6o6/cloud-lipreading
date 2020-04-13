import datetime

from lipnet import LipNet

#TODO
class DatasetGenerator(object):
    def __init__(self):
        self.train_generator = None
		self.val_generator = None
    

def train():
    lipnet = LipNet().compile_model()
    datagen = DatasetGenerator()
    
    start_time = time.time()
    
    lipnet.model.fit_generator(
		generator = datagen.train_generator,
		validation_data = datagen.val_generator,
		epochs = 1,
		verbose = 1,
		shuffle = True,
		max_queue_size = 5,
		workers = 2,
		#callbacks=callbacks,
		use_multiprocessing = True
	)
    elapsed_time = time.time() - start_time
	print('\nTraining completed in: {}'.format(datetime.timedelta(seconds=elapsed_time)))