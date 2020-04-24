from keras.layers import Input
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.core import Activation, Dense, Flatten, Lambda, SpatialDropout3D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling3D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed

from keras import backend as k
from keras.models import Model
from keras.optimizers import Adam

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return k.ctc_batch_cost(labels, y_pred[:, :, :], input_length, label_length)


class LipNet(object):
    def __init__(self, frame_count, image_channels=3, image_height=50, image_width=100, max_string=32, output_size=28):
        input_shape = self.get_input_shape(frame_count, image_channels, image_height, image_width)
        self.input_layer = Input(shape=input_shape, dtype='float32', name='input')

        self.zero_1 = ZeroPadding3D(padding=(1, 2, 2), name='zero_1')(self.input_layer)
        self.conv_1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv_1')(self.zero_1)
        self.batc_1 = BatchNormalization(name='batc_1')(self.conv_1)
        self.actv_1 = Activation('relu', name='actv_1')(self.batc_1)
        self.pool_1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool_1')(self.actv_1)
        self.drop_1 = SpatialDropout3D(0.5, name='drop_1')(self.pool_1)

        self.zero_2 = ZeroPadding3D(padding=(1, 2, 2), name='zero_2')(self.drop_1)
        self.conv_2 = Conv3D(64, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv_2')(self.zero_2)
        self.batc_2 = BatchNormalization(name='batc_2')(self.conv_2)
        self.actv_2 = Activation('relu', name='actv_2')(self.batc_2)
        self.pool_2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool_2')(self.actv_2)
        self.drop_2 = SpatialDropout3D(0.5, name='drop_2')(self.pool_2)

        self.zero_3 = ZeroPadding3D(padding=(1, 1, 1), name='zero_3')(self.drop_2)
        self.conv_3 = Conv3D(96, (3, 3, 3), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv_3')(self.zero_3)
        self.batc_3 = BatchNormalization(name='batc_3')(self.conv_3)
        self.actv_3 = Activation('relu', name='actv_3')(self.batc_3)
        self.pool_3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool_3')(self.actv_3)
        self.drop_3 = SpatialDropout3D(0.5, name='drop_3')(self.pool_3)

        self.res = TimeDistributed(Flatten())(self.drop_3)

        self.gru_1 = Bidirectional(GRU(256, return_sequences=True, activation=None, kernel_initializer='Orthogonal', name='gru_1'), merge_mode='concat')(self.res)
        self.gru_1_actv = Activation('relu', name='gru_1_actv')(self.gru_1)
        self.gru_2 = Bidirectional(GRU(256, return_sequences=True, activation=None, kernel_initializer='Orthogonal', name='gru_2'), merge_mode='concat')(self.gru_1_actv)
        self.gru_2_actv = Activation('relu', name='gru_2_actv')(self.gru_2)

        self.dense_1 = Dense(output_size, kernel_initializer='he_normal', name='dense_1')(self.gru_2_actv)
        self.y_pred  = Activation('softmax', name='softmax')(self.dense_1)

        self.input_labels = Input(shape=[max_string], dtype='float32', name='labels')
        self.input_length = Input(shape=[1], dtype='int64', name='input_length')
        self.label_length = Input(shape=[1], dtype='int64', name='label_length')

        self.loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([self.y_pred, self.input_labels, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_layer, self.input_labels, self.input_length, self.label_length], outputs=self.loss_out)


    def compile_model(self, optimizer=None):
        if optimizer is None:
            optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
        return self


    def load_weights(self, path: str):
        self.model.load_weights(path)
        return self


    @staticmethod
    def get_input_shape(frame_count, image_channels, image_height, image_width):
        if k.image_data_format() == 'channels_first':
            return image_channels, frame_count, image_width, image_height
        else:
            return frame_count, image_width, image_height, image_channels


    def predict(self, input_batch):
        return self.capture_softmax_output([input_batch, 0])[0]


    @property
    def capture_softmax_output(self):
        return k.function([self.input_layer, k.learning_phase()], [self.y_pred])