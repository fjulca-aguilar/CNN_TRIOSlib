import numpy as np
from trios.WOperator import Classifier
from sklearn.model_selection import train_test_split
import trios.util



class TFClassifier(Classifier):

    def __init__(self, cls=None, dtype=np.uint8):
        self.cls = cls
        self.minimize = False
        self.ordered = True
        self.dtype = dtype

    def train(self, dataset, kw):
        x, y = dataset
        y = y / 255

        y = np.reshape(y, (-1, 1))
        x = np.reshape(x, (-1, self.cls.input_shape[0], self.cls.input_shape[1], 1))
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        self.cls.fit(x_train, y_train, x_val, y_val)

    def apply(self, fvector):
        fvector = fvector.reshape((-1, self.cls.input_shape[0], self.cls.input_shape[1], 1))
        return self.cls.predict(fvector)[0]

    def apply_batch(self, fmatrix):
        fmatrix = fmatrix.reshape((-1, self.cls.input_shape[0], self.cls.input_shape[1], 1))
        return self.cls.predict(fmatrix)[:, 0]

    def write_state(self, obj_dict):
        obj_dict['cls'] = self.cls
        obj_dict['min'] = self.minimize
        obj_dict['dtype'] = self.dtype
        obj_dict['ordered'] = self.ordered
        
    def set_state(self, obj_dict):
        self.cls = obj_dict['cls']
        self.minimize = obj_dict['min']
        self.ordered = obj_dict['ordered']
        self.dtype = obj_dict['dtype']