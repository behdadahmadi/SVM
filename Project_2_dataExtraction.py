"""
@author: Corrado 
"""

"""
this is the code you need to run to import data.
You may have to change line 36 selecting the correct path.
"""
import os
import gzip
import numpy as np

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels




"""
To train a SVM in case of binary classification you have to convert the labels of the two classes of interest into '+1' and '-1'.
"""
