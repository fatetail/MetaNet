import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

def get_confuse_matrix(output, target):
    #chain = itertools.chain(*target)
    #target = list(chain)

    #chain = itertools.chain(*output)
    #output = list(chain)

    with torch.no_grad():
        cm = confusion_matrix(target, output)

   # np.save('confuse_matrix', cm)
    return cm

