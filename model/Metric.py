import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score


class MeanRecall:
    def __init__(self, predict, label):
        self.y_preds = predict
        self.y_labels = label
    def get_mean_recall(self):
        return recall_score(self.y_labels, self.y_preds, average='macro')


class Acc:
    def __init__(self, predict, label):
        self.y_preds = predict
        self.y_labels = label
    def get_acc(self):
        correct_num = 0
        for i in range(len(self.y_preds)):
            correct_num += (self.y_preds[i] == self.y_labels[i])
        return correct_num / len(self.y_preds)
