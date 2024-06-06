from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

np.random.seed(42)

@dataclass
class DecisionTree:
    criterion: Literal["entropy", "gini", "mse"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=50, min_samples=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.dictionary = {}

    # Helper Functions
    def check_unique(self, y : pd.Series) -> bool :
        if len(y.unique()) == 1 :
            return True
        return False

    def mode(self, y: pd.Series) :
        a = np.unique(y, return_counts=True)
        return a[0][np.argmax(a[1])]

    def _fit(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> dict:
        # Base Case
        if (self.check_unique(y)) or (depth == self.max_depth) or (len(y) <= self.min_samples):
            if (check_ifreal(y)) : # Classifier
                return self.mode(y)
            return np.mean(y) # Regressor
        
        split_val, feature = opt_split_attribute(X, y, self.criterion, X.columns)
        poss_splits = split_data(X, y, feature, split_val)

        if not check_ifreal(X[feature]) : # Discrete values in feature
            question = feature
            answer = {}
            for i, val in enumerate(split_val) :
                poss_splits[i].drop(feature, axis=1)
                y1 = poss_splits[i].pop(poss_splits[i].columns[-1])
                answer[val] = self._fit(poss_splits[i], y1, depth+1)
                
        else : # Real values in feature
            # print(poss_splits[0])
            # print(poss_splits[1])
            y1 = poss_splits[0].pop(poss_splits[0].columns[-1])
            y2 = poss_splits[1].pop(poss_splits[1].columns[-1])
            question = "{} <= {}".format(feature, poss_splits[0][feature][len(poss_splits[0])-1])
            # print(y1)
            # print(y2)
            answer = {
                        "Yes" : self._fit(poss_splits[0], y1, depth+1),
                        "No" : self._fit(poss_splits[1], y2, depth+1)
                    }
            
        return {question : answer}


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None :
        self.dictionary = self._fit(X, y, 0)


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        y_pred = np.zeros(len(X))

        # Discrete input
        if (not check_ifreal(X[X.columns[0]])) :
            for i in range(len(X)) :
                b = self.dictionary
                while type(b) == type(self.dictionary) :
                    feature = list(b.keys())[0]
                    b = b[feature][X.loc[i,feature]]
                y_pred[i] = b
            return y_pred

        # Real input
        for i in range(len(X)) :
            b = self.dictionary
            while type(b) == type(self.dictionary) :
                x = list(b.keys())[0]
                feature, _, val = x.split()
                val = float(val)
                if X.loc[i, feature] <= val :
                    b = b[x]['Yes']
                else :
                    b = b[x]['No']
            y_pred[i] = b
        return y_pred


    # def plot(self) -> None:
    #     """
    #     Function to plot the tree

    #     Output Example:
    #     ?(X1 <= 4)
    #         Yes: ?(X2 <= 7)
    #             Yes: Class A
    #             No: Class B
    #         No: Class C
    #     Where Y => Yes and N => No
    #     """
    #     pass
    



    def plot(self, d=None, indent=""):
        if d == None :
            d = self.dictionary
        if isinstance(d, dict):
            for key, value in d.items():
                print(f"?({key})")
                if isinstance(value, dict):
                    for k, v in value.items() :
                        print(f"{indent}  {k}: ", end="")
                        self.plot(v, indent + "        ")
        else :
            print(d)