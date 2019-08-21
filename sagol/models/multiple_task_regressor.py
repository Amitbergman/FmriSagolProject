from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from itertools import combinations
import numpy as np

class Multiple_Task_Regressor(BaseEstimator, RegressorMixin):  

    def __init__(self, weak_learner=SVR, hyper_params={}, tasks_in_weak_learner=1, first_index_of_task=None):
        self.weak_learner = weak_learner # sklearn model class (e.g SVR)
        self.hyper_params = hyper_params # Dictionary of hyper parameters according to the model (can be empty)
        
        # Each weak learner will be trained on 'tasks_in_weak_learner' contrasts (so the number of weak learners will be: 'number of
        # tasks' choose tasks_in_weak_learner)
        self.tasks_in_weak_learner = tasks_in_weak_learner 
        
        self.first_index_of_task = first_index_of_task # This is needed in order to get the contrast from the X data
        
        self.weak_learners = None
        
    
    # X here is a list of tuples of the form: (instance's features, instance's contrast). Do not be afraid to change it carefully if
    # necessary (you have to change it in predict method too)
    def fit(self, X, y=None):
        if self.first_index_of_task is None:
            raise RuntimeError("You must pass as parameter the first index of task! Look at the constructor of this class!")
        T = [np.where(x[self.first_index_of_task:] == 1)[0][0] for x in X]
        tasks = sorted(set(T)) # The unique tasks
            
        # 'tasks' choose 'tasks_in_weak_learner' combinations of contrasts
        combs = list(combinations(tasks, self.tasks_in_weak_learner))
        if len(combs) > 1000: # To prevent collapsing, fitting more than this number of regressors is not allowed
            raise RuntimeError("Currently, fitting more than 1000 regressors is not allowed!")
        
        # A dict of combination and X and y belong to one of the contrasts
        X_y_new = {comb:([], []) for comb in combs}
        for i in range(len(X)):
            for item in X_y_new.items():
                if T[i] in item[0]:
                    item[1][0].append(X[i])
                    item[1][1].append(y[i])
        
        weak_learners = {}
        for comb in combs:
            model = self.weak_learner(**self.hyper_params) # Generating a weak learner with given hyper params
            X_temp, y_temp = X_y_new[comb] # Training on X and y belong to the combination only
            
            # test size can be a given param in the future
            X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.25)
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test) # The score of the training (which is evaluated upon the validation data)
            model.fit(X_temp, y_temp) # After evaluation upon unseen data, training the model on all data
            
            # The output model is saved as a weak learner and its score will be used as its weight in future predictions
            # (if score is less than or equal to 0, it will not be used at all)
            weak_learners[comb] = (model, score)

        self.weak_learners = weak_learners
        return self

    
    # X here is a list of tuples of the form: (instance's features, instance's contrast). Do not be afraid to change it carefully if
    # necessary (you have to change it in fit method too)
    def predict(self, X):
        if self.weak_learners is None:
            raise RuntimeError("You must train before predicting data!")

        T = [np.where(x[self.first_index_of_task:] == 1)[0][0] for x in X]
        # Predicting X by all weak learners
        wl_preds = {comb: wl[0].predict(X) for comb, wl in self.weak_learners.items()}
        
        # The real predictions are based only on relevant weak learners
        preds = [0] * len(X)
        for i in range(len(X)):
            pred = 0
            total_scores = 0
            for wl in self.weak_learners.items():
                # A weak learner is counted only if the contrasts it consists of contain the the contrast from which the instance
                # was taken and only if its training score is above 0. Its weight in predicting is its training score
                if T[i] in wl[0] and wl[1][1] > 0:
                    pred += wl_preds[wl[0]][i] * wl[1][1]
                    total_scores += wl[1][1]
            preds[i] = (pred / total_scores) if total_scores > 0 else 0 # Normalizing by the sum of all scores taken into account
        return preds

    
    # Score is calculated the same as in SVR
    def score(self, X, y=None):
        y = np.array(y)
        y_pred = np.array(self.predict(X))
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
