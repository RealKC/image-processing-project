import pickle

import numpy as np
from sklearn.svm import LinearSVC
import os
import tqdm

def train_svm(hog_folder1, hog_folder2, hog_folder3, feature_length=3000):
    def process_folder(folder, label):
        X = []
        y = []

        for filename in tqdm.tqdm(os.listdir(folder)):
            with open(os.path.join(folder, filename), 'r') as f:
                features = np.array([float(x) for x in f.read().split()])

            # Resize or pad features to a fixed length
            if len(features) < feature_length:
                features = np.pad(features, (0, feature_length - len(features)), mode='constant')
            elif len(features) > feature_length:
                features = features[:feature_length]

            X.append(features)
            y.append(label)

        return X, y

    X1, y1 = process_folder(hog_folder1, 0) 
    X2, y2 = process_folder(hog_folder2, 1) 
    X3, y3 = process_folder(hog_folder3, 2) 

    X = np.concatenate((X1, X2, X3), axis=0) 
    y = np.concatenate((y1, y2, y3)) 

    clf = LinearSVC()
    clf.fit(X, y)

    model_filename = open('svm_model.pkl', 'wb')
    pickle.dump(clf, model_filename)
    print(f"Trained SVM model saved as '{model_filename}'")


train_svm('HOG/HOG_01', 'HOG/HOG_02', 'HOG/HOG_03')
