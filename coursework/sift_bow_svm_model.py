import os
from imblearn.over_sampling import ADASYN
from joblib import dump
import numpy as np
import cv2
from skimage.util import img_as_ubyte
from sklearn import svm, metrics
from sklearn.cluster import KMeans

#================================================
# Training the classifier on the given testing data.
# I have used the instructions from Lab 07(Computer Vision - IN3060) to
# implement the model, but adapted it to the given dataset.
#================================================

X_test_image_filenames = os.listdir('./dataset/test/images')
y_train = []
desc_list = []
sift = cv2.SIFT_create()

for filename in os.listdir('./dataset/train/images'):
    img = cv2.imread(os.path.join('dataset/train/images', filename))
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(img, None)
    if des is None: # If SIFT cannot compute key points for an image, it is skipped entirely
        continue
    desc_list.append(des)

    label_file = open(os.path.join('dataset/train/labels', os.path.splitext(filename)[0] + '.txt'), 'r')
    label = label_file.read()
    y_train.append(label)
des_array = np.vstack(desc_list)

# Clustering the descriptors to create codewords using K-means
k = len(np.unique(y_train)) * 10
kmeans = KMeans(n_clusters=k, random_state=0).fit(des_array)

# Saving the KMeans model to use it for classifying new images
dump(kmeans, 'Models/kmeans_model.joblib')

# Convert descriptors into histograms of codewords for each image.
hist_list = []
for des in desc_list:
    hist = np.zeros(k)

    idx = kmeans.predict(des)
    for j in idx: # normalising entries in the histogram
        hist[j] = hist[j] + (1 / len(des))

    hist_list.append(hist)
hist_array = np.vstack(hist_list)

# Balancing the dataset
X_train_resampled, y_train_resampled = ADASYN().fit_resample(hist_array, y_train)


# Training the classifier
classifier = svm.SVC(kernel="rbf")
classifier.fit(X_train_resampled, y_train_resampled)
dump(classifier, 'Models/sift_bow_svm_model.joblib')


#================================================
# Testing the classifier on the given testing data
#================================================

hist_list = []
y_test = []
for filename in os.listdir('./dataset/train/images'):
    img = cv2.imread(os.path.join('dataset/train/images', filename))
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(img, None)
    if des is None:
        continue

    label_file = open(os.path.join('dataset/train/labels', os.path.splitext(filename)[0] + '.txt'), 'r')
    label = label_file.read()
    y_test.append(label)

    hist = np.zeros(k)
    idx = kmeans.predict(des)
    for j in idx:
        hist[j] = hist[j] + (1/len(des))

    hist_list.append(hist)

hist_array = np.vstack(hist_list)
y_pred = classifier.predict(hist_array)
y_pred = y_pred.tolist()

print(f"""Classification report for classifier {classifier}:\n
      {metrics.classification_report(y_test, y_pred)}""")



