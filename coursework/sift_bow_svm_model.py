import os
from collections import Counter
from PIL import Image
from imblearn.over_sampling import ADASYN
from joblib import dump
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.util import img_as_ubyte
from sklearn import svm, metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

#================================================
# Training the classifier on the given testing data.
# I have used the instructions from Lab 07(Computer Vision - IN3060)
# to implement the model and visualise the results,
# but adapted it to the given dataset.
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

    # Taking the image name from the variable 'filename' without the '.jpeg' and adding
    # '.txt' at the end instead. Then opening that file to get the label.
    label_file = open(os.path.join('dataset/train/labels', os.path.splitext(filename)[0] + '.txt'), 'r')
    label = label_file.read()
    y_train.append(label)
des_array = np.vstack(desc_list)


# Clustering the descriptors to create codewords using K-means
k = len(np.unique(y_train)) * 10
batch_size = des_array.shape[0] // 4
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(des_array)

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
classifier = svm.SVC(kernel="rbf", C=1, gamma='scale')
classifier.fit(X_train_resampled,y_train_resampled)
dump(classifier, 'Models/sift_bow_svm_model.joblib')


#================================================
# Testing the classifier on the given testing data
#================================================

hist_list = []
y_test = []
for filename in os.listdir('./dataset/test/images'):
    img = cv2.imread(os.path.join('dataset/test/images', filename))
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(img, None)
    if des is None:
        continue

    label_file = open(os.path.join('dataset/test/labels', os.path.splitext(filename)[0] + '.txt'), 'r')
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

#================================================
# Matplotlib figures to help visualise the predicted
# performance of the label.
#================================================

fig, axes = plt.subplots(2, 5, figsize=(14, 7), sharex=True, sharey=True)
ax = axes.ravel()
image_index = 10

for i in range(10):
    test_img = np.asarray(Image.open('./dataset/test/images/' + X_test_image_filenames[image_index]))
    ax[i].imshow(test_img)
    ax[i].set_title(f'Label: {y_test[image_index]} \n Prediction: {y_pred[image_index]}')
    ax[i].set_axis_off()
    image_index = image_index + 1

fig.tight_layout()
#plt.show()

metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
#plt.show()


