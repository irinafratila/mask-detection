import os
from collections import Counter
from imblearn.over_sampling import ADASYN
from joblib import dump

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from skimage.util import img_as_ubyte
from sklearn import svm, metrics
from sklearn.cluster import KMeans

X_train_image_filenames = os.listdir('./dataset/train/images')
X_test_image_filenames = os.listdir('./dataset/test/images')

# ---- Getting training data and training the SVM model
y_train = []
desc_list = []
sift = cv2.SIFT_create()
for filename in os.listdir('./dataset/train/images'):
    img = cv2.imread(os.path.join('dataset/train/images', filename))
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(img, None)
    if des is None:
        continue

    desc_list.append(des)
    label_file = open(os.path.join('dataset/train/labels', os.path.splitext(filename)[0] + '.txt'), 'r')
    label = label_file.read()
    y_train.append(label)

des_array = np.vstack(desc_list)

# Step 3: clustering the descriptors to create codewords using K-means
k = len(np.unique(y_train)) * 10
kmeans = KMeans(n_clusters=k, random_state=0).fit(des_array)

#Step 4: Generating histograms of words, one for each image
# Convert descriptors into histograms of codewords for each image.
hist_list = []
idx_list = []

for des in desc_list:
    hist = np.zeros(k)

    idx = kmeans.predict(des)
    idx_list.append(idx)

    for j in idx: # normalising entries in the histogram
        hist[j] = hist[j] + (1 / len(des))

    hist_list.append(hist)

hist_array = np.vstack(hist_list)

# ---- Balancing the dataset
X_train_resampled, y_train_resampled = ADASYN().fit_resample(hist_array, y_train)
print(Counter(y_train_resampled))



# Training a classifier
classifier = svm.SVC(kernel="rbf")
classifier.fit(hist_array, y_train)


# --- Testing the model on testing data
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

# Compare the actual labels with predicted labels by the model
fig, axes = plt.subplots(2, 5, figsize=(14, 7), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    test_img = np.asarray(Image.open('./dataset/test/images/' + X_test_image_filenames[i]))
    ax[i].imshow(test_img)
    ax[i].set_title(f'Label: {y_test[i]} \n Prediction: {y_pred[i]}')
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()



print(f"""Classification report for classifier {classifier}:\n
      {metrics.classification_report(y_test, y_pred)}""")

dump(classifier, 'bow_svm_model.joblib')




