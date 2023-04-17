import os
from collections import Counter

import tf as tf
from skimage.feature import orb, hog, sift
from skimage.util import img_as_ubyte
from sklearn import metrics, svm
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from PIL import Image
import cv2

X_train_image_filenames = os.listdir('./dataset/train/images')
X_test_image_filenames = os.listdir('./dataset/test/images')
y_train = []
hog_features = []
hog_images = []

for filename in X_train_image_filenames:
    img = cv2.imread(os.path.join('dataset/train/images', filename))
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (20,20))

    # ----- TODO try this tomorrow, maybe it can fix the whole label thing. x
    #kp, des = sift.detectAndCompute(img, None)
    #if des is None:
        #continue

    hog_des, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=True, channel_axis=None)
    hog_images.append(hog_image)
    hog_features.append(hog_des)



    label_file = open(os.path.join('dataset/train/labels', os.path.splitext(filename)[0] + '.txt'), 'r')
    label = label_file.read()
    y_train.append(label)


to_display = np.asarray(Image.open('./dataset/train/images/' + X_train_image_filenames[51]))
plt.imshow(hog_images[51])
#plt.imshow(to_display)

# Create a classifier: Multi-Layer Perceptron
X_train = np.array(hog_features)

#classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,solver='sgd', verbose=True, random_state=1,learning_rate_init=.1)
classifier = svm.SVC(kernel="rbf")
classifier.fit(X_train, y_train)


# ---- Getting the testing data
X_test = []
y_test = []
hog_features = []
hog_images = []
for filename in X_test_image_filenames:
    img = cv2.imread(os.path.join('dataset/test/images', filename))
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (20,20))

    hog_des, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=True, channel_axis=None)
    hog_images.append(hog_image)
    hog_features.append(hog_des)

    label_file = open(os.path.join('dataset/test/labels', os.path.splitext(filename)[0] + '.txt'), 'r')
    label = label_file.read()
    y_test.append(label)



X_test = np.array(hog_features)
y_pred = classifier.predict(X_test)
#y_pred = y_pred.tolist()


# Compare labels with predicted labels
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

print(Counter(y_test))
print(Counter(y_train))

print(f"""Classification report for classifier {classifier}:\n
      {metrics.classification_report(y_test, y_pred)}""")