import os
from imblearn.over_sampling import ADASYN
from skimage.feature import hog
from skimage.util import img_as_ubyte
from sklearn import metrics, svm
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from PIL import Image
import cv2

#================================================
# Training the classifier on the given testing data.
#================================================

X_train_image_filenames = os.listdir('./dataset/train/images')
X_test_image_filenames = os.listdir('./dataset/test/images')
y_train = []
hog_features = []

for filename in X_train_image_filenames:
    label_file = open(os.path.join('dataset/train/labels', os.path.splitext(filename)[0] + '.txt'), 'r')
    label = label_file.read()
    y_train.append(label)

    img = cv2.imread(os.path.join('dataset/train/images', filename))
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (20,20))

    hog_des, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=True, channel_axis=None)
    hog_features.append(hog_des)

hog_features_array = np.vstack(hog_features)

# Balancing the dataset
X_train_resampled, y_train_resampled = ADASYN().fit_resample(hog_features_array, y_train)

classifier = svm.SVC(kernel="rbf")
classifier.fit(X_train_resampled, y_train_resampled)
dump(classifier, 'Models/hog_svm_model.joblib')


#================================================
# Testing the classifier on the given testing data
#================================================
y_test = []
hog_features = []
for filename in X_test_image_filenames:
    img = cv2.imread(os.path.join('dataset/test/images', filename))
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (20,20))

    hog_des, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=True, channel_axis=None)
    hog_features.append(hog_des)

    label_file = open(os.path.join('dataset/test/labels', os.path.splitext(filename)[0] + '.txt'), 'r')
    label = label_file.read()
    y_test.append(label)

X_test = np.array(hog_features)
y_pred = classifier.predict(X_test)
y_pred = y_pred.tolist()

# Compare labels with predicted labels
print(f"""Classification report for classifier {classifier}:\n
          {metrics.classification_report(y_test, y_pred)}""")





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




