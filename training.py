import numpy as np

import preporcessing as pre
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


#importing dataset

dataset = np.loadtxt('dataset.csv', delimiter=',')

target = dataset[:,0]
inputs = dataset[:,1:5]

#data preprocessing
input = pre.dataNormalization(inputs)

#spliting data into train and test data
trainInput,testInput,trainTarget,testTarget = train_test_split( input, target, test_size = 0.25, random_state = 100)

#fitting clissifier using entropy/gain
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 100)

classifier.fit(trainInput, trainTarget)

#calculating accuracy
predicted_targets = classifier.predict(testInput)
accuracy = accuracy_score(testTarget,predicted_targets)*100

print('accuracy is : ',accuracy,'%')