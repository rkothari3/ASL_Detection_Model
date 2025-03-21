'''
Prepare your data (train_test_split)

Build a machine learning model (RandomForestClassifier)

Evaluate its performance (accuracy_score)

Save and load your model or data (pickle)
'''
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from pickle file
dataDict = pickle.load(open('./data.pickle', 'rb'))

# Convert data and labels to numpy arrays
data = np.asarray(dataDict['data'])
labels = np.asarray(dataDict['labels'])

# Split the data into training and testing sets
'''
Syntax: sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
'''
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,  # Data is shuffled
    stratify=labels  # Stratify ensures class distribution is preserved
)

'''
Stratify is used when dealing with imbalanced datasets.
Ex: If you have a dataset where 80% of the data belongs to class A and 20% belongs to class B, and you set stratify=y, then the training and testing sets will also have approximately 80% of class A and 20% of class B.
'''

# Create Model
model = RandomForestClassifier()

# Train classifier
model.fit(x_train, y_train)

# Test
y_predict = model.predict(x_test)

# Get a score comparing predictions with tested
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
output_file = 'model.pickle'
with open(output_file, 'wb') as f:  # Save the trained model, not the data
    pickle.dump({'model': model}, f)

print(f"Model serialized and saved to '{output_file}'")