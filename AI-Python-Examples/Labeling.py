import numpy as np
from sklearn import datasets
from sklearn import preprocessing

input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

print("\nLabel mappings:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)

print("\nLabels = ", test_labels)
print("Encoded values =", list(encoded_values))

new_encoded_values = [3, 0, 4, 1]
decoded_values = encoder.inverse_transform(new_encoded_values)
print("\nLabels = ", test_labels)
print("Decoded values =", list(decoded_values))
