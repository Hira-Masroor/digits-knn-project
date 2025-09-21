# digits-knn-project
My first ever machine learning project using KNN (k-nearest neighbors) to recognize handwritten digits.

The aim of this project is to help me understand how Machine Learning works as this is my first time ever. This project is perfect for beginners because it includes the full process of it (loading data, training, testing, checking accuracy).

KNN (K-Nearest Neighbors) is basically a machine learning algorithm where it predicts something new by looking at the most similar examples and choosing the majority label.

Everyone's handwriting looks different, so in this project we want a model that can understand different handwritings and understand what number it is (0-9).

We'll be doing it by:
Loading the "digits" dataset (which contains 8x8 pixel images of handwritten numbers) each of these images come with the correct label
Training the model by showing part of the dataset so it can "learn" patterns
Testing the model by giving it new, unseen digit images (of numbers in another handwriting) and ask it to guess the number, then we check if it's right
KNN's role: to guess the correct label of the new handwritten number, the model looks at the "k nearest neighbors" (similar digit images from training) and decides by majority vote. example: New digit looks most like 3 training images (5,5,8), since the image looks mostly like 5, it predicts "5". 
Lastly, we measure accuracy (what % of test digits it got right). that shows us how well it learned. 

We can choose how many images the model would look at, that affects the accuracy score. 

Part 1

from sklearn.datasets import load_digits
# from sklearn.datasets import load_digits, which gives ready-made dataset of handwritten digits (0-9) which are used for teaching a computer how to recognize numbers -> later used depending on goal of project.
from sklearn.model_selection import train_test_split #splits data into "train" and "test" sets
from sklearn.neighbors import KNeighborsClassifier # the ML algorithm
from sklearn.metrics import accuracy_score # measure how well model predicts

Part 2

digits = load_digits()
X, y = digits.data, digits.target # X is the pixel value of each image, y is which digit is actually is (labels)

Part 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# This part is used for splitting the data: test_size=0.2 20% of data is for testing, rest is for training, random_state=42 ensures the split is consistent.

Part 4
Training the model

model = KNeighborsClassifier(n_neighbors=3) # looks at the 3 closest points to decide the label
model.fit(X_train, y_train) # actually trains the model on the training data

Part 5

y_pred = model.predict(X_test) # the model guesses labels for unseen data
print("Accuracy:", accuracy_score(y_test, y_pred)) # compares guesses to real answers, then gives a accuracy score

print(X_train.shape, X_test.shape)
