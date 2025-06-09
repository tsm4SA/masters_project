from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# load data
iris = load_iris()
X = iris.data
y = (iris.target == 2).astype(int)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify=y, random_state= 4337)

# logistic regression model
model = LogisticRegression(max_iter= 100)

# fit to training data
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report")
print(classification_report(y_test, y_pred))