import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data

from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

project_path = "./"
data_path = os.path.join(project_path, "data", "census.csv")
#print(data_path)
data = pd.read_csv(data_path)
# print(data.shape)
# print(data.columns)
# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.2)
#train, val = train_test_split(train, test_size = 0.2)
print(train.shape, test.shape)
# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    encoder='OneHotEncoder',
    training=True,
    lb='LabelBarbarize'
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)  # your code here

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
print(f"Model saved to {model_path}")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
print(f"Model saved to {encoder_path}")

# load the model
print(f"Loading model from {model_path}")
model = load_model(
    model_path
)

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)  # your code here

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
label = "salary"
# print("cat_features", cat_features)

for col in cat_features:
    # print("sorted",sorted(test[col].unique()))
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        data = test[test[col] == slicevalue]
        # print(data)
        p, r, fb = performance_on_categorical_slice(
            test, col, slicevalue, cat_features, label, encoder, lb, model
            # your code here
            # use test, col and slicevalue as part of the input
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
