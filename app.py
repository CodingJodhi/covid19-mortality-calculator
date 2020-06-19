from __future__ import absolute_import, division, print_function, unicode_literals
from flask import Flask, render_template, request, url_for
import pandas as pd
# noinspection PyUnresolvedReferences
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
import csv


def model():
    dftrain = pd.read_csv("dataset1.csv")
    y_train = dftrain.pop("death")
    dfeval = pd.read_csv("dataset2.csv")
    y_eval = dfeval.pop("death")

    categorical_columns = ["gender", "country"]
    feature_columns = []

    for feature_name in categorical_columns:
        vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    feature_columns.append(tf.feature_column.numeric_column("age", dtype=tf.float32))

    def make_input_fn(data_df, label_df, num_epochs=30, shuffle=True, batch_size=100):
        def input_function():  # inner function, this will be returned
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            # create tf.data.Dataset object with data and its label
            if shuffle:
                ds = ds.shuffle(1000)  # randomize order of data
            ds = ds.batch(batch_size).repeat(num_epochs)
            # split dataset into batches of 100 and repeat process for number of epochs
            return ds  # return a batch of the dataset
        return input_function  # return a function object for use

    train_input_fn = make_input_fn(dftrain, y_train)
    # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    linear_est.train(train_input_fn)  # train
    accuracy = linear_est.evaluate(eval_input_fn)["accuracy"]  # get accuracy of model to display
    return accuracy, make_input_fn, linear_est


accuracy, make_input_fn, linear_est = model()
accuracy = str(round(accuracy * 100, 3)) + "%"
app = Flask(__name__)
app.secret_key = b"5T\rE\xbf\n\x0e\xdcr\xaf\xea\x85\xe9\xe0\x8a\xf9"
PARAMETERS = ["age", "gender", "country"]


@app.route("/")
def index():
    return render_template("home.html", accuracy=accuracy)


@app.route("/calculate", methods=["POST"])
def calculate():
    for param in PARAMETERS:
        if not request.form.get(param):
            return render_template("failure.html", error=param)

    with open("temp.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["gender", "age", "country"])
        writer.writerow([request.form.get("gender"), request.form.get("age"), request.form.get("country")])

    temp_stats = pd.read_csv("temp.csv")
    predict_input_fn = make_input_fn(temp_stats, [0], num_epochs=1, shuffle=False)
    result = linear_est.predict(predict_input_fn)
    for r in result:
        result = r["probabilities"][1]
        result = str(result) + "%"
    return render_template("calculate.html", age=request.form.get("age"), gender=request.form.get("gender"),
                           country=request.form.get("country"), result=result)


if __name__ == "__main__":
    app.run(debug=True)
