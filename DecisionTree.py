import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import io
import contextlib

st.title("Iris Flower Classification with Decision Tree")

# Objective and Steps
st.header("Objective")
st.write("Classify flowers based on sepal and petal dimensions using a Decision Tree.")

st.header("Steps")
st.markdown("""
1.  **Load the Iris dataset:** Extract data on sepal and petal dimensions.
2.  **Initialize the model:** Create a `DecisionTreeClassifier` instance.
3.  **Train the model:** Fit the model to the dataset.
4.  **Visualize the tree:** Display the trained decision tree using `plot_tree`.
5.  **Make predictions:** Allow the user to input flower dimensions and predict the flower class.
""")

# Dataset Information
st.header("The Dataset: Iris Dataset")
st.markdown("""
* 150 data points
* 3 classes (Setosa, Versicolor, Virginica) with 50 data points each
* No explicit test data (for simplicity, we'll train on the entire dataset)
* Columns: Sepal Length, Sepal Width, Petal Length, Petal Width, Class (0-1-2)
""")

# Code to be executed and displayed
code = """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
iris = load_iris()
x, y = iris.data, iris.target

# Create and train the Decision Tree classifier
clf = DecisionTreeClassifier(criterion="gini")
clf.fit(x, y)

# Visualization
plt.figure(figsize=(15, 12))
plot_tree(clf, filled=True,
          feature_names=iris.feature_names,
          class_names=iris.target_names)
"""

st.code(code, language="python")

# Execute the code and capture the plot
try:
    with contextlib.redirect_stdout(io.StringIO()) as code_output:
        iris = load_iris()
        x, y = iris.data, iris.target
        clf = DecisionTreeClassifier(criterion="gini")
        clf.fit(x, y)
        fig, ax = plt.subplots(figsize=(15, 12)) #Create a figure and axes object.
        plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax = ax) #plot tree into the axes object.
    st.pyplot(fig)  # Display the plot
    if code_output.getvalue():
        st.text("Code Output:")
        st.text(code_output.getvalue())
except Exception as e:
    st.error(f"An error occurred: {e}")

# Prediction Section
st.header("Predict Flower Class")
st.write("Enter the sepal and petal dimensions to predict the flower class.")

sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.5)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3)

if st.button("Predict"):
    prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    class_names = iris.target_names
    st.write(f"The predicted flower class is: {class_names[prediction[0]]}")