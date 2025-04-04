import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import io
import contextlib

st.title("Iris Flower Classification with Decision Tree")

# Theoretical Explanation
st.header("Decision Tree Theory")
st.markdown("""
A Decision Tree is a supervised machine learning algorithm that builds a tree-like model to make decisions. It works by recursively partitioning the data based on feature values. Each node in the tree represents a decision based on a feature, and each branch represents a possible outcome.

**Key Concepts:**

* **Nodes:** Represent decisions based on feature values.
* **Branches:** Represent possible outcomes of the decisions.
* **Leaves:** Represent the final predicted class or value.
* **Root Node:** The topmost node, representing the best feature to split the data.
* **Internal Nodes:** Nodes between the root and leaves, representing intermediate decisions.

**Splitting Criteria:**

Decision Trees use splitting criteria to determine the best feature to split the data at each node. Common criteria include:

* **Gini Impurity:** Measures the impurity of a set of labels. A Gini score of 0 indicates perfect purity (all labels are the same). The formula for Gini Impurity is:

    $$Gini = 1 - \sum_{i=1}^{n} P_i^2$$

    Where $P_i$ is the probability of an object being classified to a particular class.

* **Entropy:** Measures the disorder or randomness in a set of labels. Entropy of 0 indicates a perfectly homogeneous set. The formula for Entropy is:

    $$Entropy = - \sum_{i=1}^{n} P_i \log_2(P_i)$$

    Where $P_i$ is the probability of an object being classified to a particular class.

* **Information Gain:** Measures the reduction in entropy after splitting the data. The feature with the highest information gain is chosen for splitting.

**How Decision Trees Work:**
# Add GIF
st.image("ML_DecTree.gif")

1.  **Start at the root node:** Select the best feature to split the data based on the splitting criteria.
2.  **Create branches:** Divide the data into subsets based on the feature values.
3.  **Recursively repeat:** Apply the same process to each subset until a stopping condition is met (e.g., maximum depth, minimum samples per leaf).
4.  **Assign labels to leaves:** Assign the majority class or average value to each leaf node.

**Advantages:**

* Easy to understand and interpret.
* Can handle both categorical and numerical data.
* Requires minimal data preprocessing.

**Disadvantages:**

* Prone to overfitting, especially with deep trees.
* Can be sensitive to small variations in the data.
* Can create complex trees that are difficult to interpret.

""")

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
