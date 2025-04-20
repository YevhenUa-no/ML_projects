import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import contextlib

st.title("Decision Tree Classifier")

st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Data Source:", ["Upload CSV File", "Use Iris Dataset"])

if data_source == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data:")
            st.dataframe(df.head())

            # Allow user to select target and feature columns
            st.sidebar.header("Column Selection")
            all_columns = df.columns.tolist()
            target_column = st.sidebar.selectbox("Select Target Column:", all_columns)
            feature_columns = st.sidebar.multiselect("Select Feature Columns:", all_columns, default=[col for col in all_columns if col != target_column])

            if target_column and feature_columns:
                X = df[feature_columns]
                y = df[target_column]
                data_loaded = True
            else:
                st.warning("Please select both target and feature columns.")
                data_loaded = False

        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            data_loaded = False
    else:
        st.info("Please upload a CSV file to proceed.")
        data_loaded = False

elif data_source == "Use Iris Dataset":
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = iris.data
    y = iris.target
    target_column = iris.target_names.tolist()
    feature_columns = iris.feature_names
    st.subheader("Iris Dataset Loaded:")
    st.dataframe(df.head())
    data_loaded = True

if data_loaded:
    st.sidebar.header("Model Parameters")
    criterion = st.sidebar.selectbox("Select Criterion:", ["gini", "entropy"])
    max_depth = st.sidebar.slider("Max Depth:", min_value=1, max_value=20, value=None)
    min_samples_split = st.sidebar.slider("Min Samples Split:", min_value=2, max_value=20, value=2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf:", min_value=1, max_value=20, value=1)

    # Train the Decision Tree model
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    clf.fit(X, y)

    st.subheader("Trained Decision Tree:")
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=feature_columns, class_names=[str(c) for c in np.unique(y)], ax=ax)
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance:")
    importance = clf.feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_columns, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    st.dataframe(feature_importance)

    # Prediction Section
    st.subheader("Make Predictions:")
    st.write("Enter the feature values to predict the target.")

    prediction_inputs = {}
    for feature in feature_columns:
        # Try to infer input type based on the data
        if pd.api.types.is_numeric_dtype(X[feature]):
            min_val = X[feature].min()
            max_val = X[feature].max()
            prediction_inputs[feature] = st.number_input(f"{feature} (min: {min_val:.2f}, max: {max_val:.2f})", min_value=min_val, max_value=max_val)
        else:
            unique_vals = X[feature].unique().tolist()
            prediction_inputs[feature] = st.selectbox(f"{feature}", unique_vals)

    if st.button("Predict"):
        input_data = pd.DataFrame([prediction_inputs])
        # Ensure the order of columns matches the training data
        input_data = input_data[feature_columns]
        prediction = clf.predict(input_data)

        st.write("Prediction:")
        if data_source == "Use Iris Dataset":
            st.write(f"The predicted class is: {iris.target_names[prediction[0]]}")
        else:
            st.write(f"The predicted class is: {prediction[0]}")

else:
    st.info("Please select a data source to start.")
