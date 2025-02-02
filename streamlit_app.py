import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    st.title("Interactive SVM Classifier on the Iris Dataset")
    st.write(
        """
        This Streamlit app allows you to experiment with the hyperparameters for a Support Vector Machine (SVM) model and see its performance on the Iris dataset.  
        Adjust the settings in the sidebar and watch how the decision boundary and performance metrics change!
        """
    )

    # ================================
    # Sidebar: Hyperparameter Selection
    # ================================
    st.sidebar.header("SVM Hyperparameters")
    
    # Choose the kernel
    kernel = st.sidebar.selectbox("Select Kernel", options=["Linear", "Poly", "RBF", "Sigmoid"], index=2)
    
    # Regularization parameter C
    C = st.sidebar.slider("C (Regularisation parameter)", 0.01, 10.0, 1.0)
    
    # For the polynomial kernel, choose the degree
    if kernel == "poly":
        degree = st.sidebar.slider("Degree (for polynomial kernel)", 2, 5, 3)
    else:
        degree = 3  # default value (not used if kernel != 'poly')
    
    # Gamma parameter for kernels that use it
    if kernel in ["rbf", "poly", "sigmoid"]:
        gamma_option = st.sidebar.selectbox("Gamma Option", options=["Scale", "Auto", "Manual"], index=0)
        if gamma_option == "Manual":
            gamma = st.sidebar.slider("Gamma value", 0.001, 1.0, 0.1)
        else:
            gamma = gamma_option
    else:
        gamma = "scale"  # Not used for linear kernel

    # Test set size for splitting the dataset
    test_size = st.sidebar.slider("Test Set Size (Fraction)", 0.1, 0.5, 0.2, step=0.05)

    # ================================
    # Sidebar: Feature Selection for Visualization
    # ================================
    st.sidebar.header("Feature Selection for Visualization")
    iris = datasets.load_iris()
    feature_names = iris.feature_names
    x_feature = st.sidebar.selectbox("X-axis feature", options=feature_names, index=0)
    y_feature = st.sidebar.selectbox("Y-axis feature", options=feature_names, index=1)
    if x_feature == y_feature:
        st.sidebar.warning("Please select two different features for visualization.")

    # ================================
    # Load and Display the Dataset
    # ================================
    X = iris.data
    y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y

    st.subheader("Iris Dataset (first 5 rows)")
    st.write(df.head())

    # Use only the two selected features (for visualization and model training)
    X_selected = df[[x_feature, y_feature]].values

    # ================================
    # Split the Data
    # ================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=42, stratify=y
    )

    # ================================
    # Create and Train the SVM Model
    # ================================
    if kernel == "poly":
        model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, probability=True)
    else:
        model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

    model.fit(X_train, y_train)

    # ================================
    # Evaluate the Model
    # ================================
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"**Accuracy on test set:** {accuracy:.2f}")

    # Display classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    st.text("Classification Report:")
    st.write(pd.DataFrame(report).transpose())

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    st.pyplot(fig_cm)

    # ================================
    # Plot the Decision Boundary
    # ================================
    st.subheader("Decision Boundary")
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', s=50, cmap=plt.cm.coolwarm)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title("Decision Boundary on Training Data")
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    st.pyplot(fig)

    # ================================
    # Article Link Section (at the bottom)
    # ================================
    st.markdown("---")
    st.markdown("### Further Reading")
    st.markdown(
        "For a detailed discussion on SVMs, check out my [Medium article](https://blochai.medium.com/)."
    )

    # ================================
    # Add Footer
    # ================================
    footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: black;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
      <p>© 2025 Bloch AI LTD - All Rights Reserved. <a href="https://www.bloch.ai" style="color: white;">www.bloch.ai</a></p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
