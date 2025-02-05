#**********************************************
# Interactive SVM Classifier on Iris Dataset App
# Version 1
# 2nd February 2025
# Jamie Crossman-Smith
# jamie@bloch.ai
#**********************************************
# This Python code creates an interactive educational web application using Streamlit.
# The application demonstrates how a Support Vector Machine (SVM) works on the Iris dataset.
# Users can adjust various hyperparameters (e.g. Kernel, Regularisation parameter (C), Degree, Gamma) 
# and select features for 2D visualisation of the decision boundary.
#
# The app displays:
# - The dataset (with class labels linked to actual species: Setosa, Versicolor, Virginica)
# - Model performance metrics such as overall accuracy, and a detailed classification report (including
#   precision, recall, F1-score, and support)
# - A confusion matrix to show correct versus misclassifications
# - Separate decision boundary plots which visualise the regions of the feature space assigned to each class for training and test data.
#
# Educational explanations are provided throughout the app (via sidebar expanders and info boxes)
# to help users understand the effect of each hyperparameter, the significance of each evaluation metric,
# and how to interpret the visual outputs.
#
# This app is intended to enhance understanding of SVMs in a practical, interactive manner.
#**********************************************

# Import necessary libraries for building the Streamlit app and handling data
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import libraries for machine learning and evaluation
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    st.title("Interactive SVM Classifier on the Iris Dataset")
    st.write(
        """
        Welcome to this educational app that demonstrates how a Support Vector Machine (SVM) works on the Iris dataset.
        Experiment with the hyperparameters using the sidebar, and observe how these settings affect the models decision boundary and performance.
        """
    )

    # ================================
    # Sidebar: Hyperparameter Selection
    # ================================
    st.sidebar.header("SVM Hyperparameters")
    
    # Expandable section with educational information on hyperparameters
    with st.sidebar.expander("Learn about SVM Hyperparameters"):
        st.markdown(
            """
            **Kernel:**  
            The kernel determines the transformation applied to your data before finding a decision boundary.  
            - **Linear:** Uses a straight line to separate classes. Best for data that is linearly separable.  
            - **Poly:** Uses a polynomial function. Adjust the degree to control the curvature of the decision boundary.  
            - **RBF:** Uses a radial basis function. Ideal for capturing non-linear relationships in the data.  
            - **Sigmoid:** Mimics the activation function of a neural network.
            
            **C (Regularisation parameter):**  
            Controls the trade-off between maximising the margin (the gap between classes) and minimising misclassification errors.  
            A higher value of C emphasises correct classification of all training points, which can lead to overfitting.
            
            **Degree:**  
            Applicable only for the Poly kernel.  
            Determines the degree of the polynomial used to create the decision boundary.
            
            **Gamma:**  
            Defines the influence of each training example.  
            A low gamma means that the influence reaches far, while a high gamma means it is more localised.  
            For RBF, Poly, and Sigmoid kernels, selecting the right gamma is crucial.
            
            **Test Set Size (Fraction):**  
            Specifies the proportion of the dataset to be reserved for testing.  
            A smaller fraction leaves more data for training, which may improve model performance.
            """
        )

    # Choose the kernel (displayed with capitalised names)
    kernel_options = {"Linear": "linear", "Poly": "poly", "RBF": "rbf", "Sigmoid": "sigmoid"}
    selected_kernel = st.sidebar.selectbox("Select Kernel", options=list(kernel_options.keys()), index=2)
    kernel = kernel_options[selected_kernel]
    
    # Regularisation parameter C
    C = st.sidebar.slider("C (Regularisation parameter)", 0.01, 10.0, 1.0)
    
    # For the Poly kernel, choose the degree
    if kernel == "poly":
        degree = st.sidebar.slider("Degree (for Poly kernel)", 2, 5, 3)
    else:
        degree = 3  # Default value (not used if kernel != 'poly')
    
    # Gamma parameter for applicable kernels
    if kernel in ["rbf", "poly", "sigmoid"]:
        # Remove 'Auto' option to avoid deprecation issues
        gamma_options = {"Scale": "scale", "Manual": "manual"}
        selected_gamma_option = st.sidebar.selectbox("Gamma Option", options=list(gamma_options.keys()), index=0)
        gamma_option_value = gamma_options[selected_gamma_option]
        if gamma_option_value == "manual":
            gamma = st.sidebar.slider("Gamma value", 0.001, 1.0, 0.1)
        else:
            gamma = gamma_option_value
    else:
        gamma = "scale"  # Not used for Linear kernel

    # Test set size for splitting the dataset
    test_size = st.sidebar.slider("Test Set Size (Fraction)", 0.1, 0.5, 0.2, step=0.05)

    # ================================
    # Sidebar: Feature Selection for Visualisation
    # ================================
    st.sidebar.header("Feature Selection for Visualisation")
    with st.sidebar.expander("Learn about Feature Selection"):
        st.markdown(
            """
            **X-axis Feature & Y-axis Feature:**  
            Choose two different features from the Iris dataset to visualise the data and the SVM decision boundary in 2D.
            Selecting two distinct features allows for a meaningful visualisation of how the model separates the classes.
            """
        )
    iris = datasets.load_iris()
    feature_names = iris.feature_names
    x_feature = st.sidebar.selectbox("X-axis Feature", options=feature_names, index=0)
    y_feature = st.sidebar.selectbox("Y-axis Feature", options=feature_names, index=1)
    if x_feature == y_feature:
        st.sidebar.error("X and Y features must be different.")
        st.stop()  # Halt execution if the selected features are identical

    # ================================
    # Load and Display the Dataset
    # ================================
    X = iris.data
    y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y

    st.subheader("Iris Dataset (first 5 rows)")
    st.write(df.head())
    
    # Explain the class labels and their corresponding species names
    st.info(
        "Note: In the Iris dataset, the target classes correspond to the following species:\n"
        "- **0:** Setosa\n"
        "- **1:** Versicolor\n"
        "- **2:** Virginica"
    )

    # Use only the two selected features (for visualisation and model training)
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
    # Removed probability=True to avoid unnecessary overhead
    if kernel == "poly":
        model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    else:
        model = SVC(kernel=kernel, C=C, gamma=gamma)

    model.fit(X_train, y_train)

    # ================================
    # Evaluate the Model
    # ================================
    st.info(
        "The **Model Performance** section below displays how well the SVM has classified the test data. "
        "It includes the overall accuracy, a detailed classification report, and a confusion matrix. "
        "These metrics help you understand the models strengths and any areas that might require tuning."
    )
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"**Accuracy on test set:** {accuracy:.2f}")

    # Display the classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    st.text("Classification Report:")
    st.write(pd.DataFrame(report).transpose())

    st.info(
        "The **Classification Report** provides detailed metrics for each class:\n\n"
        "- **Precision:** The proportion of correct predictions among those predicted as a given class.\n"
        "- **Recall:** The proportion of actual instances of the class that were correctly predicted.\n"
        "- **F1-Score:** The harmonic mean of precision and recall; a higher value indicates a better balance between the two.\n"
        "- **Support:** The number of actual occurrences of the class in the test set.\n\n"
        "These metrics allow you to assess the performance of the model on each species."
    )

    # Confusion matrix: Visualises correct vs misclassifications
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.info(
        "The **Confusion Matrix** visualises the classifiers performance. Each row represents the actual class, while each column represents the predicted class. "
        "The diagonal elements show the number of correct predictions for each class, whereas off-diagonal elements indicate misclassifications. "
        "Ideally, most of the values should be concentrated along the diagonal."
    )

    # ================================
    # Plot the Decision Boundary
    # ================================
    # Compute the decision boundary over a grid that covers the feature space
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary with training data
    st.info(
        "The **Decision Boundary** plot below shows the regions in the feature space that the model assigns to each class, "
        "overlaid with the training data points."
    )
    st.subheader("Decision Boundary on Training Data")
    fig_train, ax_train = plt.subplots()
    ax_train.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter_train = ax_train.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', s=50, cmap=plt.cm.coolwarm)
    ax_train.set_xlabel(x_feature)
    ax_train.set_ylabel(y_feature)
    ax_train.set_title("Decision Boundary on Training Data")
    legend_train = ax_train.legend(*scatter_train.legend_elements(), title="Training Classes")
    ax_train.add_artist(legend_train)
    st.pyplot(fig_train)
    
    # Plot decision boundary with test data
    st.info(
        "This plot shows the same decision boundary, now overlaid with the test data points to illustrate how well the model generalises."
    )
    st.subheader("Decision Boundary on Test Data")
    fig_test, ax_test = plt.subplots()
    ax_test.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter_test = ax_test.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='^', edgecolor='k', s=70, cmap=plt.cm.coolwarm)
    ax_test.set_xlabel(x_feature)
    ax_test.set_ylabel(y_feature)
    ax_test.set_title("Decision Boundary on Test Data")
    legend_test = ax_test.legend(*scatter_test.legend_elements(), title="Test Classes")
    ax_test.add_artist(legend_test)
    st.pyplot(fig_test)

    # ================================
    # Article Link Section (at the bottom)
    # ================================
    st.markdown("---")
    st.markdown("### Further Reading")
    st.markdown(
        "For a detailed discussion on SVMs, check out my [Medium article](https://blochai.medium.com/the-complete-guide-to-support-vector-machines-from-theory-to-practice-5d1a7de9f5fe?sk=da315377fe68360a72081d437ee2aa5c)."
    )

    # ================================
    # Add Footer
    # ================================
    # Fixed footer across the bottom of the screen
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
