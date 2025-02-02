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
    kernel = st.sidebar.selectbox("Select Kernel", options=["linear", "poly", "rbf", "sigmoid"], index=2)
    
    # Regularization parameter C
    C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, 1.0)
    
    # For the polynomial kernel, choose the degree
    if kernel == "poly":
        degree = st.sidebar.slider("Degree (for polynomial kernel)", 2, 5, 3)
    else:
        degree = 3  # default value (not used if kernel != 'poly')
    
    # Gamma parameter for kernels that use it
    if kernel in ["rbf", "poly", "sigmoid"]:
        gamma_option = st.sidebar.selectbox("Gamma Option", options=["scale", "auto", "manual"], index=0)
        if gamma_option == "manual":
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
    # For the polynomial kernel, include the degree parameter; otherwise, ignore it.
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
    # Create a meshgrid for plotting the decision boundary
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    # Predict over the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contour and training points
    fig, ax = plt.subplots()
    contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', s=50, cmap=plt.cm.coolwarm)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title("Decision Boundary on Training Data")
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    st.pyplot(fig)

    # ================================
    # Expandable Article Section
    # ================================
    with st.expander("Read the full article on SVMs"):
        article_content = """
**In the vast landscape of machine learning algorithms, Support Vector Machines (SVMs) stand out as a powerful and elegant solution for classification problems.**

*Originally developed in the 1990s, SVMs have proven their worth across numerous fields, from medical diagnosis to text classification. But what makes them so special, and when should you consider using them?*

### Understanding SVMs Through Everyday Examples
Imagine you're organising a massive library with thousands of books. Your task is to separate fiction from non-fiction, but it's not always a clear-cut decision. Some books, like historical fiction or creative non-fiction, blur the lines between categories. This is exactly the kind of challenge that SVMs excel at handling.

An SVM works by finding the best possible dividing line (or plane in higher dimensions), the "hyperplane" between different categories. But it doesn't just find any line—it finds the line that creates the widest possible "corridor" between the categories. Think of it like creating the widest possible aisle in your library, making it less likely for books to be misplaced or miscategorised. This is the magic behind its accuracy.

### Real-World Applications: Where SVMs Shine
SVMs have found their way into countless real-world applications. In medical imaging, they help identify potential tumours in X-rays and MRIs. Email systems use them to filter out spam messages. Bioinformaticians employ them to classify genes and predict protein functions. Financial institutions leverage SVMs for forecasting and risk assessment.

What makes SVMs particularly valuable in these contexts is their precision. Unlike some other machine learning algorithms that might make probabilistic guesses, because of that "corridor," SVMs draw relatively clear, definitive boundaries between categories. This makes them especially useful in situations where certainty is crucial.

### The Power and Limitations of SVMs: Understanding When and Why to Use Them
Every tool has its niche, and SVMs are no exception. To understand their capabilities and limitations, we need to explore both their computational characteristics and practical applications.

**Where SVMs Excel**  
SVMs shine in scenarios with clear category separation and moderate-sized datasets. Their ability to handle high-dimensional data (aka data with lots of columns or features) makes them particularly effective for text classification and image recognition tasks. This strength comes from their mathematical foundation, the ability to work in high-dimensional spaces without explicitly computing the coordinates using what we call the "kernel trick."

**Understanding the Limitations**  
However, SVMs come with notable limitations that practitioners need to understand. The most significant challenge emerges with kernel-based SVMs and large datasets. Here's where the mathematics becomes crucial: kernel methods require calculating similarities between every pair of data points, leading to quadratic growth in both computational needs and memory requirements.

**The Magic Behind the Mathematics**  
While the mathematics behind SVMs can appear daunting, the core concept is beautifully simple. It's all about finding the optimal boundary between categories while maximising the margin, that "corridor" we talked about earlier. The algorithm pays special attention to the points closest to this boundary, called support vectors, which give the method its name.

### The Power of the Kernel Trick
Before we dive into different types of kernels, let's understand what we mean by a 'kernel' and why mathematicians call this approach 'the kernel trick'. In computing, a kernel typically refers to the core part of something, like the kernel of an operating system that manages all the basic operations. Similarly, in mathematics, a kernel function is at the core of how SVMs handle complex data.

*[The article continues with detailed explanations on Linear, Polynomial, RBF, and Sigmoid kernels, as well as tuning SVM hyperparameters, their strengths, limitations, and real-world applications.]*        
        """
        st.markdown(article_content)

if __name__ == '__main__':
    main()

