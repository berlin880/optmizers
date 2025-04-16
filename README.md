![Training GIF](https://doimages.nyc3.cdn.digitaloceanspaces.com/010AI-ML/content/images/2018/05/68747470733a2f2f707669676965722e6769746875622e696f2f6d656469612f696d672f70617274312f6772616469656e745f64657363656e742e676966.gif)


# 🧠 Logistic Regression from Scratch with GD, SGD, and Mini-Batch

A custom logistic regression classifier built using NumPy and trained using different optimization strategies:  
**Gradient Descent (GD)**, **Stochastic Gradient Descent (SGD)**, and **Mini-Batch Gradient Descent**.

This project demonstrates core machine learning concepts including:
- Manual implementation of logistic regression
- L2 regularization
- Early stopping
- Learning rate decay
- Visualization of training loss
- Performance evaluation using accuracy, precision, recall, and F1-score

---

## 📌 Features

✅ Logistic Regression for binary classification  
✅ Supports 3 optimizers: `gd`, `sgd`, and `mini-batch`  
✅ L2 regularization to reduce overfitting  
✅ Learning rate decay over time  
✅ Early stopping to avoid unnecessary training  
✅ Evaluation metrics and confusion matrix  
✅ Loss curve visualization for each optimizer  

---

## 🧪 Demo

```bash
python main.py
##🔧 Requirements
Install dependencies:

bash
Copy
Edit
pip install numpy matplotlib scikit-learn


LogisticRegression(
    learning_rate=0.001,
    epochs=1000,
    optimizer='gd',        # 'gd', 'sgd', or 'mini-batch'
    batch_size=64,
    regularization=0.2,
    early_stopping=20,
    lr_decay=0.005,
    random_state=42,
    verbose=True
)



🚀 How It Works
Synthetic Dataset is created using make_classification()

Data is split into training and testing sets

Logistic regression is trained using each optimizer

Loss is recorded and plotted

Model is evaluated on the test set using classification metrics


##📈 Metrics
Accuracy

Precision

Recall

F1 Score

Confusion Matrix

##🧠 Why Build From Scratch?
Because it's the best way to learn! You get to:

Understand each step of training

See how optimizers behave differently

Learn how regularization and early stopping improve performance

