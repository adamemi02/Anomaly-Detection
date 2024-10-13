from pyod.utils.data import *
import matplotlib.pyplot as pp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc,balanced_accuracy_score
from sklearn.preprocessing import label_binarize




x_train,x_test,y_train,y_test=generate_data(n_train=400, n_test=100,contamination=0.1)
def plot():
    pp.scatter(x_train[y_train == 0][:, 0], x_train[y_train == 0][:, 1],
            label='Inliers (Train)', alpha=0.5, color='blue')

    pp.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1],
               label='Outliers (Train)', alpha=0.5, color='red')
    pp.show()

def Knn():
    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(x_train, y_train)

    y_pred_test = knn.predict(x_test)

    conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(conf_matrix)

    TN, FP, FN, TP = conf_matrix.ravel()

    TPR=TP/(TP+FN)
    TNR=TN/(TN+FP)

    balanced_accuracy_score=(TPR+TNR)/2


    print(f'Balanced Accuracy: {balanced_accuracy_score:.2f}')


    y_test_bin = label_binarize(y_test, classes=[0, 1])
    y_score = knn.predict_proba(x_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.2f}')
    pp.figure()
    pp.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    pp.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    pp.xlim([0.0, 1.0])
    pp.ylim([0.0, 1.0])
    pp.xlabel('False Positive Rate')
    pp.ylabel('True Positive Rate')
    pp.title('ROC Curve')
    pp.legend(loc='lower right')
    pp.grid()
    pp.show()

def ex3_z():
    x_train, y_train = generate_data(n_train=1000, n_features=1, contamination=0.1,train_only=True)


    mean = np.mean(x_train)
    std_dev = np.std(x_train)
    z_scores = np.abs((x_train - mean) / std_dev)

    threshold = np.quantile(z_scores, 1 - 0.1)

    y_pred_train = (z_scores > threshold).astype(int).ravel()

    conf_matrix = confusion_matrix(y_train, y_pred_train)

    balanced_acc = balanced_accuracy_score(y_train, y_pred_train)

    TN, FP, FN, TP = conf_matrix.ravel()
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"TN (True Negatives): {TN}, FP (False Positives): {FP}, FN (False Negatives): {FN}, TP (True Positives): {TP}")
    print(f"Balanced Accuracy: {balanced_acc:.2f}")



def ex4(n_samples=1000, n_features=5, contamination_rate=0.1):

    np.random.seed(42)

    n_inliers = int(n_samples * (1 - contamination_rate))
    n_outliers = n_samples - n_inliers

    mean_inliers = np.zeros(n_features)
    cov_inliers = np.eye(n_features)  # Identity matrix as covariance (independent dimensions)
    x_inliers = np.random.multivariate_normal(mean_inliers, cov_inliers, n_inliers)

    x_outliers = np.random.uniform(low=-6, high=6, size=(n_outliers, n_features))

    x_train = np.vstack([x_inliers, x_outliers])
    y_train = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])

    z_scores = np.abs((x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0))

    z_scores_sum = np.sum(z_scores, axis=1)

    threshold = np.quantile(z_scores_sum, 1 - contamination_rate)

    y_pred = (z_scores_sum > threshold).astype(int)


    balanced_acc = balanced_accuracy_score(y_train, y_pred)
    return balanced_acc



n_samples = 1000
n_features = 3
contamination_rate = 0.1

balanced_acc=ex4(n_samples,n_features,contamination_rate)

print(f"Balanced Accuracy for {n_features}-dimensional data: {balanced_acc}")


