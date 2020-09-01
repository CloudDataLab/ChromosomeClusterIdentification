import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc


feature_file_path = '../dataset/chromosome_training_feature_v2.csv'
# feature_file_path = '../dataset/chromosome_geometric_features.csv'

train = pd.read_csv(feature_file_path)
train_df = train.drop('chromosome_num', axis=1)
#train_df = train
train_df['type_of_two'] = train_df['type'].map(lambda x:1 if x>0 else 0)
data2 = train_df.iloc[:, 0:11]
target2 = train_df.iloc[:, 12:13]
data = data2.values
target = target2.values


def kfold_model_evaluation(model):
    scores = []
    aucs = []
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(data,target):
        model.fit(data[train_index], target[train_index])
        score = model.score(data[test_index], target[test_index])
        pre_y = model.predict(data[test_index])
        # fpr, tpr, thresholds = metrics.roc_curve(, scores, pos_label=2)
        roc_auc = roc_auc_score(target[test_index], pre_y)
        aucs.append(roc_auc)
        scores.append(score)
    model_acc = str(round(np.mean(scores) * 100, 2)) + ' +/- ' + str(round(np.std(scores)*100,2))
    model_auc = str(round(np.mean(aucs), 4)) + ' +/- ' + str(round(np.std(aucs), 4))
    return model_acc, model_auc


if __name__ == '__main__':
    lr = LogisticRegression()
    svc = SVC()
    knn = KNeighborsClassifier(n_neighbors=2)
    gaussian = GaussianNB()
    perceptron = Perceptron()
    sgd = SGDClassifier()
    decision_tree = DecisionTreeClassifier()
    random_forest = RandomForestClassifier(n_estimators=100)

    acc_lr, auc_lr = kfold_model_evaluation(lr)
    acc_svc, auc_svc = kfold_model_evaluation(svc)
    acc_knn, auc_knn = kfold_model_evaluation(knn)
    acc_gaussian, auc_gaussian = kfold_model_evaluation(gaussian)
    acc_perceptron, auc_perceptron = kfold_model_evaluation(perceptron)
    acc_sgd, auc_sgd = kfold_model_evaluation(sgd)
    acc_decision_tree, auc_decision_tree = kfold_model_evaluation(decision_tree)
    acc_random_forest, auc_random_forest = kfold_model_evaluation(random_forest)
    acc_scores = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest', 'Naive Bayes', 'Perceptron',
                  'Stochastic Gradient Decent', 'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_lr,
                  acc_random_forest, acc_gaussian, acc_perceptron,
                  acc_sgd, acc_decision_tree],
        'AUC' : [auc_svc, auc_knn, auc_lr,
                  auc_random_forest, auc_gaussian, auc_perceptron,
                  auc_sgd, auc_decision_tree]
    })
    acc_scores = acc_scores.sort_values(by='Score', ascending=False)
    print(acc_scores)

