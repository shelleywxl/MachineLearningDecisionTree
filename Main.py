from __future__ import print_function
from __future__ import division

import pandas as pd
from sklearn.model_selection import train_test_split

import DecisionTree
import Pruning

# ---------- Function: accuracy ----------
# Calculate prediction accuracy.
# Args:
#   predicted_label: predicted labels of type list.
#   original_label: original labels of type list.
# Returns:
#   a ratio number (number of correctly predicted labels / overal number of labels).


def accuracy(predicted_label, original_label):
    n = len(original_label)
    correct_count = 0
    # compare two label lists:
    for idx in range(n):
        if predicted_label[idx] == original_label.iloc[idx]:
            correct_count += 1
    return correct_count / n


# ---------- Function: confusion_matrix ----------
# Calculate confusion matrix (true/false positive/negative)
# Args:
#   predicted_label: predicted label list.
#   original_label: correct labels in the dataset, of type pandas.DataFrame.
#   true_value: the value to represent "true" result in the dataset
#   neg_value: the value to represent "false" result in the dataset
# Returns:
#   print out the ratios of true/negative postive/negative

def confusion_matrix(predicted_label, original_label, true_value):
    true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
    n = len(original_label)
    for idx in range(n):
        if predicted_label[idx] == original_label.iloc[idx]:
            if predicted_label[idx] == true_value:
                true_positive += 1
            else:
                true_negative += 1
    
        else:
            if predicted_label[idx] == true_value:
                false_positive += 1
            else:
                false_negative += 1
    print("True positive: ", float(true_positive) / float(true_positive + false_negative), "(", true_positive, ")")
    print("False positive: ", float(false_positive) / float(false_positive + true_negative), "(", false_positive, ")")
    print("False negative: ", float(false_negative) / float(false_negative + true_positive), "(", false_negative, ")")
    print("True negative: ", float(true_negative) / float(true_negative + false_positive), "(", true_negative, ")")


# ---------- Main ----------

if __name__ == "__main__":
    
    # Dataset 1
    # 1.import data
    print("\nDataset: Statlog (Heart)")
    df = pd.read_csv("heart.dat", sep="\s+", header=None)
    discrete_attr = set([1, 2, 5, 6, 8, 10, 12])

    # 2.split data into training and testing sets; set discrete attribute columns
    label = df.iloc[:, -1]
    data = df.drop(df.columns[[-1]], axis=1)  # drop the last column

    training_data, testing_data, training_label, testing_label = \
        train_test_split(data, label, test_size=0.2)  # 20% data as testing sets

    # 3.training
    # allow_multicut: using multicut or not
    for allow_multicut in [False, True]:
        print("\n******************************************")
        print("allow multi cut: ", allow_multicut)

        # dt: type of DecisionTree
        dt = DecisionTree.DecisionTree(discrete_attr, allow_multicut=allow_multicut)
        dt.fit(training_data, training_label)

        # 4.testing, print training accuracy and testing accuracy
        predicted_training_label_list = []
        dt.predict(training_data, predicted_training_label_list)
        print("Training accuracy before pruning: ", accuracy(predicted_training_label_list, training_label))

        predicted_testing_label_list = []
        dt.predict(testing_data, predicted_testing_label_list)
        testing_accuracy = accuracy(predicted_testing_label_list, testing_label)
        print("Testing accuracy before pruning: ", testing_accuracy)

        # confusion matrix
        print("\nConfusion Matrix")
        print("Training result before pruning:")
        confusion_matrix(predicted_training_label_list, training_label, 2, 1)  # 1: absence; 2: presence
        print("Testing result before pruning:")
        confusion_matrix(predicted_testing_label_list, testing_label, 2, 1)

        # pruning the tree and print the new result
        Pruning.prune_tree(dt, dt.tree_root_, training_data, training_label)
        # get training and testing result
        predicted_training_label_list = []
        dt.predict(training_data, predicted_training_label_list)
        print("=================Pruning...=================")
        print("Training accuracy after pruning: ", accuracy(predicted_training_label_list, training_label))
        predicted_testing_label_list = []
        dt.predict(testing_data, predicted_testing_label_list)
        print("Testing accuracy after pruning:", accuracy(predicted_testing_label_list, testing_label))

        # confusion matrix
        print("\nConfusion Matrix")
        print("Training result with pruning:")
        confusion_matrix(predicted_training_label_list, training_label, 2, 1)  # 1: absence; 2: presence
        print("Testing result with pruning:")
        confusion_matrix(predicted_testing_label_list, testing_label, 2, 1)

    # Dataset 2
    print("\n************************************")
    print("\n************************************")
    print("\nDataset: Contraceptive Method Choice")
    # 1.import data
    df = pd.read_csv("cmc.data", sep=",", header=None)
    discrete_attr = set([1,2,4,5,6,7])

    # 2. split data into training and testing sets; set discrete attribute columns
    label = df.iloc[:, -1]
    data = df.drop(df.columns[[-1]], axis=1)  # drop the last column

    training_data, testing_data, training_label, testing_label = \
        train_test_split(data, label, test_size=0.2)  # 20% data as testing sets

    # 3. training
    # allow_multicut: using multicut or not
    for allow_multicut in [False, True]:
        print("\n******************************************")
        print("allow multi cut: ", allow_multicut)

        # dt: type of DecisionTree
        dt = DecisionTree.DecisionTree(discrete_attr, allow_multicut=allow_multicut)
        dt.fit(training_data, training_label)

        # 4. testing, print training accuracy and testing accuracy
        predicted_training_label_list = []
        dt.predict(training_data, predicted_training_label_list)
        print("Training accuracy before pruning: ", accuracy(predicted_training_label_list, training_label))

        predicted_testing_label_list = []
        dt.predict(testing_data, predicted_testing_label_list)
        # print ("predicted test", predicted_testing_label_list)
        testing_accuracy = accuracy(predicted_testing_label_list, testing_label)
        print("Testing accuracy before pruning: ", testing_accuracy)

        # confusion matrix
        print("\nConfusion Matrix")
        print("Training result before pruning:")
        confusion_matrix(predicted_training_label_list, training_label, 2, 1)  # 1: absence; 2: presence
        print("Testing result before pruning:")
        confusion_matrix(predicted_testing_label_list, testing_label, 2, 1)

        # pruning the tree and print the new result
        Pruning.prune_tree(dt, dt.tree_root_, training_data, training_label)
        # get training and testing result
        predicted_training_label_list = []
        dt.predict(training_data, predicted_training_label_list)
        print("=================Pruning...=================")
        print("Training accuracy after pruning: ", accuracy(predicted_training_label_list, training_label))
        predicted_testing_label_list = []
        dt.predict(testing_data, predicted_testing_label_list)
        print("Testing accuracy after pruning:", accuracy(predicted_testing_label_list, testing_label))

        # confusion matrix
        print("\nConfusion Matrix")
        print("Training result with pruning:")
        confusion_matrix(predicted_training_label_list, training_label, 2, 1)  # 1: absence; 2: presence
        print("Testing result with pruning:")
        confusion_matrix(predicted_testing_label_list, testing_label, 2, 1)
