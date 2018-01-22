from sklearn.model_selection import train_test_split
from Main import accuracy


# ---------- Function: prune_tree ----------
# Prune a tree according to validation score
# Args:
#   dt: decision tree.
#   root: root of the decision tree.
#   data: training dataset
#   label: training label
# Returns:
#   Procedure with no returns.

def prune_tree(dt, root, data, label):
    # if node is a leaf
    if root.parent_ != None and root.label_ != None:
        # get original score
        old_score = 0
        for i in range(0, 2, 1):
            old_score += score(dt, data, label)
        # let root's parent be the leaf
        parent = root.parent_
        # find root's children to do majority vote
        if len(parent.continuous_children_) != 0:
            children = parent.continuous_children_
        else:
            children = parent.discrete_children_.values()
        new_label = majority_vote(children)
        parent.label_ = new_label
        # if pruning is not better, change it back
        new_score = 0
        for i in range(0, 2, 1):
            new_score += score(dt, data, label)
        if new_score < old_score:
            parent.label_ = None

    # if root is not a leaf search leaves in its children to do pruning
    else:
        for node in root.continuous_children_:
            prune_tree(dt, node, data, label)
        for node in root.discrete_children_.values():
            prune_tree(dt, node, data, label)


# ---------- Function: score ----------
# Return predicting score based on validation data
# Args:
#   dt: decision tree.
#   data: training dataset
#   label: training label
# Returns:
#   testing_accuracy: Prediction score based on validation data.

def score(dt, data, label):
    # get validation data from training data (20%)
    training_data, testing_data, training_label, testing_label = \
        train_test_split(data, label, test_size=0.2)
    # test the tree to get prediction accuracy on validation data
    predicted_testing_label_list = []
    # after split function, only use testing data as validation data
    dt.predict(testing_data, predicted_testing_label_list)
    testing_accuracy = accuracy(predicted_testing_label_list, testing_label)
    return testing_accuracy


# ---------- Function: majority_vote ----------
# Take a majority vote for a node by all its children_leaf_node
# Args:
#   children: all node of a subtree rooted on one parent node which is under pruning in function prune_tree.
# Returns:
#   majority: a number equals the max occurrence of labels in the subtree

def majority_vote(children):
    label_and_num = {}
    # record all labels in this subtree and their counts
    build_dictionary(children, label_and_num)
    majority = None
    majority_num = 0
    # find the max counts of the label
    for label in label_and_num.keys():
        if label_and_num[label] > majority_num:
            majority_num = label_and_num[label]
            majority = label
    return majority


# ---------- Function: majority_vote ----------
# find labels and their occurrence of all leaf_nodes for one parent
# Args:
#   children: all node of a subtree rooted on one parent node which is under pruning in function prune_tree.
#   label_and_num: a dictionary with key = labels and value = occurrence of each node
# Returns:
#  Procedure with no returns

def build_dictionary(children, label_and_num):
    for node in children:
        # if the node is a leaf node, record its label
        if node.label_ != None:
            label = node.label_
            label_and_num[label] = label_and_num.get(label, 0) + 1
        # if the node is not a leaf node, record label of leaves of the subtree rooted in this node
        else:
            if len(node.continuous_children_) != 0:
                grand_children = node.continuous_children_
            else:
                grand_children = node.discrete_children_.values()
            build_dictionary(grand_children, label_and_num)
