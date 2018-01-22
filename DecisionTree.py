from __future__ import division

import math

import pandas as pd
import Node as nd


# ---------- Function: entropy ----------
# Calculate the entropy.
# Args:
#   target: labels of type list.
# Returns:
#   entropy of type float number.

def entropy(target):
    label_and_num = {}   # dict, key: label, value: number of existence
    for label in target:
        label_and_num[label] = label_and_num.get(label, 0) + 1
    return_val = 0.0
    for (_, val) in label_and_num.items():
        p = val / len(target)
        return_val += -p * math.log(p, 2)
    return return_val


# ---------- Function: find_majority ----------
# Return the majority label of an attribute.
# Args:
#   target: target attributes of type list.
# Returns:
#   the majority label of the target attribute.

def find_majority(target):
    label_and_num = {}
    for label in target:
        label_and_num[label] = label_and_num.get(label, 0) + 1
    majority = None
    majority_num = 0
    for label in label_and_num.keys():
        if label_and_num[label] > majority_num:
            majority_num = label_and_num[label]
            majority = label
    return majority


# ---------- Function: unique_num ----------
# Args:
#   lis: a list.
# Returns:
#   number of unique elements in the list.

def unique_num(lis):
    num_set = set()
    for num in lis:
        num_set.add(num)
    return len(num_set)


# ---------- Function: pass_MDLPC ----------
# For multicut: Minimum Description Length Principle Criterion.
# If passes, keep cutting.

def pass_MDLPC(s, s1, s2, gain):
    n = len(s)
    k = unique_num(s)
    k1 = unique_num(s1)
    k2 = unique_num(s2)
    delta_gain = math.log((math.pow(3, k)-2), 2) - (k*entropy(s)
                                                    - k1*entropy(s1) - k2*entropy(s2))
    return gain > (math.log(n-1, 2)/n + delta_gain/n)


# ---------- Function: find_cuts_recursively ----------
# Args:
#   depth
#   attr_val_list
#   target_list
#   all_cuts: list of cuts found.
#   allow_multiple_cuts: whether multiple cuts are allowed.
# Returns:
#   None

def find_cuts_recursively(depth, attr_val_list, target_list, all_cuts, allow_multicut):
    best_gain = float('-inf')
    best_cut = None
    best_cut_index = None
    for i in range(0, len(attr_val_list)-1):
        if target_list[i] != target_list[i+1] and attr_val_list[i] != attr_val_list[i+1]:
            # One potential threshold found
            current_cut = (attr_val_list[i] + attr_val_list[i+1]) / 2
            current_gain = compute_multi_interval_gain(
                    attr_val_list,
                    target_list,
                    [current_cut])
            if current_gain > best_gain:
                best_gain = current_gain
                best_cut = current_cut
                best_cut_index = i
    if best_cut is not None:
        all_cuts.append(best_cut)
        # Always accepts the top-level cuts. Apply MDLPC criterion to decide
        # whether to accept other cuts.
        if not allow_multicut:
            return
        if (depth > 0 and (not pass_MDLPC(
            target_list,
            target_list[0:best_cut_index+1],
            target_list[best_cut_index+1:],
                best_gain))):
                return
                
        find_cuts_recursively(depth+1, attr_val_list[:best_cut_index+1],
                              target_list[:best_cut_index+1], all_cuts, allow_multicut)
        find_cuts_recursively(depth+1, attr_val_list[best_cut_index+1:],
                              target_list[best_cut_index+1:], all_cuts, allow_multicut)


# ---------- Function: compute_multi_interval_gain ----------
# Args:
#   attr_val_list: of type list.
#   target_list: of type list.
#   all_cuts: contains unsorted cut points of type list.
# Returns:
#   multi-interval gain of type float number.

def compute_multi_interval_gain(attr_val_list, target_list, all_cuts):
    sample_size = len(attr_val_list)
    val_to_return = entropy(target_list)
    interval_start = 0
    interval_end = 0
    for curr_cut in all_cuts:
        while attr_val_list[interval_end] < curr_cut and interval_end < sample_size:
            interval_end += 1
        val_to_return -= (interval_end - interval_start)/sample_size *\
                         entropy(target_list[interval_start:interval_end])
        interval_start = interval_end
    # deal with the last interval
    val_to_return -= (sample_size - interval_start) / \
                     sample_size*entropy(target_list[interval_start:])
    return val_to_return


# ---------- Function: find_pindex_based_on_cuts ----------
# For building tree and prediction on continuous attribute values.
# Args:
#   attr_val: attribute value (continuous).
#   cuts: cut points of type list.
# Returns:
#   index of a child node in the .continuous_children_ list.

def find_pindex_based_on_cuts(attr_val, cuts):
    cut_idx = 0
    while cut_idx < len(cuts) and attr_val > cuts[cut_idx]:
        cut_idx += 1
    return cut_idx


# ---------- Function: get_tree_size ----------
# Args:
#   node: a tree node of type nd.Node.
# Returns:
#

def get_tree_size(node):
    if node is None:
        return 0
    # count the current node
    return_val = 1
    for child in node.continuous_children_:
        return_val += get_tree_size(child)
    return return_val

# ========== Class: DecisionTree ==========


class DecisionTree:

    # ---------- initialize ----------
    # Parameters:
    #   discrete_attr: discrete attribute names of type list.
    #   max_depth: stop building tree when reaching the maximum depth.
    #   allow_multicut: whether to use binary cut or multicut of type boolean.
    
    def __init__(self, discrete_attr, max_depth=100, allow_multicut=False):
        self.tree_root_ = None
        self.discrete_attr_ = discrete_attr
        self.max_depth_ = max_depth
        self.allow_multicut_ = allow_multicut

    # ---------- Function: fit ----------
    # Training model.
    # Args:
    #   training_data: training data of type pandas.DataFrame.
    #   target: predicting target of type pandas.Series.
    # Returns:
    #   None.
    
    def fit(self, training_data, target):
        self.tree_root_ = self.build_tree(0, training_data, target, None)

    # ---------- Function: predict ----------
    # Make prediction with a decision tree.
    # Args:
    #   testing_data: testing data of type pandas.DataFrame.
    #   predicted_label_list: predicted labels of type List.
    # Returns:
    #   None, predicted_label_list is changed.

    def predict(self, testing_data, predicted_label_list):
        for idx in range(len(testing_data)):
            current_node = self.tree_root_
            while current_node.label_ is None:  # not a leaf
                testing_data_attr_val = testing_data.iloc[idx][current_node.attribute_name_]
                
                # discrete case:
                if current_node.attribute_name_ in self.discrete_attr_:
                    if testing_data_attr_val in current_node.discrete_children_:
                        current_node = current_node.discrete_children_[testing_data_attr_val]
                    else:   # the attribute value does not exist in the training data
                        break
                # continuous case:
                else:
                    interval_idx = find_pindex_based_on_cuts(testing_data_attr_val, current_node.cut_points_)
                    current_node = current_node.continuous_children_[interval_idx]
        
            # now reach a leaf node
            predicted_label_list.append(current_node.label_)

    # ---------- Function: tree_size ----------

    def tree_size(self):
        return get_tree_size(self.tree_root_)

    # ---------- Function: build_tree ----------
    # Recursively build a decision tree.
    # Args:
    #   depth: current depth.
    #   training_data: training data of type pandas.DataFrame.
    #   target: training target of pandas.Series.
    # Returns:
    #   The reference to the node being built of type nd.Node or None.

    def build_tree(self, depth, training_data, target, parent):

        # Build leaf node
        if target.unique().size == 1:
            return nd.Node(label=target[0], parent=parent)

        # Stop building the tree when reaching maximum depth.
        if depth > self.max_depth_:
            return nd.Node(label=find_majority(target), parent=parent)

        # Find the attribute (best_attr) with the highest information gain (best_gain).
        best_attr = None
        best_cuts = None   # for continuous case, the best list of cut points
        best_gain = float('-inf')
        # Iterate over all attr.
        for attr in training_data:
            attr_col = training_data[attr]     # attribute's column of type pandas.Series
            all_cuts = None

            # Discrete case:
            if attr in self.discrete_attr_:
                # Calculate information gain for the attr
                val_to_label = {}   # key: attr's values; value: all corresponding labels of type list
                for idx in range(len(training_data)):
                    if attr_col.iloc[idx] in val_to_label:
                        val_to_label[attr_col.iloc[idx]].append(target.iloc[idx])
                    else:
                        val_to_label[attr_col.iloc[idx]] = [target.iloc[idx]]
                subset_entropy = 0.0
                # Boundary case:
                if len(val_to_label.keys()) == 1:
                    continue

                for key in val_to_label.keys():
                    subset_entropy += len(val_to_label[key]) / len(training_data) * entropy(val_to_label[key])
                current_gain = entropy(target.tolist()) - subset_entropy
        
            # Continuous case:
            else:
                # Build the list of (attr_val, target) pair.
                list_of_pairs = [[attr_col.iloc[idx], target.iloc[idx]] for idx in range(len(training_data))]
                # Sort by attr_val.
                list_of_pairs.sort(key=lambda tup: tup[0])
                # turn [[1,5], [2,6]] to [1,2] and [5,6]
                # Note that attr_val_list has already been sorted.
                attr_val_list, target_list = zip(*list_of_pairs)
                
                all_cuts = []
                find_cuts_recursively(0, attr_val_list, target_list, all_cuts, self.allow_multicut_)
                # Boundary case:
                if len(all_cuts) == 0:
                    continue
                
                all_cuts.sort()
                current_gain = compute_multi_interval_gain(attr_val_list, target_list, all_cuts)

            if current_gain > best_gain:
                best_gain = current_gain
                best_attr = attr
                best_cuts = all_cuts

        # Boundary case: no splitting is possible.
        if best_attr is None:
            # build a leaf node with label set as the majority target.
            return nd.Node(label=find_majority(target), parent=parent)

        # Discrete case:
        if best_attr in self.discrete_attr_:
            # print "Best attribute is discrete"
           
            # Parameters for building subtree.
            attr_val_to_data = {}    # key: attribute val, value: training data of type pandas.DataFrame
            attr_val_to_target = {}   # key: attribute val, value: target of type list (later be converted to Series)
            for row_idx in range(len(training_data)):
                curr_val = training_data.iloc[row_idx][best_attr]
                if curr_val in attr_val_to_data:
                    attr_val_to_data[curr_val] = attr_val_to_data[curr_val].append(training_data.iloc[row_idx])
                    attr_val_to_target[curr_val].append(target.iloc[row_idx])
                else:
                    attr_val_to_data[curr_val] = pd.DataFrame()
                    attr_val_to_data[curr_val] = attr_val_to_data[curr_val].append(training_data.iloc[row_idx])
                    attr_val_to_target[curr_val] = [target.iloc[row_idx]]
        
            new_node = nd.Node(attr_name=best_attr, parent=parent)    # as the parent node!

            children = {}   # key: attribute value, value: child node
            for val in attr_val_to_data.keys():
                child_data = attr_val_to_data[val]   # of type pandas.DataFrame
                child_target = pd.Series(attr_val_to_target[val])   # convert list to pandas.Series
                children[val] = self.build_tree(depth+1, child_data, child_target, new_node)

            new_node.discrete_children_ = children
            return new_node

        # Continuous case: partition the training_data and target using best_attr, best_threshold.
        else:
            # print "Best attribute is continuous"
            num_intervals = len(best_cuts) + 1  # number of intervals = number of cut points + 1
            training_data_partitions = [pd.DataFrame() for _ in range(num_intervals)]
            # will later be converted to list of pandas.Series
            target_list_partitions = [[] for _ in range(num_intervals)]
            for row_idx in range(len(training_data)):
                row_attr_val = training_data.iloc[row_idx][best_attr]
                partition_idx = find_pindex_based_on_cuts(row_attr_val, best_cuts)
                training_data_partitions[partition_idx] = \
                    training_data_partitions[partition_idx].append(training_data.iloc[row_idx])
                target_list_partitions[partition_idx].append(target.iloc[row_idx])

            new_node = nd.Node(attr_name=best_attr, cut_points=best_cuts, parent=parent)
            # Convert list of list to list of pandas.Series.
            target_partitions = [pd.Series(lis) for lis in target_list_partitions]
            children = [self.build_tree(depth + 1, training_data_partitions[idx], target_partitions[idx], new_node)
                        for idx in range(num_intervals)]

            new_node.continuous_children_ = children
            return new_node
