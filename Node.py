# Node of a decision tree.
class Node:
    
    # Args:
    #   attr_name: attribute name of type string.
    #   threshold: threshold of type float.
    #   left, right: reference to left child and right child.
    def __init__(self, attr_name="", cut_points=[], continuous_children=[], discrete_children={}, parent=None, label=None):
        self.attribute_name_ = attr_name
        # A list of cut points in increasing order
        self.cut_points_ = cut_points
        # List of children
        # For cut_points with length N, there will be (N+1) children.
        # children_[i] represents the interval [cut_points[i-1], cut_points[i]
        # where cut_points[-1] = -Inf and cut_points[N] = +Inf
        self.continuous_children_ = continuous_children
        self.discrete_children_ = discrete_children
        self.parent_ = parent
        # Only leaf node will have label.
        self.label_ = label
