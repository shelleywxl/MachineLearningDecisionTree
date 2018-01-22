# MachineLearningDecisionTree
COEN 240 - Machine Learning Project

A group project for COEN 240 Machine Learning course.

Implement a decision tree from scratch. The decision tree can deal with both discrete values and continuous values.
ID3 algorithm (using Entropy / Information Gain to decide which attribute to be put in the root of a subtree).
[Extension] two methods to discretize continuous attribute values:
  1) Binary cut:
    Sort according to continuous attribute;
    Consider adjacent values with different labels;
    Midpoint as a candidate threshold
    Find one threshold (cut point) that maximum the information gain.
  2) Multi-interval discretization:
    Same as binary cut. Find the "best" threshold;
    Recursively cut until it does not meet MDLPC (Minimum Description Length Principle Criterion).
    
Example datasets (both from UCI Datasets):
  1) Contraceptive Method Choice Data Set (1473 instances, 9 attributes (6 discrete, 3 continuous));
  2) Heart Data Set (270 instances, 13 attributes (7 discrete, 6 continuous)).
  
  
Cutting method resources:
Fayyad, Usama M., Irani, Keki B. (1992): On the handling of continuous-valued attributes in decision tree generation.
Fayyad, Usama M., Irani, Keki B. (1993): Multi-interval discretization of continuous-valued attributes for classificatio learning.
