# bagging-tree
Decision Tree, Bagging Trees and Random Forests Implementation from scratch in Python.

Execute the following scripts to see the results:

```python preprocess-assg4.py``` to preprocess and split the data into training and test set.

```python trees.py trainingSet.csv testSet.csv 1``` to run decision tree model.

```python trees.py trainingSet.csv testSet.csv 2``` to run bagging tree model.

```python trees.py trainingSet.csv testSet.csv 1``` to run random forest model with sampled `sqrt(p)` attributes.

```python cv_depth.py``` to check the performance of models with varying depth limit of the trees.

```python cv_frac.py``` to plot the learning curves of the three models.

```python cv_numtrees.py``` to compare the performance of models with varying number of trees in the ensemble.

```python neural_net.py trainingSet.csv testSet.csv``` to check the performance of dataset with neural networks.

Results for the above models and scripts are available in `hw4.pdf`
