
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0, root_split_feature=0, root_split_threshold=3.6)
cross_val_score(regressor, boston.data, boston.target, cv=10)