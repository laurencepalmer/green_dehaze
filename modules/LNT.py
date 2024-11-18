import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
import numpy as np
import xgboost as xgb


class LNT:
    def __init__(self, num_tree, depth, args, CLASSIFICATION=False):
        self.no_tree = num_tree
        self.depth = depth
        self.CLASSIFICATION = CLASSIFICATION
        self.args = args
 
    def fit(self, X, y):
        if self.CLASSIFICATION:
            y = self.split_label(y)

        params = {
            'objective': 'binary:logistic' if self.CLASSIFICATION else 'reg:squarederror',
            'max_depth': self.depth,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1,
            'colsample_bylevel': 1,
            'colsample_bynode': 1,
            'n_estimators': self.no_tree,
            'learning_rate': self.args['lnt_lr'],
            'tree_method': "hist", # gpu_hist for gpu enabled device
            'gpu_id': self.args['gpu_id']
        }

        model = xgb.XGBClassifier(**params) if self.CLASSIFICATION else xgb.XGBRegressor(**params)
        eval_set = [(X, y)]
        
        model.fit(X, y, eval_set=eval_set, verbose=True)

        tree_paths = self.get_path(model)
        # print(tree_paths)

        num_features = X.shape[1]
        A = np.zeros((num_features, self.no_tree))

        for i in range(len(tree_paths)):
            selected = tree_paths[i]
            selected = [int(idx) for idx in selected]
            X_sel = X[:, selected]

            lin_reg = LinearRegression()
            lin_reg.fit(X_sel, y)

            theta = np.zeros(num_features)
            theta[selected] = lin_reg.coef_

            A[:, i] = theta

        self.A = A

    def transform(self, X):
        return X @ self.A

    # split labels into 2 categories
    def split_label(self, y):
        def helper(labels, group1, group2):
            if len(labels) == 0:
                if len(group1) == 0 or len(group2) == 0:
                    pass
                elif all(group1 != prev_group2 for prev_group2 in record):
                    record.append(group2)
                    res[f'{str(group1)}_{str(group2)}'] = np.vectorize(lambda x: 1 if x in group1 else 0)(y)

            else:
                label = labels[0]
                helper(labels[1:], group1 + [label], group2)
                helper(labels[1:], group1, group2 + [label])

        labels = np.unique(y)
        record = []
        res = pd.DataFrame()

        helper(labels, [], [])
        res = res.to_numpy().astype(np.float64)
        res = np.squeeze(res)
        # print(res.shape)
        return res

    def get_path(self, model):
        trees_df = model.get_booster().trees_to_dataframe()
        tree_paths = []

        def helper(df, node_id, path):
            node = df[df['Node'] == node_id].iloc[0]
            if node['Feature'] == 'Leaf':
                if path:
                    tree_paths.append(tuple(sorted(path)))
                return
            path.add(int(node['Feature'][1:]))
            left_node = int(node['Yes'].split('-')[-1])
            right_node = int(node['No'].split('-')[-1])
            helper(df, left_node, path.copy())
            helper(df, right_node, path.copy())

        for tree_index in range(self.no_tree):
            tree_df = trees_df[trees_df['Tree'] == tree_index].reset_index(drop=True)
            if tree_df.empty:
                self.no_tree = tree_index
                break
            root_node = int(tree_df.loc[0, 'Node'])
            helper(tree_df, root_node, set())

        unique_tree_paths = set(tree_paths)
        tree_paths = [np.array(path) for path in unique_tree_paths]
        self.no_tree = len(tree_paths)
        return tree_paths
        
