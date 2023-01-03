from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

class ML:

    def __init__(self):
        self.models = {
            'ols': LinearRegression(),
            'lasso': Lasso(),
            'pls': PLSRegression(),
            'cart': DecisionTreeRegressor(),
            'knn': KNeighborsRegressor(),
            'rfr': RandomForestRegressor(),
            'gbr': GradientBoostingRegressor(),
            'svr': SVR(cache_size=1000),
            'mlp': MLPRegressor(activation='relu', solver='adam', alpha=0.01, batch_size=64, max_iter=10000, early_stopping=True),
            }

    def train(self, X, y, method):
        print(method.upper())
        param_grids = self.param_dict_maker(X)
        pipe = Pipeline(
        [('scaler', StandardScaler()),
        ('mdl', self.models[method])]
        )
        reg = GridSearchCV(
            estimator=pipe,
            param_grid=param_grids[method],
            cv=10,
            scoring='neg_root_mean_squared_error', 
            n_jobs=-1
            )
        reg.fit(X, y)
        print(reg.best_params_)
        print('RMSE_CV = {:.3f}\n'.format(-reg.best_score_))
        self.model = reg.best_estimator_
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)

    def param_dict_maker(self, X):
        self.param_grids = {
            'ols': {
                'mdl__fit_intercept': [True, False]
            },
            'lasso': {
                'mdl__alpha': [0.001, 0.01, 0.1],
                },
            'pls': {
                'mdl__n_components': range(1, X.shape[1]),
                },
            'cart': {
                'mdl__max_depth': range(1, 10),
                },
            'knn': {
                'mdl__n_neighbors': range(1, 10),
                },
            'rfr': {
                # 'mdl__max_features': ['sqrt', 'log2', None, 1.0],
                'mdl__n_estimators': [10, 50, 100,],
                },
            'gbr': {
                'mdl__max_features': ['sqrt', 'log2', None, 1.0],
                'mdl__n_estimators': [100, 1000],
                'mdl__learning_rate': [0.0001, 0.001, 0.01, 0.1],
                },
            'svr': {
                'mdl__kernel':['linear', 'poly', 'rbf'],
                'mdl__C': [0.1, 1, 10],
                'mdl__epsilon': [0.01, 0.1, 1, 100],
                'mdl__gamma': [0.001, 0.01, 0.1, 1],
                },
            'mlp': {
                'mdl__hidden_layer_sizes':[(16, 16),],
                'mdl__learning_rate_init': [0.001,],
                },
        }
        return self.param_grids
