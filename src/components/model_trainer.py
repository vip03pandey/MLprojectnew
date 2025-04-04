import os 
import sys
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import(
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Initiating model trainer")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor()
            }

            param_grid = {
                'RandomForestRegressor': {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10, 20]
                },
                'GradientBoostingRegressor': {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5]
                },
                'AdaBoostRegressor': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1]
                },
                'KNeighborsRegressor': {
                    'n_neighbors': [3, 5, 7]
                },
                'DecisionTreeRegressor': {
                    'max_depth': [None, 10, 20]
                },
                'LinearRegression': {}  
            }

            best_model = None
            best_score = -1
            best_model_name = ""
            model_report = {}

            for model_name, model in models.items():
                logging.info(f"Training and tuning {model_name}...")
                params = param_grid.get(model_name, {})

                if params:
                    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, scoring='r2', n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    tuned_model = grid_search.best_estimator_
                    score = grid_search.best_score_
                else:
                    model.fit(X_train, y_train)
                    tuned_model = model
                    score = r2_score(y_test, tuned_model.predict(X_test))

                model_report[model_name] = score

                if score > best_score:
                    best_score = score
                    best_model = tuned_model
                    best_model_name = model_name

            if best_score < 0.6:
                raise CustomException("Best model score is less than 0.6", sys)

            logging.info(f"Best model: {best_model_name} with score: {best_score}")

            save_object(
                file_path=self.trainer_config.trained_model_file_path,
                obj=best_model
            )

            final_predictions = best_model.predict(X_test)
            r2 = r2_score(y_test, final_predictions)

            return r2

        except Exception as e:
            raise CustomException(e, sys)
