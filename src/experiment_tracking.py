import pandas as pd
import mlflow 
from typing import Protocol
from dataclasses import dataclass, field 
from src.exception_handler import MLFlowError


class ExperimentTracker(Protocol):

    def __start__(self):
        ...

    def log(self):
        ...

    def find_best_model(self):
        ...

@dataclass
class ModelSelection:


    model_selection_dataframe: pd.DataFrame = field(default_factory = lambda: pd.DataFrame())

@dataclass 
class MLFlowTracker:

    experiment_name: str
    tracking_uri: str = "file:/./artifacts"

       def __start__(self) -> None:
        """
        Dunder method that sets tracking URI and experiment name
        to MLFlow engine.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
    
    def log(self) -> None:
        """
        Initialize auto-logging for tracking. This will log model
        artifacts, parameters and metrics in the ./artifacts directory.
        """
        self.__start__()
        mlflow.keras.autolog()

    def find_best_model(self, metric: str) -> ModelSelection(pd.DataFrame):
        """
        Method for model selection. Provides functionalities to find and sort
        the best model based on the given metric in descending order from all
        models within the given experiment directory which makes it easier to
        select best performing model.

        Note: This can also be done with mlflow using `mlflow ui` command. But,
        this is a code implementation of the same.

        Parameters
        ----------
        metric: str
            Metric name to sort the models.

        Returns
        -------
        instanceof: ModelSelection(pd.DataFrame)
            Resulting dataframe.

        Raises
        ------
        MLFlowError: Exception
            If the experiment id or experiment name is none/invalid.
        """
        
        experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
        experiment_id = experiment['experiment_id']

        if experiment is None or experiment_id is None:
            raise MLFlowError(
                f"Invalid experiment details. Please re-check them and try again !!!")

        result_df = mlflow.search_runs([experiment_id], 
                                        order_by=[f"metrics.{metric} DESC"])
        return ModelSelection(model_selection_dataframe = result_df[
                                        ["experiment_id", "run_id", f"metrics.{metric}"]
                                        ])