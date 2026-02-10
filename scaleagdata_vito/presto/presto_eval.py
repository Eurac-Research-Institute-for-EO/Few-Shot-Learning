import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from presto.presto import (
    FinetuningHead,
    Presto,
    PrestoFineTuningModel,
    param_groups_lrd,
)
from presto.utils import DEFAULT_SEED, device
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    explained_variance_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from scaleagdata_vito.presto.datasets import ScaleAGDataset
from scaleagdata_vito.presto.presto_utils import reinitialize_pos_embedding

logger = logging.getLogger("__main__")

SklearnStyleModel = Union[BaseEstimator, CatBoostClassifier]


@dataclass
class Hyperparams:
    lr: float = 2e-5
    max_epochs: int = 100
    batch_size: int = 256
    patience: int = 20
    num_workers: int = 8
    catboost_iterations: int = 8000


class ScaleAGEval:
    threshold = 0.5

    def __init__(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        target_name: str,
        dekadal: bool = True,  # can be inferred from timesteps or the other way around?
        task: Literal["regression", "binary", "multiclass"] = "regression",
        seed: int = DEFAULT_SEED,
        min_samples_per_class: int = 3,
        name: Optional[str] = "",
    ):
        self.seed = seed
        self.num_timesteps = 36 if dekadal else 12
        # SAR cannot equal 0.0 since we take the log of it
        self.train_df = self.prep_dataframe(train_data, dekadal=dekadal)
        self.val_df = self.prep_dataframe(val_data, dekadal=dekadal)
        self.test_df = self.val_df.copy()
        self.target_name = target_name
        self.dekadal = dekadal
        self.task = task
        self.name = name
        self.min_samples_per_class = min_samples_per_class
        self.num_outputs = (
            len(self.train_df[target_name].unique()) if task == "multiclass" else 1
        )

        if task == "multiclass":
            # compress all classes in train that contain less
            # than MIN_SAMPLES_PER_CLASS samples into "other"
            class_counts = self.train_df[
                target_name
            ].value_counts()  ###### why distinction ft classes and ds classes?
            small_classes = class_counts[class_counts < min_samples_per_class].index
            # if no classes with n_samples < classes_threshold are present in train,
            # force the "other" class using the class with minimal number of samples
            # this is done so that the other class is always present,
            # thus making test set with new labels compatible with the model,
            # as in this way unseen labels will be mapped into "other" class
            if len(small_classes) == 0:
                small_classes = [class_counts.index[-1]]
            self.train_df.loc[
                self.train_df[target_name].isin(small_classes), target_name
            ] = "other"
            self.val_df.loc[
                self.val_df[target_name].isin(small_classes), target_name
            ] = "other"
            self.test_df.loc[
                self.test_df[target_name].isin(small_classes), target_name
            ] = "other"
            # create one-hot representation from obtained labels
            # one-hot is needed for finetuning,
            # while downstream CatBoost can work with categorical labels
            self.class_list = self.train_df[target_name].unique().tolist()
            self.train_df = self.convert_to_onehot(
                self.train_df
            )  # no columns will be added since the classes are retrieved from this df
            self.val_df = self.convert_to_onehot(self.val_df)
            self.test_df = self.convert_to_onehot(self.test_df)
        self.ds_class = ScaleAGDataset

    def convert_to_onehot(
        self,
        df: pd.DataFrame,
    ):
        df[f"{self.target_name}_onehot"] = df[self.target_name].copy()
        df = pd.get_dummies(
            df, prefix="", prefix_sep="", columns=[f"{self.target_name}_onehot"]
        )
        cols_to_add = [xx for xx in self.class_list if xx not in df.columns]
        if len(cols_to_add) > 0:
            for col in cols_to_add:
                df[col] = 0
        return df

    @staticmethod
    def prep_dataframe(
        df: pd.DataFrame,
        dekadal: bool = False,
    ):
        # SAR cannot equal 0.0 since we take the log of it
        cols = [
            f"SAR-{s}-ts{t}-20m"
            for s in ["VV", "VH"]
            for t in range(36 if dekadal else 12)
        ]
        df = df.drop_duplicates(subset=["lat", "lon", "end_date"])
        df = df[~pd.isna(df).any(axis=1)]
        df = df[~(df.loc[:, cols] == 0.0).any(axis=1)]
        return df

    def _construct_finetuning_model(
        self, pretrained_model: Union[Presto, PrestoFineTuningModel]
    ) -> PrestoFineTuningModel:
        if isinstance(pretrained_model, PrestoFineTuningModel):
            # if we are passing a model that had already been fine tuned then we need
            # then we need to adjust a bit the initialization as the construct_finetuning_model
            # expects a Presto architecture but here we already have a PrestoFineTuningModel
            model = cast(Callable, self.construct_from_finetuned)(
                encoder=pretrained_model.encoder
            )
        else:
            model = cast(Callable, pretrained_model.construct_finetuning_model)(
                num_outputs=self.num_outputs
            )

        # reinitialize positional embedding if dekadal
        if self.dekadal:
            model = reinitialize_pos_embedding(
                model, max_sequence_length=self.num_timesteps * 2
            )
        model.to(device)
        return model

    def construct_from_finetuned(
        self,
        encoder,
    ) -> PrestoFineTuningModel:
        head = FinetuningHead(
            num_outputs=self.num_outputs,
            hidden_size=encoder.embedding_size,
        )
        model = PrestoFineTuningModel(encoder, head).to(encoder.pos_embed.device)
        model.train()
        return model

    def setup_model_dict(self, models, hyperparams=Hyperparams()) -> Dict:
        model_dict = {}
        for model in models:
            if model == "Logistic Regression":
                model_dict[model] = LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=self.seed
                )
            elif model == "Random Forest Classifier":
                model_dict[model] = RandomForestClassifier(
                    class_weight="balanced", random_state=self.seed
                )
            elif model == "CatBoostClassifier":
                model_dict[model] = CatBoostClassifier(
                    iterations=hyperparams.catboost_iterations,
                    depth=8,
                    learning_rate=0.05 if self.task == "binary" else 0.1,
                    early_stopping_rounds=50,
                    l2_leaf_reg=3 if self.task == "binary" else 30,
                    random_state=self.seed,
                    eval_metric="F1" if self.task == "binary" else "MultiClass",
                    loss_function="Logloss" if self.task == "binary" else "MultiClass",
                )
            elif model == "Linear Regression":
                model_dict[model] = LinearRegression()

            elif model == "Random Forest Regressor":
                model_dict[model] = RandomForestRegressor(random_state=self.seed)

            elif model == "CatBoostRegressor":
                model_dict[model] = CatBoostRegressor(random_state=self.seed)
        return model_dict

    @torch.no_grad()
    def finetune_sklearn_model(
        self,
        dl: DataLoader,
        val_dl: DataLoader,
        pretrained_model: PrestoFineTuningModel,
        models: List[str] = ["Logistic Regression", "Random Forest Classifier"],
        hyperparams: Hyperparams = Hyperparams(),
    ) -> Sequence[BaseEstimator]:

        for model_mode in models:
            assert model_mode in [
                "Linear Regression",
                "Logistic Regression",
                "Random Forest Classifier",
                "CatBoostClassifier",
                "Random Forest Regressor",
                "CatBoostRegressor",
            ]
        pretrained_model.eval()

        # move outside of the function ???
        def dataloader_to_encodings_and_targets(
            dl: DataLoader,
        ) -> Tuple[np.ndarray, np.ndarray]:
            encoding_list, target_list = [], []
            for x, y, dw, latlons, month, variable_mask in dl:
                x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                    t.to(device) for t in (x, dw, latlons, month, variable_mask)
                ]
                target_list.append(y)

                with torch.no_grad():
                    encodings = (
                        pretrained_model.encoder(
                            x_f,
                            dynamic_world=dw_f.long(),
                            mask=variable_mask_f,
                            latlons=latlons_f,
                            month=month_f,
                        )
                        .cpu()
                        .numpy()
                    )
                    encoding_list.append(encodings)
            encodings_np = np.concatenate(encoding_list)
            targets = np.concatenate(target_list)
            if len(targets.shape) == 2 and targets.shape[1] == 1:
                targets = targets.ravel()
            return encodings_np, targets

        train_encodings, train_targets = dataloader_to_encodings_and_targets(dl)
        val_encodings, val_targets = dataloader_to_encodings_and_targets(val_dl)

        fit_models = []
        model_dict = self.setup_model_dict(models, hyperparams)
        for model in tqdm(models, desc="Fitting sklearn models"):
            if model == "CatBoostClassifier":
                fit_models.append(
                    clone(model_dict[model]).fit(
                        train_encodings,
                        train_targets,
                        eval_set=Pool(val_encodings, val_targets),
                    )
                )
            else:
                fit_models.append(
                    clone(model_dict[model]).fit(train_encodings, train_targets)
                )
        return fit_models

    # @staticmethod
    def _inference_for_dl(
        self,
        dl,
        finetuned_model: Union[PrestoFineTuningModel, SklearnStyleModel],
        pretrained_model: Optional[PrestoFineTuningModel] = None,
    ) -> Tuple:
        test_preds, test_probs, targets = [], [], []

        for x, y, dw, latlons, month, variable_mask in dl:
            targets.append(y)
            x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                t.to(device) for t in (x, dw, latlons, month, variable_mask)
            ]
            if isinstance(finetuned_model, (Presto, PrestoFineTuningModel)):
                finetuned_model.eval()
                preds = finetuned_model(
                    x_f,
                    dynamic_world=dw_f.long(),
                    mask=variable_mask_f,
                    latlons=latlons_f,
                    month=month_f,
                ).squeeze(dim=1)
                # binary classification
                if self.task == "binary":
                    preds = torch.sigmoid(preds).cpu().numpy()
                    probs = preds.copy()
                # multiclass classification
                elif self.task == "multiclass":
                    preds = nn.functional.softmax(preds, dim=1).cpu().numpy()
                    probs = preds.copy()
                    preds = preds.argmax(axis=-1)
                else:
                    preds = preds.cpu().numpy()

            else:
                cast(Presto, pretrained_model).eval()
                encodings = (
                    cast(Presto, pretrained_model)
                    .encoder(
                        x_f,
                        dynamic_world=dw_f.long(),
                        mask=variable_mask_f,
                        latlons=latlons_f,
                        month=month_f,
                    )
                    .cpu()
                    .numpy()
                )
                if self.task == "binary":
                    preds = finetuned_model.predict_proba(encodings)[:, 1]
                else:
                    preds = finetuned_model.predict(encodings)

            test_preds.append(preds)
            test_probs.append(probs)

        test_preds_np = np.concatenate(test_preds)
        test_probs_np = np.concatenate(test_probs)
        target_np = np.concatenate(targets)

        return test_preds_np, test_probs_np, target_np

    def compute_metrics(self, target_np, test_preds_np, prefix, classes=[]):
        if self.task == "regression":
            return {
                f"{prefix}_RMSE": float(
                    np.sqrt(mean_squared_error(target_np, test_preds_np))
                ),
                f"{prefix}_R2_score": float(r2_score(target_np, test_preds_np)),
                f"{prefix}_explained_var_score": float(
                    explained_variance_score(target_np, test_preds_np)
                ),
                f"{prefix}_MAPE": float(
                    mean_absolute_percentage_error(target_np, test_preds_np)
                ),
            }
        else:
            results = classification_report(
                target_np, test_preds_np, labels=classes, output_dict=True
            )
            _results_df = pd.DataFrame(results).T.reset_index()
            _results_df.columns = pd.Index(
                ["class", "precision", "recall", "f1-score", "support"]
            )
            corrected_macro_f1 = _results_df.loc[
                (_results_df["support"] >= self.min_samples_per_class)
                & (
                    ~_results_df["class"].isin(
                        ["accuracy", "macro avg", "weighted avg"]
                    )
                ),
                "f1-score",
            ].mean()
            _results_df.loc[_results_df["class"] == "macro avg", "f1-score"] = (
                corrected_macro_f1
            )
            _results_df.loc[_results_df["class"] == "macro avg", "f1-score"] = (
                corrected_macro_f1 if not np.isnan(corrected_macro_f1) else 0
            )

            # revert to dict
            results = _results_df.to_dict()
            results = {f"{prefix}_{k}": v for k, v in results.items()}
        return results

    @torch.no_grad()
    def evaluate(
        self,
        finetuned_model: Union[PrestoFineTuningModel, BaseEstimator],
        pretrained_model: Optional[PrestoFineTuningModel] = None,
    ) -> Dict:
        test_ds = self.ds_class(
            self.test_df, self.target_name, self.task, self.num_timesteps
        )

        dl = DataLoader(
            test_ds,
            batch_size=2048,
            shuffle=False,  # keep as False!
            num_workers=Hyperparams.num_workers,
        )
        assert isinstance(dl.sampler, torch.utils.data.SequentialSampler)

        test_preds_np, test_probs_np, target_np = self._inference_for_dl(
            dl, finetuned_model, pretrained_model
        )
        prefix = f"{self.name}_{finetuned_model.__class__.__name__}"
        if self.task == "binary":
            test_preds_np = test_preds_np >= self.threshold
            results = self.compute_metrics(
                target_np, test_preds_np, prefix, classes=[0, 1]
            )
        elif self.task == "multiclass":
            # map indices to classes
            test_preds_np = [test_ds.index_to_class[int(t)] for t in test_preds_np]
            target_np = [test_ds.index_to_class[int(t)] for t in target_np]
            test_classes = list(test_ds.index_to_class.values())
            results = self.compute_metrics(
                target_np, test_preds_np, prefix, classes=test_classes
            )
        # targets are normalized during training to avoid gradient explosion.
        # revert to original units
        elif self.task == "regression":
            target_np = test_ds.revert_to_original_units(target_np)
            test_preds_np = test_ds.revert_to_original_units(test_preds_np)
            results = self.compute_metrics(target_np, test_preds_np, prefix)
        return results

    def finetune(
        self, pretrained_model, hyperparams: Hyperparams = Hyperparams()
    ) -> PrestoFineTuningModel:

        model = self._construct_finetuning_model(pretrained_model)

        parameters = param_groups_lrd(model)
        optimizer = AdamW(parameters, lr=hyperparams.lr)

        train_ds = self.ds_class(
            self.train_df, self.target_name, self.task, self.num_timesteps
        )
        val_ds = self.ds_class(
            self.val_df, self.target_name, self.task, self.num_timesteps
        )

        if self.task == "regression":
            loss_fn = nn.MSELoss()
        elif self.task == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_dl = DataLoader(
            train_ds,
            batch_size=hyperparams.batch_size,
            shuffle=True,
            num_workers=hyperparams.num_workers,
            generator=generator,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=hyperparams.batch_size,
            shuffle=False,
            num_workers=hyperparams.num_workers,
        )

        train_loss = []
        val_loss = []
        best_loss = None
        best_model_dict = None
        epochs_since_improvement = 0

        run = None
        try:
            import wandb

            run = wandb.run
        except ImportError:
            pass

        for _ in (pbar := tqdm(range(hyperparams.max_epochs), desc="Finetuning")):
            model.train()
            epoch_train_loss = 0.0
            for x, y, dw, latlons, month, variable_mask in tqdm(
                train_dl, desc="Training", leave=False
            ):
                x, y, dw, latlons, month, variable_mask = [
                    t.to(device) for t in (x, y, dw, latlons, month, variable_mask)
                ]
                input_dict = dict(
                    x=x,
                    dynamic_world=dw.long(),
                    mask=variable_mask,
                    latlons=latlons,
                    month=month,
                )

                optimizer.zero_grad()
                preds = model(**input_dict)
                if self.task == "multiclass":
                    loss = loss_fn(
                        preds.squeeze(-1), y.long()
                    )  # if ce y must be the index of the class
                else:
                    loss = loss_fn(preds.squeeze(-1), y.float())
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss.append(epoch_train_loss / len(train_dl))

            model.eval()
            all_preds, all_y = [], []
            for x, y, dw, latlons, month, variable_mask in val_dl:
                x, y, dw, latlons, month, variable_mask = [
                    t.to(device) for t in (x, y, dw, latlons, month, variable_mask)
                ]
                input_dict = dict(
                    x=x,
                    dynamic_world=dw.long(),
                    mask=variable_mask,
                    latlons=latlons,
                    month=month,
                )
                with torch.no_grad():
                    preds = model(**input_dict)

                    if self.task == "multiclass":
                        all_y.append(y.long())
                    else:
                        all_y.append(y.float())
                    all_preds.append(preds.squeeze(-1))

            val_loss.append(loss_fn(torch.cat(all_preds), torch.cat(all_y)))
            # val_loss.append(nn.functional.huber_loss(torch.cat(all_preds), torch.cat(all_y)))
            pbar.set_description(
                f"Train metric: {train_loss[-1]}, Val metric: {val_loss[-1]}"
            )

            if run is not None:
                wandb.log(
                    {
                        f"{self.name}_finetuning_val_loss": val_loss[-1],
                        f"{self.name}_finetuning_train_loss": train_loss[-1],
                    }
                )

            if best_loss is None:
                best_loss = val_loss[-1]
                best_model_dict = deepcopy(model.state_dict())
            else:
                if val_loss[-1] < best_loss:
                    best_loss = val_loss[-1]
                    best_model_dict = deepcopy(model.state_dict())
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= hyperparams.patience:
                        logger.info("Early stopping!")
                        break
        assert best_model_dict is not None
        model.load_state_dict(best_model_dict)

        model.eval()
        return model

    def finetuning_results_sklearn(
        self,
        sklearn_model_modes: List[str],
        finetuned_model: PrestoFineTuningModel,
        hyperparams: Hyperparams = Hyperparams(),
    ) -> Tuple[Dict, List[BaseEstimator]]:

        results_dict = {}
        if len(sklearn_model_modes) > 0:

            train_ds = self.ds_class(
                self.train_df, self.target_name, self.task, self.num_timesteps
            )
            val_ds = self.ds_class(
                self.val_df, self.target_name, self.task, self.num_timesteps
            )

            dl = DataLoader(
                train_ds,
                batch_size=2048,
                shuffle=False,
                num_workers=4,
            )

            val_dl = DataLoader(
                val_ds,
                batch_size=2048,
                shuffle=False,
                num_workers=4,
            )

            sklearn_models = self.finetune_sklearn_model(
                dl,
                val_dl,
                finetuned_model,
                models=sklearn_model_modes,
                hyperparams=hyperparams,
            )
            for sklearn_model in sklearn_models:
                logger.info(f"Evaluating {sklearn_model}...")
                results_dict.update(self.evaluate(sklearn_model, finetuned_model))
        return results_dict, sklearn_models

    def finetuning_results(
        self,
        pretrained_model,
        sklearn_model_modes: List[str],
    ) -> Tuple[Dict, PrestoFineTuningModel, List[BaseEstimator]]:
        for model_mode in sklearn_model_modes:
            assert model_mode in [
                "Linear Regression",
                "Logistic Regression",
                "Random Forest Classifier",
                "CatBoostClassifier",
                "Random Forest Regressor",
                "CatBoostRegressor",
            ]
        results_dict = {}
        # we want to always finetune the model, since the sklearn models
        # will use the finetuned model as a base

        finetuned_model = self.finetune(pretrained_model)
        logger.info("Finetuning done")
        results_dict.update(self.evaluate(finetuned_model, None))
        logger.info("Finetuning head evaluation done")
        results_df_sklearn, sklearn_models_trained = self.finetuning_results_sklearn(
            sklearn_model_modes, finetuned_model
        )
        results_dict.update(results_df_sklearn)
        return results_dict, finetuned_model, sklearn_models_trained
