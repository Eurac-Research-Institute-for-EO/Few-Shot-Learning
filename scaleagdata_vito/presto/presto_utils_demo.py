import io
from typing import Literal, Optional, Tuple, Union, cast

import catboost as cb
import numpy as np
import pandas as pd
import requests
import torch
from loguru import logger
from presto.presto import Presto, PrestoFineTuningModel, get_sinusoid_encoding_table
from sklearn.metrics import (
    classification_report,
    explained_variance_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from torch import nn
from torch.utils.data import DataLoader

default_model_kwargs = {
    "encoder_embedding_size": 128,
    "channel_embed_ratio": 0.25,
    "month_embed_ratio": 0.25,
    "encoder_depth": 2,
    "mlp_ratio": 4,
    "encoder_num_heads": 8,
    "decoder_embedding_size": 128,
    "decoder_depth": 2,
    "decoder_num_heads": 8,
}


def load_pretrained_model(
    path_to_weigths=None,
    dekadal=True,
    finetuned=False,
    ss_dekadal=False,
    device="cpu",
    num_outputs=1,
    max_sequence_length=72,
):
    architecture = "dekadal" if dekadal else "monthly"
    if finetuned:
        # initialize architecture without loading pretrained model
        model = Presto.construct(**default_model_kwargs)
        if dekadal:
            # extend model architecture to dekadal
            model = reinitialize_pos_embedding(
                model, max_sequence_length=max_sequence_length
            )
        # if we try to load a PrestoFT model, the architecture will be encoder + head
        # so we run the command to construct the same FT model architecture to be able
        # to correctly load weights
        model = model.construct_finetuning_model(num_outputs=num_outputs)
        logger.info(f"Initialize Presto {architecture} architecture with PrestoFT...")
        best_model = torch.load(path_to_weigths, map_location=device)
        model.load_state_dict(best_model)
    else:
        # load pretrained default Presto
        if path_to_weigths is not None:
            if ss_dekadal:
                logger.info(
                    f"Initialize Presto {architecture} architecture with 10d ss trained WorldCereal Presto weights..."
                )
                # if model was self-supervised trained as dekadal, first reinitialize positional
                # embeddings then load weights
                model = Presto.construct(**default_model_kwargs)
                if dekadal:
                    model = reinitialize_pos_embedding(
                        model, max_sequence_length=max_sequence_length
                    )
                best_model = torch.load(path_to_weigths, map_location=device)
                model.load_state_dict(best_model)
            else:
                logger.info(
                    f"Initialize Presto {architecture} architecture with 30d ss trained WorldCereal Presto weights..."
                )
                # if the model was self-supervised trained as monthly, first load weights then
                # reinitialize positional embeddings if dekadal
                model = Presto.construct(**default_model_kwargs)
                best_model = torch.load(path_to_weigths, map_location=device)
                model.load_state_dict(best_model)
                if dekadal:
                    model = reinitialize_pos_embedding(
                        model, max_sequence_length=max_sequence_length
                    )
        else:
            logger.info(
                f"Initialize Presto {architecture} architecture with pretrained Presto weights..."
            )
            model = Presto.load_pretrained()
            if dekadal:
                model = reinitialize_pos_embedding(
                    model, max_sequence_length=max_sequence_length
                )
    model.to(device)
    return model


def load_pretrained_model_from_url(
    model_url,
    dekadal=True,
    finetuned=False,
    ss_dekadal=False,
    strict=False,
    device="cpu",
    num_outputs=1,
    max_sequence_length=36,
):
    architecture = "dekadal" if dekadal else "monthly"
    if finetuned:
        # initialize architecture without loading pretrained model
        model = Presto.construct(**default_model_kwargs)
        if dekadal:
            # extend model architecture to dekadal
            model = reinitialize_pos_embedding(
                model, max_sequence_length=max_sequence_length
            )
        # if we try to load a PrestoFT model, the architecture will be encoder + head
        # so we run the command to construct the same FT model architecture to be able
        # to correctly load weights
        model = model.construct_finetuning_model(num_outputs=num_outputs)
        logger.info(
            f"Initialize Presto {architecture} architecture with dekadal PrestoFT..."
        )
        response = requests.get(model_url)
        best_model = torch.load(io.BytesIO(response.content), map_location=device)
        model.load_state_dict(best_model, strict=strict)
    else:
        if model_url != "":
            if ss_dekadal:
                logger.info(
                    f"Initialize Presto {architecture} architecture with 10d ss trained WorldCereal Presto weights..."
                )
                model = Presto.construct(**default_model_kwargs)
                # if model was self-supervised trained as dekadal, first reinitialize positional
                # embeddings then load weights
                if dekadal:
                    model = reinitialize_pos_embedding(
                        model, max_sequence_length=max_sequence_length
                    )
                response = requests.get(model_url)
                best_model = torch.load(
                    io.BytesIO(response.content), map_location=device
                )
                model.load_state_dict(best_model, strict=strict)
            else:
                logger.info(
                    f"Initialize Presto {architecture} architecture with 30d ss trained WorldCereal Presto weights..."
                )
                # if the model was self-supervised trained as monthly, first load weights then
                # reinitialize positional embeddings if dekadal
                model = Presto.construct(**default_model_kwargs)
                response = requests.get(model_url)
                best_model = torch.load(
                    io.BytesIO(response.content), map_location=device
                )
                model.load_state_dict(best_model, strict=strict)
                if dekadal:
                    model = reinitialize_pos_embedding(
                        model, max_sequence_length=max_sequence_length
                    )
        else:
            logger.info(
                f"Initialize Presto {architecture} architecture with pretrained Presto weights..."
            )
            model = Presto.load_pretrained()
            if dekadal:
                model = reinitialize_pos_embedding(
                    model, max_sequence_length=max_sequence_length
                )
    model.to(device)
    return model


def reinitialize_pos_embedding(
    model: Union[Presto, PrestoFineTuningModel], max_sequence_length: int
):  # PrestoFineTuningModel
    # reinitialize encoder pos embed to stretch max length of time series
    model.encoder.pos_embed = nn.Parameter(
        torch.zeros(1, max_sequence_length, model.encoder.pos_embed.shape[-1]),
        requires_grad=False,
    )
    pos_embed = get_sinusoid_encoding_table(
        model.encoder.pos_embed.shape[1], model.encoder.pos_embed.shape[-1]
    )
    model.encoder.pos_embed.data.copy_(pos_embed)

    if isinstance(model, Presto):
        # reinitialize decoder pos embed to stretch max length of time series
        model.decoder.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, model.decoder.pos_embed.shape[-1]),
            requires_grad=False,
        )
        pos_embed = get_sinusoid_encoding_table(
            model.decoder.pos_embed.shape[1], model.decoder.pos_embed.shape[-1]
        )
        model.decoder.pos_embed.data.copy_(pos_embed)
    return model


def get_encodings(
    dl: DataLoader,
    pretrained_presto: PrestoFineTuningModel,
    device: Literal["cpu", "cuda"] = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    pretrained_presto.eval()
    batch_encodings, batch_targets = [], []
    for x, y, dw, latlons, month, variable_mask in dl:
        batch_targets.append(y)
        x_f, dw_f, latlons_f, month_f, variable_mask_f = [
            t.to(device) for t in (x, dw, latlons, month, variable_mask)
        ]
        with torch.no_grad():
            cast(Presto, pretrained_presto).eval()
            encodings = (
                cast(Presto, pretrained_presto)
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
            batch_encodings.append(encodings)
    batch_encodings_np = np.concatenate(batch_encodings)
    batch_targets_np = np.concatenate(batch_targets)
    if len(batch_targets_np.shape) == 2 and batch_targets_np.shape[1] == 1:
        batch_targets_np = batch_targets_np.ravel()
    return batch_encodings_np, batch_targets_np


def predict_with_head(
    dl: DataLoader,
    finetuned_model: PrestoFineTuningModel,
    task: Literal["regression", "binary", "multiclass"] = "regression",
    device: Literal["cpu", "cuda"] = "cpu",
):
    test_preds, targets = [], []
    for x, y, dw, latlons, month, variable_mask in dl:
        targets.append(y)
        x_f, dw_f, latlons_f, month_f, variable_mask_f = [
            t.to(device) for t in (x, dw, latlons, month, variable_mask)
        ]
        finetuned_model.eval()
        with torch.no_grad():
            preds = finetuned_model(
                x_f,
                dynamic_world=dw_f.long(),
                mask=variable_mask_f,
                latlons=latlons_f,
                month=month_f,
            ).squeeze(dim=1)
            # binary classification
            if task == "binary":
                preds = torch.sigmoid(preds)
            # multiclass classification
            elif task == "multiclass":
                preds = nn.functional.softmax(preds)
                preds = np.argmax(preds, axis=-1)
            test_preds.append(preds.cpu().numpy())
    test_preds_np = np.concatenate(test_preds)
    target_np = np.concatenate(targets)
    return test_preds_np, target_np


def revert_to_original_units(y_norm, upper_bound, lower_bound):
    return y_norm * (upper_bound - lower_bound) + lower_bound


def normalize_target(y, upper_bound, lower_bound):
    return (y - lower_bound) / (upper_bound - lower_bound)


def evaluate(
    pretrained_model: PrestoFineTuningModel,
    dl_val: DataLoader,
    ds_model: Union[cb.CatBoostRegressor, cb.CatBoostClassifier, None] = None,
    task: Literal["regression", "binary", "multiclass"] = "regression",
    up_val: Optional[Union[int, float]] = None,
    low_val: Optional[Union[int, float]] = None,
):
    """
    Evaluate the performance of a deep learning model on a validation dataset.

    Parameters:
    - pretrained_model: The pretrained model used for generating the encodings.
    - ds_model: The ML model trained for the downstream task to evaluate.
    - dl_val: The validation dataloader.
    - task: The type of task, which can be 'regression', 'binary', or 'multiclass'.

    Returns:
    - metrics: The performance metrics on validation dataset.

    """
    if ds_model is None:
        preds, targets = predict_with_head(dl_val, pretrained_model)
        if task == "binary":
            preds = preds > 0.5
        # TBT on multiclass
        elif task == "multiclass":
            preds = [dl_val.dataset.index_to_class[int(t)] for t in preds]
            targets = [dl_val.dataset.index_to_class[int(t)] for t in targets]
    else:
        encodings, targets = get_encodings(dl_val, pretrained_model)
        preds = ds_model.predict(encodings)
    if up_val is not None and low_val is not None:
        targets = revert_to_original_units(targets, up_val, low_val)
        preds = revert_to_original_units(preds, up_val, low_val)
    if task == "regression":
        metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(targets, preds))),
            "R2_score": float(r2_score(targets, preds)),
            "explained_var_score": float(explained_variance_score(targets, preds)),
            "MAPE": float(mean_absolute_percentage_error(targets, preds)),
        }
    elif task == "binary":
        metrics = classification_report(targets, preds, output_dict=True)
    elif task == "multiclass":
        metrics = classification_report(targets, preds, output_dict=True)
    return metrics, preds, targets


def get_feature_list(num_time_steps=36):
    feature_list = ["DEM-alt-20m", "DEM-slo-20m", "lat", "lon"]
    for i in range(num_time_steps):
        feature_list += [
            f"OPTICAL-B02-ts{i}-10m",
            f"OPTICAL-B03-ts{i}-10m",
            f"OPTICAL-B04-ts{i}-10m",
            f"OPTICAL-B05-ts{i}-20m",
            f"OPTICAL-B06-ts{i}-20m",
            f"OPTICAL-B07-ts{i}-20m",
            f"OPTICAL-B08-ts{i}-10m",
            f"OPTICAL-B8A-ts{i}-20m",
            f"OPTICAL-B11-ts{i}-20m",
            f"OPTICAL-B12-ts{i}-20m",
            f"SAR-VH-ts{i}-20m",
            f"SAR-VV-ts{i}-20m",
            f"METEO-precipitation_flux-ts{i}-100m",
            f"METEO-temperature_mean-ts{i}-100m",
        ]
    return feature_list


def evaluate_catboost(
    ds_model: Union[cb.CatBoostRegressor, cb.CatBoostClassifier],
    val_x: Union[np.ndarray, pd.DataFrame],
    val_y: Union[np.ndarray, pd.Series],
    task: Literal["regression", "binary", "multiclass"],
    up_val: Optional[Union[int, float]] = None,
    low_val: Optional[Union[int, float]] = None,
):
    preds = ds_model.predict(val_x)
    if up_val is not None and low_val is not None:
        val_y = revert_to_original_units(val_y, up_val, low_val)
        preds = revert_to_original_units(preds, up_val, low_val)
    if task == "regression":
        metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(val_y, preds))),
            "R2_score": float(r2_score(val_y, preds)),
            "explained_var_score": float(explained_variance_score(val_y, preds)),
            "MAPE": float(mean_absolute_percentage_error(val_y, preds)),
        }
    elif task == "binary":
        metrics = classification_report(val_y, preds, output_dict=True)
    elif task == "multiclass":
        metrics = classification_report(val_y, preds, output_dict=True)
    return metrics, preds, val_y


def train_catboost_on_encodings(
    dl_train: DataLoader,
    presto_model: PrestoFineTuningModel,
    task: Literal["regression", "binary", "multiclass"],
    cb_device: Literal["GPU", None] = None,
):

    if task == "regression":
        cbm = cb.CatBoostRegressor(
            random_state=3,
            task_type=cb_device,
            logging_level="Silent",
            loss_function="RMSE",
        )
    else:
        cbm = cb.CatBoostClassifier(
            random_state=3,
            task_type=cb_device,
            logging_level="Silent",
        )
    logger.info("Computing Presto encodings")
    encodings_np, targets = get_encodings(dl_train, presto_model, device="cpu")
    logger.info("Fitting Catboost model on Presto encodings")
    train_dataset = cb.Pool(encodings_np, targets)
    cbm.fit(train_dataset)
    return cbm
