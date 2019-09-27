import json
import os
from shutil import rmtree
from dataclasses import dataclass, asdict

import spacy
from sklearn.base import BaseEstimator


@dataclass
class TrainParams(object):
    n_iter: int = 30
    early_stopping_iter: int = 2
    n_examples: int = 0
    use_gpu: int = -1  # use -1 for CPU, 0 or 1 for GPU
    noise_level: float = 0.0
    gold_preproc: bool = True
    verbose: bool = False


@dataclass
class TestParams(object):
    # uncomment for gpu usage:
    # gpu_id: int = 1
    gold_preproc: bool = True
    displacy_path: str = None
    displacy_limit: int = 25


class SpacyModel(BaseEstimator):
    """
    Base class for spaCy model wrappers that has sklearn-like interface
    which works nicely for evaluation and hyperparameter tuning.
    """

    def __init__(self, pipeline: str, vectors_path: str, location: str, hyperparams: dict = dict(), lang: str = 'pl'):
        """
        Initializes model that can be fitted or used to make predictions and evaluate itself.
        :param pipeline: pipeline that will be passed to spacy.cli.train
        :param vectors_path: path to vectors that will be used for training the model
        :param location: location where model will be saved, should contain model-best or model-final folder inside
        :param hyperparams: dict of hyperparameters for model training
        :param lang: language for which the model is trained
        """
        self.lang = lang
        self.pipeline = pipeline
        self.vectors_path = vectors_path
        self.location = location
        self.hyperparams = hyperparams

    @property
    def best_model_path(self):
        return os.path.join(self.location, 'model-best')

    @property
    def final_model_path(self):
        return os.path.join(self.location, 'model-final')

    @property
    def model_path(self):
        # models trained using older version of spaCy don't remember best iteration:
        if os.path.exists(self.best_model_path):
            return self.best_model_path
        else:
            return self.final_model_path

    @property
    def meta_path(self):
        return os.path.join(self.model_path, 'meta.json')

    # noinspection PyAttributeOutsideInit
    def fit(self, train_path: str, dev_path: str, train_params: TrainParams = TrainParams(), refit: bool = True):
        """
        Trains the model, saves the one from best epoch.
        If model.location already contains a trained model, it will be erased (same as re-fitting sklearn models)
        unless refit=False is passed.
        :param refit: if True, model will be fitted starting from random weights
        :param train_path: path to training data in spacy format
        :param dev_path: path to evaluation data in spacy format
        :param train_params: parameters for model training, passed to CLI
        :return: self
        """

        if refit is True:
            base_model = None
        else:
            # specify itself as a base model to continue training
            base_model = self.model_path

        # set hyperparameters (in spacy loaded only via environment variables)
        for key in self.hyperparams:
            os.environ[key] = str(self.hyperparams[key])

        spacy.cli.train(
            lang=self.lang,
            pipeline=self.pipeline,
            output_path=self.location,
            train_path=train_path,
            dev_path=dev_path,
            vectors=self.vectors_path,
            base_model=base_model,
            **asdict(train_params)
        )

        # remove all paths except the model path (best or final - depending on spacy version)
        for filename in os.listdir(self.location):
            filepath = os.path.join(self.location, filename)
            if os.path.isdir(filepath) and filepath != self.model_path:
                rmtree(filepath)

        self.meta_ = json.load(open(self.meta_path))
        return self

    # noinspection PyAttributeOutsideInit
    def score(self, data_path: str, test_params: TestParams = TestParams()) -> dict:
        """
        Calls spacy.cli.evaluate on itself and returns available metrics
        :param data_path: path to evaluation data in spacy format
        :param test_params: forwarded to spacy.cli.evaluate
        :return: scores dict
        """
        self.scores_ = spacy.cli.evaluate(
            model=self.model_path,
            data_path=data_path,
            return_scores=True,
            **asdict(test_params)
        )
        return self.scores_

    def get_nlp(self):
        """
        Get the underlying spacy model (eg. to make predictions, tag text, etc.)
        """
        model = spacy.load(self.model_path)
        return model
