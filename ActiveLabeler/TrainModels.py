from pathlib import Path
import sys
import pytorch_lightning as pl
import yaml

#Internal imports
sys.path.insert(0,"./Self-Supervised-Learner")
sys.path.insert(1,"./ActiveLabelerModels")
from model import SIMCLR, SIMSIAM
from .Linear_models import SSLEvaluator
from .ClassifierModel import ClassifierModel


class TrainModels():
    def __init__(self, config_path, model_path, unlabled_dataset_path, log_name):
        def load_config(config_path):
            with open(config_path) as file:
                config = yaml.safe_load(file)
            return config

        self.model_path = model_path
        self.parameters = load_config(config_path)
        self.log_name = log_name
        self.log_count = 0

        if self.parameters['encoder']['encoder_type'] == "SIMCLR":
            self.encoder = SIMCLR.SIMCLR.load_from_checkpoint(self.parameters['encoder']['encoder_path'], unlabled_dataset_path)
        elif self.parameters['encoder']['encoder_type'] == "SIMSIAM":
            self.encoder = SIMSIAM.SIMSIAM.load_from_checkpoint(self.parameters['encoder']['encoder_path'], unlabled_dataset_path)

        if self.parameters['classifier']['classifier_type'] == "SSLEvaluator":
            self.linear_model = SSLEvaluator(
                    n_input=self.parameters['encoder']['e_embedding_size'],
                    n_classes=self.parameters['classifier']['c_num_classes'],
                    p=self.parameters['classifier']['c_dropout'],
                    n_hidden=self.parameters['classifier']['c_hidden_dim']
                    )
        else:
            raise NameError("Not Implemented")
        self.model = ClassifierModel(self.parameters, self.encoder, self.linear_model)

    def train_all(self, training_dataset, validation_dataset):
        self.model.unfreeze_encoder()
        trainer = pl.Trainer(gpus=1, max_epochs=self.parameters['training']['epochs'])
        trainer.fit(self.model, training_dataset, validation_dataset)
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(f"{self.model_path}{self.log_name}_{self.log_count}")
        self.log_count += 1
    
    def train_linear(self, training_dataset, validation_dataset):
        self.model.freeze_encoder()
        trainer = pl.Trainer(gpus=1, max_epochs=self.parameters['training']['epochs'])
        trainer.fit(self.model, training_dataset, validation_dataset)
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(f"{self.model_path}{self.log_name}_{self.log_count}")
        self.log_count += 1
    
    def get_model(self):
        return self.model
