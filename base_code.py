# Librairie pandas qui permet de charger et manipuler facilement la donnée
import pandas as pd
# Numpy est une librairie proposant un ensemble d'outils mathématiques optimisé pour les performances sur les CPU modernes
import numpy as np
# pytorch est la librairie de deep learning que nous allons utiliser pour implémenter les réseaux de neurones
import torch
# pytorch-lightning est une surcouche à pytorch permettant de simplifier l'entraînement des modèles
import pytorch_lightning as pl
# sklearn regroupe une ensemble de fonctions essentielles en machine learning
import sklearn
import sklearn.model_selection

# Small abstraction layer for pytorch
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels=None, **kwargs):
        self.data = inputs
        self.has_labels = False
        if isinstance(labels, np.ndarray):
            self.has_labels = True
            self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, ids):
        if self.has_labels:
            return torch.tensor(inputs[ids], dtype=torch.float32), torch.tensor(labels[ids], dtype=torch.float32)
        return torch.tensor(inputs[ids], dtype=torch.float32)

def batch_loader(dataset, batch_size=64, num_workers=4):
    sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(dataset), batch_size=batch_size, drop_last=False)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, num_workers=num_workers)
    return loader

def create_data_loaders(inputs, labels, batch_size=64, val_split=0.2):
    # First shuffle data
    p = np.random.permutation(len(inputs))
    inputs, labels = inputs[p], labels[p]
    split_point = int(len(inputs)*(1-val_split))
    train_loader = batch_loader(NumpyDataset(inputs[:split_point], labels[:split_point]), batch_size=batch_size)
    val_loader = batch_loader(NumpyDataset(inputs[:split_point], labels[split_point:]), batch_size=batch_size)
    return train_loader, val_loader


class MyNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # TODO 2: ajouter la définition des différentes couches du réseau
        # ex: self.layer1 = torch.nn.Linear(...)

        # TODO 3: Ajouter la définition de la fonction de coût (loss)
        self.loss = ...

    def forward(self, x):
        # TODO 2: Appeler les différentes couches du réseau ici
        # ex: yh = self.layer1(x)
        return yh


    def training_step(self, batch, batch_idx):
        # On commence par séprarer les entrées 'x' des labels 'y'
        x, y = batch
        # Ensuite on demande au modèle de prédire la valeur de y en fonction de x
        yh = self(x)

        loss = self.loss(yh, y)

        self.log('train_loss', loss)
        return loss

    # La validation permet de vérifier périodiquement les performances du modèle
    # sur des données qui ne servent pas à mettre à jour les paramètres
    def validation_step(self, batch, batch_idx):
        x, y = batch
        yh = self(x)
        # Dans un premier temps, la même fonction de coût servira à l'entraînement et la validation
        val_loss = self.loss(yh, y)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)


def load_data():
    # TODO 1
    dataframe = pd.read_csv('dataset.csv')
    dataframe_label = dataframe['label']
    dataframe_values = dataframe.drop(columns=['label'])
    inputs = dataframe_values.values
    labels = dataframe_label.values
    print(inputs)
    return inputs, labels

def preprocess_data(data):
    # TODO 5
    return data

def create_model():
    return MyNet()

def train_model(model, inputs, labels):
    train_loader, validation_loader = create_data_loaders(inputs, labels, batch_size=128)
    # TODO 3
    return model

def measure_model_perf(true_labels, pred_labels):
    # TODO 4
     score = 0
     return score

def cross_validation(inputs, labels, init_model):
    scores = []
    xval = sklearn.model_selection.KFold(n_splits=10, shuffle=True)
    for train_index, test_index in xval.split(inputs):
        # Très important de prendre une copie propre du modèle à chaque tour de boucle
        model = copy.deepcopy(init_model)
        model = train_model(model, inputs[train_index], labels[train_index])

        # Fait passer le modèle en mode "inférence"
        model.eval()
        model.freeze()
        # Il faut transformer les vecteurs numpy en tensor que pytorch peut utiliser
        testset = torch.tensor(inputs[test_index], dtype=torch.float32)
        # On prédit les labels sur les données inconnues
        pred_labels = model(testset).numpy()

        # On mesure les perf du modèle
        scores.append(measure_model_perf(labels[test_index], pred_labels))
    return scores

class Percep(torch.nn.Module):

    def __init__(self,p):

        super(MyPS,self).__init__()

        self.layer1 = torch.nn.Linear(4,1)

        self.layer2 = torch.nn.Linear(1,4)

        self.ft1 = torch.nn.Sigmoid()


inputs, labels = load_data()
inputs = preprocess_data(inputs)
model = create_model()

scores = cross_validation(inputs, labels, model)
avg = np.average(scores)
print("Average score: %s (+- %s)"%(avg, max(avg - np.min(scores), np.max(scores) - avg)))
