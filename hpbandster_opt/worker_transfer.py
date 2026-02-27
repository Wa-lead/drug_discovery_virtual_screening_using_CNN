import numpy as np
import time
import pandas as pd

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from rdkit import Chem
from rdkit.Chem import AllChem


import ConfigSpace as CS
from hpbandster.core.worker import Worker

from chemception.model_transfer import Chemception
from chemception.featurizer import ChemCeptionizer

class Chemception_wroker(Worker):

    def __init__(self, sleep_interval=0, **kwargs):
        super().__init__(**kwargs)
        self.train_df = pd.read_csv('../data/train_aid686978_0.7.csv')
        self.val_df = pd.read_csv('../data/val_aid686978_0.7.csv')
        self.sleep_interval = sleep_interval
        print('Featurizing the data (Chemceptionizing)')
        self.X_train, self.y_train = self._format_data(self.train_df)
        self.X_val, self.y_val = self._format_data(self.val_df)
        time.sleep(3)
        print('Done featurizing the data (Chemceptionizing)')

    
    def _format_data(self, df):
        featurizer = ChemCeptionizer(embed=20)
        df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
        df["molimage"] = df["mol"].apply(featurizer.featurize)
        df.dropna(subset=["molimage"], inplace=True)
        X = df["molimage"].to_numpy()
        X = np.stack(X, axis=0)
        y = df["label"]
        
        indeces_where_nan = np.any(np.isnan(X), axis=(1, 2, 3))
        X = X[~indeces_where_nan]
        y = y[~indeces_where_nan]
        return X, y        
        
        
    def compute(self, config, budget, **kwargs):
        
        # Duplicate to insert it in info
        config_clean = config.copy()
        learning_rate = config_clean.pop('lr')
        # embed = config_clean.pop('embed')
        
        
        
        
        # print('Featurizing the data (Chemceptionizing)')
        # # X_train, y_train = self._format_data(self.train_df, embed)
        # # X_val, y_val = self._format_data(self.val_df, embed)
        # X_train, y_train = self._format_data(self.train_df, 'train')
        # X_val, y_val = self._format_data(self.val_df, 'val')
        
        
        print('Building the model ..')
        # config['img_size'] = X_train.shape[1]
        # config_clean['img_size'] = X_train.shape[1]
        
        model = Chemception(config=config_clean)
        model = model.build()
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        
        batch_size = 256
        steps_per_epoch = len(self.train_df) // batch_size
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=1e-6, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        
        
        generator = ImageDataGenerator(data_format='channels_last')
        g = generator.flow(self.X_train, self.y_train, batch_size=batch_size)
        
        print('Fitting the model ..')
        
        
        history = model.fit(g,
                            steps_per_epoch=steps_per_epoch,
                            epochs=int(budget),
                            validation_data=(self.X_val, self.y_val),
                            callbacks=[reduce_lr, early_stopping]
                            )
            
        
        loss = model.evaluate(self.X_val, self.y_val, verbose=0)[0]
        time.sleep(self.sleep_interval)

        return({
                    'loss': float(loss),  # this is the a mandatory field to run hyperband
                    'info': config
                })
    
    @staticmethod
    def get_configspace():
        space = {
            'dense_layers': [1,2,3],
            'neurons': [128, 256, 512, 1024],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            # 'embed': [20,25,30],
            'lr': [0.001, 0.0005, 0.002, 0.0001],
        }
        cs = CS.ConfigurationSpace(space)
        return cs
    
    
if __name__ == '__main__':
    w = Chemception_wroker(sleep_interval=0)
    config = w.get_configspace().sample_configuration().get_dictionary()
    res = w.compute(config=config, budget=1)
    print(res)