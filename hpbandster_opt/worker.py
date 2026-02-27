import numpy as np
import time
import pandas as pd

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from rdkit import Chem
from rdkit.Chem import AllChem


import ConfigSpace as CS
from hpbandster.core.worker import Worker
from hpbandster.core.worker import Worker

from chemception.model import Chemception
from chemception.featurizer import ChemCeptionizer

class Chemception_wroker(Worker):

    def __init__(self, sleep_interval=0, **kwargs):
        super().__init__(**kwargs)
        self.train_df = pd.read_csv('../data/train_aid686978.csv')
        self.val_df = pd.read_csv('../data/val_aid686978.csv')
        self.sleep_interval = sleep_interval
    
        
    def _format_data(self, df, dataset_type):
        if dataset_type == 'train':
            try:
                return self.X_train, self.y_train
            except:
                pass
        if dataset_type == 'val':
            try:
                return self.X_val, self.y_val
            except:
                pass
            
        featurizer = ChemCeptionizer()
        df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
        df["molimage"] = df["mol"].apply(featurizer.featurize)
        df.dropna(subset=["molimage"], inplace=True)
        X = df["molimage"].to_numpy()
        X = np.stack(X, axis=0)
        y = df["label"]
        
        if dataset_type == 'train':
            self.X_train = X
            self.y_train = y
        if dataset_type == 'val':
            self.X_val = X
            self.y_val = y
            
        return X, y        
        
        
    def compute(self, config, budget, **kwargs):

        learning_rate = config.pop('lr')
        
        
        print('Building the model ..')
        model = Chemception(config=config)
        model = model.build()
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        
        batch_size = 32
        steps_per_epoch = len(self.train_df) // batch_size
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=1e-6, verbose=1)

        print('Featurizing the data (Chemceptionizing)')
        X_train, y_train = self._format_data(self.train_df, 'train')
        X_val, y_val = self._format_data(self.val_df, 'val')
        
        generator = ImageDataGenerator(data_format='channels_last')
        g = generator.flow(X_train, y_train, batch_size=batch_size)
        
        print('Fitting the model ..')
        history = model.fit(g,
                            steps_per_epoch=steps_per_epoch,
                            epochs=int(budget), verbose=2,
                            validation_data=(X_val, y_val),
                            callbacks=[reduce_lr]
                            )
            
        
        loss = history.history['val_loss'][-1]
        print(loss)
        time.sleep(self.sleep_interval)

        return({
                    'loss': float(loss),  # this is the a mandatory field to run hyperband
                    # 'info': config
                })
    
    @staticmethod
    def get_configspace():
        space = {
            'N': [8,16,32,64],
            'inceptionA_count': [1, 2, 3],
            'inceptionB_count': [1, 2, 3],
            'inceptionC_count': [1, 2, 3],
            'reductionA_count': [0,1],
            'reductionB_count': [0,1],
            'lr': [1e-4, 0.5e-4, 1e-5, 0.5e-5, 1e-6],
        }
        cs = CS.ConfigurationSpace(space)
        return cs
    
    
if __name__ == '__main__':
    w = Chemception_wroker(sleep_interval=0)
    config = w.get_configspace().sample_configuration().get_dictionary()
    res = w.compute(config=config, budget=1)
    print(res)