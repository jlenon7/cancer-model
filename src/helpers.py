import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras.models as keras

from os.path import exists
from typing import Optional
from tensorflow.keras.layers import Dense, Dropout

class Path:
  def plots(self, path: Optional[str]):
    path = self.clean_path(path)

    return f'storage/plots/{path}'

  def storage(self, path: Optional[str]):    
    path = self.clean_path(path) 

    return f'storage{path}'

  def resources(self, path: Optional[str]):    
    path = self.clean_path(path) 

    return f'resources{path}'

  def clean_path(self, path: Optional[str]):
    if path is None:
      return ''

    if path.endswith('/') is True:
      path = path[:-1]

    if path.startswith('/') is True:
      return path 

    return f'/{path}'

path = Path()

def load_model():
  model_exists = exists('storage/cancer-model.keras')

  if (model_exists):
    return keras.load_model('storage/cancer-model.keras')

  model = keras.Sequential() 

  model.add(Dense(30, activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(15, activation='relu'))
  model.add(Dropout(0.5))
  # In binary classifiction problems we want the 
  # last neuron activation to be sigmoid.
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer='adam')

  return model

def create_df(file_path: str, with_plots = True):
  df = pd.read_csv(file_path)  

  if (with_plots is True):
    plot(
      path.plots('dataframe/c-benign-malign.png'),
      lambda: sns.countplot(x='benign_0__mal_1', data=df)
    )

    plot(
      path.plots('dataframe/c-benign-malign.png'),
      lambda: sns.countplot(x='benign_0__mal_1', data=df)
    )

    plot(
      path.plots('dataframe/bar-correlation.png'),
      lambda: sns.barplot(df.corr()['benign_0__mal_1'][:-1].sort_values())
    )

    plot(
      path.plots('dataframe/hm-correlation.png'),
      lambda: sns.heatmap(df.corr())
    )

  return df

def plot(figure_name: str, lamb):
 plt.figure(figsize=(10,8))
 lamb()
 plt.savefig(figure_name)
