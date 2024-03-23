import numpy as np
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from helpers import path, create_df, load_model, plot 
from sklearn.metrics import confusion_matrix, classification_report

df = create_df(path.resources('cancer.csv'))

X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.25,
  random_state=101
)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = load_model()

model.fit(
  x=X_train,
  y=y_train,
  epochs=600,
  validation_data=(X_test, y_test),
  callbacks=[
    EarlyStopping(
      monitor='val_loss',
      mode='min',
      verbose=1,
      patience=25
    )
  ]
)

plot(
  path.plots('model/is-overfitting-train-test-data.png'), 
  lambda: sns.lineplot(model.history.history)
)

# For binary classification
predictions = model.predict(X_test)
predictions = np.where(predictions > 0.5, 1, 0)
predictions = predictions.flatten()

print()
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print()
print('Classification Report:')
print(classification_report(y_test, predictions, zero_division=0))

print()
print('Saving the model at', path.storage('cancer-model.keras'))
model.save(path.storage('cancer-model.keras'))
