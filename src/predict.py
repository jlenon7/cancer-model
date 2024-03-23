import numpy as np

from helpers import path, create_df, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = create_df(path.resources('cancer.csv'), with_plots=False)

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

original_exam = df.sample()
scaled_exam = scaler.transform(original_exam.drop('benign_0__mal_1', axis=1).values.reshape(-1, 30))

# For binary classification
exam_prediction = model.predict(scaled_exam)
exam_prediction = np.where(exam_prediction > 0.5, 1, 0)

print()
print('Exam result:          ', original_exam['benign_0__mal_1'].iloc[0])
print('Exam predicted result:', exam_prediction[0][0])
