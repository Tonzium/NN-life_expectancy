import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

dataset = pd.read_csv("life_expectancy.csv")

print("Dataset Preview:")
print(dataset.head())

print("Dataset Description:")
print(dataset.describe())

#remove country to avoid confusion in model
df = dataset.drop("Country", axis=1)

#set labels and features
labels = df["Life expectancy"]
features = df.iloc[:, 0:-1]

#hot-encode categorical variables
features = pd.get_dummies(df)

#split data
features_train, features_test, labels_train,  labels_test = train_test_split(features, labels, test_size=0.2, random_state=9)

#standardize data
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

#transform data
features_train_scaled = ct.fit_transform(features_train)

features_test_scaled = ct.transform(features_test)

print(features_train_scaled.shape)

# my model
my_model = Sequential()
my_input = InputLayer(input_shape = (features_train_scaled.shape[1], ))

#build model, input layer, hidden layer with 64 units and output layer with single output.
my_model.add(my_input)
my_model.add(Dense(64, activation = "relu"))
my_model.add(Dense(1))
print(my_model.summary())

#create Adam optimizer and compile
opt = Adam(learning_rate = 0.01)
my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

#fit model
my_model.fit(features_train_scaled, labels_train, epochs = 40, batch_size = 1, verbose = 1)

#evaluate model
final_mse, final_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)
print("Result mse:", final_mse)
print("Result mae:", final_mae)

# make predictions
predictions = my_model.predict(features_test_scaled)

#visualize
sns.set(style='darkgrid')
plt.figure(figsize=(40, 6))
sns.lineplot(x=range(len(labels_test)), y=labels_test, color='blue', label='Actual Values')
sns.lineplot(x=range(len(predictions)), y=predictions.flatten(), color='red', label='Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.title('Actual vs. Predicted Values')
plt.legend()

# save the figure
plt.savefig('visualization.png')