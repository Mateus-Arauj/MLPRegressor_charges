import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
data = pd.read_csv('insurance.csv')
# Tratar os dados
data['sex'] = data.get('sex').replace({'male':0, 'female': 1})
data['smoker'] = data.get('smoker').replace({'no': 0, 'yes': 1})
#Descobrir quais são as regiões
#print(sorted(set(data['region'])))
data['region'] = data.get('region').replace({'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3})

y = data.get('charges')
X = data.iloc[:,:6]

# print(y.head())
# print(X.head())

#Divisão para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#Definindo os parametros do modelo e aplicando a regressão 
regr = MLPRegressor(hidden_layer_sizes= (100,100), random_state=35, max_iter=4000).fit(X_train, y_train)
y_pred = regr.predict(X_test)
#Salvando modelo


# Função de Erro
result = mean_absolute_error(y_test, y_pred)
print(result)