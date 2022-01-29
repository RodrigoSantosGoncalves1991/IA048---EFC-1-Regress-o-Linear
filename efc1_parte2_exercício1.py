import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


manchas_solares = pd.read_csv('/content/drive/My Drive/Colab Notebooks/monthly-sunspots.csv')

for k in np.arange(1, 25):
  manchas_solares['x[n-'+str(k)+']'] = manchas_solares['Monthly Mean Total Sunspot Number']
  manchas_solares['x[n-'+str(k)+']'] = manchas_solares['x[n-'+str(k)+']'].shift(k)

rmse_validation_medium_array = []
rmse_train_medium_array = []
rmse_test_medium_array = []
linear_regression_array = []
rmse_test_minimum = 1000000
rmse_validation_minimum = 1000000
datas = np.array(manchas_solares.iloc[3132:, 1:2]).reshape(1, -1)
datas = datas[0]
print(datas)
for K in np.arange(1, 25):
  linha_maxima_kfold = 3132
  hiperparametro_K = K

  x = manchas_solares.iloc[hiperparametro_K:linha_maxima_kfold, 3:hiperparametro_K+3]
  y = manchas_solares.iloc[hiperparametro_K:linha_maxima_kfold, 2:3]
  x_test = manchas_solares.iloc[linha_maxima_kfold:, 3:hiperparametro_K+3]
  y_test = manchas_solares.iloc[linha_maxima_kfold:, 2:3]
  x = np.asarray(x)
  y = np.asarray(y)
  x_test = np.asarray(x_test)
  y_test = np.asarray(y_test)

  kf = KFold(n_splits = len(x)) 
  rmse_validation_array = []
  rmse_train_array = []
  rmse_test_array = []

  for train_index, validation_index in kf.split(x):
    x_train, x_validation = x[train_index], x[validation_index]
    y_train, y_validation = y[train_index], y[validation_index]

    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)

    prediction_validation = linear_regression.predict(x_validation)
    predicition_train = linear_regression.predict(x_train)
    prediction_test = linear_regression.predict(x_test)
    
    rmse_validation = np.sqrt(mean_squared_error(y_validation, prediction_validation))
    rmse_train = np.sqrt(mean_squared_error(y_train, predicition_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, prediction_test))

    rmse_validation_array.append(rmse_validation)
    rmse_train_array.append(rmse_train)
    rmse_test_array.append(rmse_test)

    if K == 24:
      if (rmse_validation_minimum > rmse_validation):
        rmse_validation_minimum = rmse_validation
        prediction_validation_best = linear_regression.predict(x_test)
        Melhor_K_validacao = K

    if (rmse_test_minimum > rmse_test):
      rmse_test_minimum = rmse_test
      prediction_test_best = prediction_test
      Melhor_K_teste = K

    print("Hiperparâmetro K = " + str(K))
    print("Erro RMS de treino da interação = ", rmse_train)
    print("Erro RMS de validação da interação = ", rmse_validation)
    print("Erro RMS de teste da interação = ", rmse_test)

  rmse_validation_medium = np.mean(rmse_validation_array)
  rmse_train_medium = np.mean(rmse_train_array)
  rmse_test_medium = np.mean(rmse_test_array)

  rmse_validation_medium_array.append(rmse_validation_medium)
  rmse_train_medium_array.append(rmse_train_medium)
  rmse_test_medium_array.append(rmse_test_medium)

print('valor minimo rmse treino:',min(rmse_train_medium_array))
print('valor minimo rmse validação:',min(rmse_validation_medium_array))
print('valor minimo rmse teste:',min(rmse_test_medium_array))

print('Melhor hiperparâmetro K no treino:',rmse_train_medium_array.index(min(rmse_train_medium_array))+1)
print('Melhor hiperparâmetro K na validação:',rmse_validation_medium_array.index(min(rmse_validation_medium_array))+1)
print('Melhor hiperparâmetro K no teste:',rmse_test_medium_array.index(min(rmse_test_medium_array))+1)

print(rmse_train_medium_array)
print(rmse_validation_medium_array)
print(rmse_test_medium_array)

hiperparametros = np.arange(1, 25)

plt.title('RMSE médio de treino e validação')
plt.ylabel('RMSE')
plt.xlabel(r'$\ Hiperparâmetro \ K  \ (1\  \leq K \leq 24 \ ) $')
plt.grid(True)
plt.plot(hiperparametros, rmse_train_medium_array, color = 'b', label='RMSE Treino')
plt.plot(hiperparametros, rmse_validation_medium_array, color = 'r', label='RMSE Validação')
plt.legend()
plt.show()

plt.title('RMSE médio de teste')
plt.ylabel('RMSE')
plt.xlabel(r'$\ Hiperparâmetro \ K  \ (1\  \leq K \leq 24 \ ) $')
plt.grid(True)
plt.plot(hiperparametros, rmse_test_medium_array, color = 'y', label='RMSE Teste')
plt.legend()
plt.show()

print('RMSE de validação mínimo : ',np.sqrt(mean_squared_error(y_test, prediction_validation_best)))
plt.title('Série de manchas solares, conjuto de dados de teste')
plt.ylabel('Número de manchas solares por mês')
plt.xlabel('Tempo (anos)')
plt.grid(True)
plt.plot(datas,y_test, color = 'b', label='Série original')
plt.plot(datas,prediction_validation_best, color = 'r', label='Predição, K = 24')
plt.xticks(rotation=45)
plt.xticks(datas[np.arange(0,len(datas),12)])
plt.legend(loc='best')
plt.savefig('figura_manchassolares1.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()

print('RMSE de teste mínimo : ',rmse_test_minimum)
plt.title('Série de manchas solares, conjuto de dados de teste')
plt.ylabel('Número de manchas solares por mês')
plt.xlabel('Tempo (anos)')
plt.grid(True)
plt.plot(datas,y_test, color = 'b', label='Série original')
plt.plot(datas,prediction_test_best, color = 'y', label='Predição, K = 15')
plt.xticks(rotation=45)
plt.xticks(datas[np.arange(0,len(datas),12)])
plt.legend(loc='best')
plt.savefig('figura_manchassolares2.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()

print("Melhor K validação: ", Melhor_K_validacao)
print("Melhor K teste:", Melhor_K_teste)