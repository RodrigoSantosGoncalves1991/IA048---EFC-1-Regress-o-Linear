import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, PowerTransformer

manchas_solares = pd.read_csv('/content/drive/My Drive/Colab Notebooks/monthly-sunspots.csv')

for k in np.arange(1, 9):
  manchas_solares['x[n-'+str(k)+']'] = manchas_solares['Monthly Mean Total Sunspot Number']
  manchas_solares['x[n-'+str(k)+']'] = manchas_solares['x[n-'+str(k)+']'].shift(k)

datas = np.array(manchas_solares.iloc[3132:, 1:2]).reshape(1, -1)
datas = datas[0]

manchas_solares_x = manchas_solares.iloc[8:, 3:11]
manchas_solares_y = manchas_solares.iloc[8:, 2:3]

x = np.asarray(manchas_solares_x)
y = np.asarray(manchas_solares_y)

np.random.seed(3)
W = np.random.uniform(low=-0.01, high= 0.01, size=(8,100))

WT_x_X = np.zeros((len(x),100))
for T in np.arange(0, len(x)):
  WT_x_X[T,:] = np.dot(W.T, x[T,:].reshape(-1, 1)).reshape(1, -1)

WtX = (2.7 / np.max(np.abs(WT_x_X))) * WT_x_X

X = np.tanh(WtX)

rmse_validation_medium_array = []
rmse_train_medium_array = []
rmse_test_medium_array = []
alpha_best_medium_array = []
rmse_test_minimum = 1000000
rmse_validation_minimum = 1000000

for T in np.arange(1, 101):
  linha_maxima_kfold = 3124

  Xt = X[:linha_maxima_kfold,0:T]
  _y = y[:linha_maxima_kfold,:]
  Xt_teste = X[linha_maxima_kfold:,0:T]
  _y_teste = y[linha_maxima_kfold:,:]

  kf = KFold(n_splits = 10)

  rmse_validation_array = []
  rmse_train_array = []
  rmse_test_array = []
  alpha_best_array = []

  for train_index, validation_index in kf.split(Xt):
    X_train, X_validation = Xt[train_index], Xt[validation_index]
    y_train, y_validation = _y[train_index], _y[validation_index]

    linear_regression = RidgeCV(alphas=list(np.arange(0.01, 10.1, 0.01)))
    linear_regression.fit(X_train, y_train)

    print('Melhor alpha: ',linear_regression.alpha_)
    alpha_best = linear_regression.alpha_
    alpha_best_array.append(alpha_best)

    prediction_validation = linear_regression.predict(X_validation)
    predicition_train = linear_regression.predict(X_train)
    prediction_test = linear_regression.predict(Xt_teste)

    rmse_validation = np.sqrt(mean_squared_error(y_validation,prediction_validation))
    rmse_train = np.sqrt(mean_squared_error(y_train, predicition_train))
    rmse_test = np.sqrt(mean_squared_error(_y_teste, prediction_test))

    rmse_validation_array.append(rmse_validation)
    rmse_train_array.append(rmse_train)
    rmse_test_array.append(rmse_test)

    if T == 92:
      if (rmse_validation_minimum > rmse_validation):
        rmse_validation_minimum = rmse_validation
        prediction_validation_best = linear_regression.predict(Xt_teste)
        best_all_alpha_validation = alpha_best
        print("Melhor lambda de validação:",best_all_alpha_validation)
        Melhor_T_validacao = T

    if T == 16:
      if (rmse_test_minimum > rmse_test):
        rmse_test_minimum = rmse_test
        prediction_test_best = prediction_test
        best_all_alpha_test = alpha_best
        Melhor_T_teste = T
    
    print("Número de atributos T = " + str(T))
    print("Erro RMS de treino da interação = ", rmse_train)
    print("Erro RMS de validação da iteração = ", rmse_validation)
    print("Erro RMS de teste da iteração = ", rmse_test)

  rmse_validation_medium = np.mean(rmse_validation_array)
  rmse_train_medium = np.mean(rmse_train_array)
  rmse_test_medium = np.mean(rmse_test_array)
  alpha_best_medium = np.mean(alpha_best_array)

  rmse_validation_medium_array.append(rmse_validation_medium)
  rmse_train_medium_array.append(rmse_train_medium)
  rmse_test_medium_array.append(rmse_test_medium)
  alpha_best_medium_array.append(alpha_best_medium)

print('valor minimo rmse treino:',min(rmse_train_medium_array))
print('Valor minimo rmse validação:',min(rmse_validation_medium_array))
print('Valor minimo rmse teste:',min(rmse_test_medium_array))

print('Melhor quantidade de parâmetros T no treino:',rmse_train_medium_array.index(min(rmse_train_medium_array))+1)
print('Melhor quantidade de parâmetros T na validação:',rmse_validation_medium_array.index(min(rmse_validation_medium_array))+1)
print('Melhor quantidade de parâmetros T no teste:',rmse_test_medium_array.index(min(rmse_test_medium_array))+1)

print('Melhor lambda validação:', best_all_alpha_validation)
print('Melhor lambda teste:', best_all_alpha_test)

print(rmse_train_medium_array)
print(rmse_validation_medium_array)
print(rmse_test_medium_array)

parametros_T = np.arange(1, 101)

plt.title('RMSE médio de treino e validação')
plt.ylabel('RMSE')
plt.xlabel(r'$\ Atributos \ T  \ (1\  \leq K \leq 100 \ ) $')
plt.grid(True)
plt.plot(parametros_T, rmse_train_medium_array, color = 'b', label='RMSE Treino')
plt.plot(parametros_T, rmse_validation_medium_array, color = 'r', label='RMSE Validação')
plt.legend()
plt.show()

plt.title('RMSE médio de teste')
plt.ylabel('RMSE')
plt.xlabel(r'$\ Atributos \ T  \ (1\  \leq K \leq 100 \ ) $')
plt.grid(True)
plt.plot(parametros_T, rmse_test_medium_array, color = 'y', label='RMSE Teste')
plt.legend()
plt.show()

plt.title(r'$\ Parâmetro \ de \ regularização \ \lambda$')
plt.ylabel(r'$ \lambda $')
plt.xlabel(r'$\ Atributos \ T  \ (1\  \leq K \leq 100 \ ) $')
plt.plot(parametros_T, alpha_best_medium_array, color = 'g')
plt.show()

#print(np.sqrt(mean_squared_error(_y_teste, prediction_test_best)))
#plt.plot(_y_teste, color = 'b')
#plt.plot(prediction_test_best, color = 'r')
#plt.show()

print('RMSE de validação mínimo : ',np.sqrt(mean_squared_error(_y_teste, prediction_validation_best)))
plt.title('Série de manchas solares, conjuto de dados de teste')
plt.ylabel('Número de manchas solares por mês')
plt.xlabel('Tempo (anos)')
plt.grid(True)
plt.plot(datas,_y_teste, color = 'b', label='Série original')
plt.plot(datas,prediction_validation_best, color = 'r', label='Predição, T = 92')
plt.xticks(rotation=45)
plt.xticks(datas[np.arange(0,len(datas),12)])
plt.legend(loc='best')
plt.savefig('figura_manchassolares1.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()

print('RMSE de teste mínimo : ',np.sqrt(mean_squared_error(_y_teste, prediction_test_best)))
plt.title('Série de manchas solares, conjuto de dados de teste')
plt.ylabel('Número de manchas solares por mês')
plt.xlabel('Tempo (anos)')
plt.grid(True)
plt.plot(datas,_y_teste, color = 'b', label='Série original')
plt.plot(datas,prediction_test_best, color = 'y', label='Predição, T = 16')
plt.xticks(rotation=45)
plt.xticks(datas[np.arange(0,len(datas),12)])
plt.legend(loc='best')
plt.savefig('figura_manchassolares2.png', dpi=1080, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None,transparent=False, bboxinches=None, padinches=0.1,frameon=None, bbox_inches='tight')
plt.show()
print("Melhor T validação: ", Melhor_T_validacao)
print("Melhor lambda validação: ",best_all_alpha_validation)
print("Melhor T teste:", Melhor_T_teste)
print("Melhor lambda teste: ",best_all_alpha_test)