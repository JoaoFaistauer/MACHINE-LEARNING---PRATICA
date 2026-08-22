# %% 
# IMPORTAÇÃO BIBLIOTECAS

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import Orange
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from collections import Counter

# %%
# BASE DE DADOS
 
base_credit = pd.read_csv('Data/credit_data.csv')
resultados = {}

# %%
#DEFINIÇÕES

def  linha():
    print ('=' * 60)


def mostrar_ranking():
    ranking = sorted(resultados.items(), key = lambda item: item[1], reverse= True)

    linha()
    print('\033[92mRANKING COMPARATIVO DE ALGORITMOS\033[0m')
    linha()

    for posicao, (nome, accuracy) in enumerate (ranking, start=1):
        if posicao == 1:
            print('\033[92m{}º:\033[0m{} - {:.2%}'.format(posicao, nome, accuracy))
        elif accuracy < 0.858500:
            print('\033[92m{}º:\033[0m{} - \033[91m{:.2%}\033[0m'.format(posicao, nome, accuracy))
        else:
            print('\033[92m{}º:\033[0m{} - {:.2%}'.format(posicao, nome, accuracy))


# %%
# TRATAMENTO DE VALORES INCONSISTENTES 

#base_credit2 = base_credit.drop(base_credit[base_credit['age'] < 0].index) (ISOLAR AS INCONSISTENTES)
#base_credit2['age'].mean() (ACHAR A MÉDIA)
base_credit.loc[base_credit['age']<0, 'age'] = 40.92
#base_credit.loc[base_credit['age']< 0] 

# %% 
# TRATAMENTO DE VALORES FALTANTES

#base_credit.isnull().sum() (ENCONTRAR NA)
#base_credit.loc[pd.isnull(base_credit['age'])] 
base_credit.fillna(base_credit['age'].mean(), inplace = True)
base_credit.loc[(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)]

# %%
# VISUALIZAÇÃO DE DADOS
#base_credit.describe()

grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
#grafico.show()

#np.unique(base_credit['default'], return_counts=True)
#sns.countplot(x = base_credit['default']);
#plt.hist (x= base_credit['loan']);

# %%
# PREVISORES X CLASSES

x_credit = base_credit.iloc[:,1:4].values
y_credit = base_credit.iloc[:,4].values


scaler_credit = StandardScaler()
x_credit = scaler_credit.fit_transform(x_credit)

# %% 
# BASE DE TREINAMENTO E TESTE

x_credit_treinamento , x_credit_teste , y_credit_treinamento , y_credit_teste = train_test_split(x_credit, y_credit, test_size=0.25, random_state=0)
with open ('credit.pkl', mode = 'wb') as f:
    pickle.dump([x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f)

# %%
# APRENDIZAGEM POR REGRAS - INICIO

base_credit = Orange.data.Table('Data/credit_data_regras.csv')
base_credit.domain

# %%
# APRENDIZAGEM POR REGRAS - DIVISAO DE BASES

base_dividida = Orange.evaluation.testing.sample(base_credit, n = 0.25)

base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

# %%
# APRENDIZAGEM POR REGRAS - TRATAMENTO DE DADOS BASE TREINAMENTO

cn2 = Orange.classification.rules.CN2Learner()
regras_credit = cn2 (base_treinamento)

for regras in regras_credit.rule_list:
    print (regras)

# %%
# APRENDIZAGEM POR REGRAS - PREVISOES

predict_rules = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: regras_credit])

# %%
# APRENDIZAGEM POR REGRAS - MAJORITY LEARNER

majority =  Orange.classification.MajorityLearner()
predict_majority = Orange.evaluation.testing.TestOnTestData(base_credit, base_credit, [majority])

# %%
# APRENDIZAGEM POR REGRAS - REGISTRO DE CLASSES E DEFINIÇÃO PONTO BASE PARA AVALIAÇÃO

for registro in  base_credit:
    print (registro)

linha()
print(Counter(str(registro.get_class()) for registro in base_credit))
linha()

min_accuracy = (1717/2000)

def verifica_accuracy(accuracy_algoritmo):
    print('Min accuracy para um algoritmo nessa base é: {:f}'.format(min_accuracy))
    if  accuracy_algoritmo <= min_accuracy:
        print ('\033[31mEsse algoritmo não é ideal para essa base de dados!\033[0m')
    else:
        print('\033[92mEsse alogoritmo é ideal para esta base de dados!\033[0m')

# %% 
# APRENDIZAGEM POR REGRAS - COMPARAÇÕES FINAIS

resultados['Regras (CN2)'] = Orange.evaluation.CA(predict_rules)[0]
print ('Accuracy Aprendizagem por regras:', Orange.evaluation.CA(predict_rules))
verifica_accuracy (Orange.evaluation.CA(predict_rules))
linha()
resultados['Majority Learner'] = Orange.evaluation.CA(predict_rules)[0]
print ('Accuracy Majority Learner:', Orange.evaluation.CA(predict_majority))
verifica_accuracy(Orange.evaluation.CA(predict_majority))

#  %%
# APRENDIZAGEM BAYESIANA - INICIO

naive_credit = GaussianNB()
naive_credit.fit(x_credit_treinamento,y_credit_treinamento)

# %%
# APRENDIZAGEM BAYESIANA - PREVISOES

predict_naive = naive_credit.predict(x_credit_teste)
print ('Previsão base de teste:\n',predict_naive)
print ('Gabarito base de teste:\n', y_credit_teste)

# %%
# APRENDIZAGEM BAYESIANA - COMPARAÇÃO 

resultados['Naive Bayes'] = accuracy_score(y_credit_teste, predict_naive)
print ('Accuracy Naive Bayes:\n' ,accuracy_score(y_credit_teste,predict_naive))
verifica_accuracy(accuracy_score(y_credit_teste,predict_naive))
linha()
print ('Matrix de erro Naive Bayes:\n',confusion_matrix(y_credit_teste, predict_naive))

# %%
# APRENDIZAGEM BAYESIANA - ANALISE DE PRECISÃO

cm = ConfusionMatrix(naive_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, predict_naive))

# %%
# COMPARAÇÃO GERAL - ALGORITMOS

mostrar_ranking()

# %%

# RANDOM FOREST - INICIO

credit_tree = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
credit_tree = credit_tree.fit (x_credit_treinamento, y_credit_treinamento)

# %%
# RANDOM FOREST - PREVISOES

predict_tree = credit_tree.predict (x_credit_teste)
print ('Previsão base de teste:\n',predict_tree)
print ('Gabarito base de teste:\n', y_credit_teste)

# %%
# RANDOM FOREST - COMPARAÇÃO

resultados['Random Forest'] = accuracy_score(y_credit_teste, predict_tree)
print ('Accuracy Random Forest:\n' ,accuracy_score(y_credit_teste,predict_tree))
verifica_accuracy(accuracy_score(y_credit_teste,predict_tree))
linha()
print ('Matrix de erro Random Forest:\n',confusion_matrix(y_credit_teste, predict_tree))

#  %%
# RANDOM FOREST - ANALISE DE PRECISÃO

cm = ConfusionMatrix(credit_tree)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, predict_tree))

# %%
# COMPARAÇÃO GERAL - ALGORITMOS

mostrar_ranking()

# %%
# ALGORITMO kNN (INSTANCIAS) - INICIO

knn_credit = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)
knn_credit.fit (x_credit_treinamento, y_credit_treinamento)

# %%
# ALGORITMO kNN (INSTANCIAS) - PREVISOES

predict_knn = knn_credit.predict(x_credit_teste)
print ('Previsão base de teste:\n', predict_knn)
linha()
print('Gabarito base de teste:\n', y_credit_teste)

# %%
# ALGORITMO kNN (INSTANCIAS) - COMPARAÇÃO

resultados['kNN'] = accuracy_score(y_credit_teste, predict_knn)
print('Accuracy algoritmo kNN:\n', accuracy_score(y_credit_teste, predict_knn))
linha()
print('Matrix de confusão algoritmo kNN:\n', confusion_matrix(y_credit_teste,predict_knn))

# %%
# ALGORITMO kNN (INSTANCIAS) - ANALISE DE PRECISÃO

cm = ConfusionMatrix(knn_credit)
cm.fit (x_credit_treinamento, y_credit_treinamento)
cm.score (x_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, predict_knn))

# %%
# COMPARAÇÃO GERAL - ALGORITMOS

mostrar_ranking()

# %%

