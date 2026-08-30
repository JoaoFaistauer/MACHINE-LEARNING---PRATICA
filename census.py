# %% 
# #IMPORTAÇÃO BIBLIOTECAS 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import Orange
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from yellowbrick.classifier import ConfusionMatrix
from collections import Counter

# %%
#BASE DE DADOS

base_census = pd.read_csv('Data/census.csv')

print(base_census.describe())
print(base_census.isnull().sum())

resultados = {}

# %%
# DEFINIÇÕES

def linha():
    print ('=' * 60)


def mostrar_ranking():
    ranking = sorted(resultados.items(), key = lambda item: item[1], reverse= True)

    linha()
    print('\033[92mRANKING COMPARATIVO DE ALGORITMOS\033[0m')
    linha()

    for posicao, (nome, accuracy) in enumerate (ranking, start=1):
        if accuracy < 0.759190:
            print('\033[92m{}º:\033[0m{} - \033[91m{:.2%}\033[0m'.format(posicao, nome, accuracy))
        elif posicao == 1:
            print('\033[92m{}º:\033[0m{} - {:.2%}'.format(posicao, nome, accuracy))
        else:
            print('\033[92m{}º:\033[0m{} - {:.2%}'.format(posicao, nome, accuracy))

# %%
#LABELENCODER E DEFINIÇÃO DE X E Y

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])   # type: ignore
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])  # type: ignore
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])    # type: ignore
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6]) # type: ignore
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7]) # type: ignore
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])       # type: ignore
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])        # type: ignore
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])  # type: ignore

# %%
# ONEHOT E SCALER

onehotenconder_census = ColumnTransformer(
    transformers=[('Onehot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
    remainder='passthrough'
)
x_census = onehotenconder_census.fit_transform(x_census)

scaler_census = StandardScaler(with_mean=False)
x_census = scaler_census.fit_transform(x_census)

# %%
#BASE DE TREINAMENTO E TESTE

x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(
    x_census, y_census, test_size=0.15, random_state=0
)

with open('Models/census.pkl', mode='wb') as f:
    pickle.dump([x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste], f)

# %%
# APRENDIZAGEM POR REGRAS - INICIO

base_census = Orange.data.Table('Data/census_regras.csv')

# %%
# APRENDIZAGEM POR REGRAS - DIVISAO DE BASES

base_dividida = Orange.evaluation.testing.sample(base_census, n = 0.08)

base_treinamento = base_dividida[1]
base_teste = base_dividida[0]
base_treinamento_menor = Orange.evaluation.testing.sample(base_treinamento, n = 0.10)[0]


# %%
# APRENDIZAGEM POR REGRAS - TRATAMENTO DE DADOS BASE TREINAMENTO

cn2 = Orange.classification.rules.CN2Learner()
regras_census = cn2 (base_treinamento_menor)

for regras in regras_census.rule_list:
    print (regras)

# %%
# APRENDIZAGEM POR REGRAS - PREVISOES

predict_rules = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: regras_census])
print('Resultado das previsões:\n',predict_rules)

# %%
# APRENDIZAGEM POR REGRAS - MAJORITY LEARNER

majority =  Orange.classification.MajorityLearner()
predict_majority = Orange.evaluation.testing.TestOnTestData(base_census, base_census, [majority])

# %%
# APRENDIZAGEM POR REGRAS - REGISTRO DE CLASSES E DEFINIÇÃO PONTO BASE PARA AVALIAÇÃO

for registro in  base_census:
    print (registro)

linha()
print(Counter(str(registro.get_class()) for registro in base_census))
linha()

min_accuracy = (24720/(24720 + 7841))

def verifica_accuracy(accuracy_algoritmo):
    print('Min accuracy para um algoritmo nessa base é: {:f}'.format(min_accuracy))
    if  accuracy_algoritmo <= min_accuracy:
        print ('\033[31mEsse algoritmo não é adequado para essa base de dados!\033[0m')
    else:
        print('\033[92mEsse alogoritmo é adequado para esta base de dados!\033[0m')

# %%
# APRENDIZAGEM POR REGRAS - COMPARAÇÕES FINAIS

resultados['Regras (CN2)'] = Orange.evaluation.CA(predict_rules)[0]
print('Accuracy Aprendizagem por regras:' , Orange.evaluation.CA(predict_rules))
verifica_accuracy(Orange.evaluation.CA(predict_rules))
linha()
resultados['Majority Learner'] = Orange.evaluation.CA(predict_majority)[0]
print ('Accuracy Majority Learner:' , Orange.evaluation.CA(predict_majority))
verifica_accuracy(Orange.evaluation.CA(predict_majority))

# %%
#APRENDIZAGEM BAYESIANA - INICIO

naive_census = GaussianNB()
naive_census.fit(x_census_treinamento.toarray(), y_census_treinamento)

# %%
#APRENDIZAGEM BAYESIANA - PREVISOES

predict_naive = naive_census.predict(x_census_teste.toarray())
print('Previsão base de teste:\n',predict_naive)
linha()
print ('Gabarito base de teste:\n',y_census_teste)

# %%
#APRENDIZAGEM BAYESIANA - COMPARAÇÃO

resultados['Naive Bayes'] = accuracy_score(y_census_teste, predict_naive)
print('Accuracy Naive Bayes:\n', accuracy_score(y_census_teste, predict_naive))
linha()
verifica_accuracy(accuracy_score(y_census_teste, predict_naive))
linha()
print('Matriz de confusão Naive Bayes:\n', confusion_matrix(y_census_teste, predict_naive))

# %%
#APRENDIZAGEM BAYESIANA - ANALISE DE PRECISÃO

cm = ConfusionMatrix(naive_census)
cm.fit(x_census_treinamento.toarray(), y_census_treinamento)
cm.score(x_census_teste.toarray(), y_census_teste)

print(classification_report(y_census_teste, predict_naive))

# %%
# COMPARAÇÃO GERAL - ALGORITMOS

mostrar_ranking()

# %%
#RANDOM FOREST - INICIO

randomforest_census = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
randomforest_census.fit(x_census_treinamento, y_census_treinamento)

# %%
#RANDOM FOREST - PREVISOES

predict_randomforest = randomforest_census.predict(x_census_teste)
print('Previsão base de teste:\n',predict_randomforest)
linha()
print ('Gabarito base de teste:\n',y_census_teste)

# %%
# RANDOM FOREST - COMPARAÇÃO

resultados['Random Forest'] = accuracy_score(y_census_teste, predict_randomforest)
print('Accuracy Random Forest:\n', accuracy_score(y_census_teste, predict_randomforest))
linha()
verifica_accuracy(accuracy_score(y_census_teste, predict_randomforest))
linha()
print('Matriz de confusão Random Forest:\n', confusion_matrix(y_census_teste, predict_randomforest))

# %%
#RANDOM FOREST - ANALISE DE PRECISÃO

cm = ConfusionMatrix(randomforest_census)
cm.fit(x_census_treinamento.toarray(), y_census_treinamento)
cm.score(x_census_teste.toarray(), y_census_teste)

print(classification_report(y_census_teste, predict_randomforest))

# %%
# COMPARAÇÃO GERAL - ALGORITMOS

mostrar_ranking()

# %%

#ALGORITMO kNN (INSTANCIAS) - INICIO

knn_census = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)
knn_census.fit(x_census_treinamento, y_census_treinamento)

# %%
# ALGORITMO kNN (INSTANCIAS) - PREVISOES

predict_knn = knn_census.predict(x_census_teste)
print ('Previsão base de teste:\n',predict_knn)
linha()
print ('Gabarito base de teste:\n', y_census_teste)

# %%
# ALGORITMO kNN (INSTANCIAS) - COMPARAÇÃO

resultados['kNN'] = accuracy_score(y_census_teste, predict_knn)
print('Accuracy algoritmo kNN:\n', accuracy_score(y_census_teste, predict_knn))
linha()
verifica_accuracy(accuracy_score(y_census_teste, predict_knn))
linha()
print('Matrix de confusão algoritmo kNN:\n', confusion_matrix(y_census_teste, predict_knn))

# %%
# ALGORITMO kNN (INSTANCIAS) - ANALISE DE PRECISÃO

cm = ConfusionMatrix(knn_census)
cm.fit(x_census_treinamento.toarray(), y_census_treinamento)
cm.score(x_census_teste.toarray(), y_census_teste)

print(classification_report(y_census_teste, predict_knn))

# %%
# COMPARAÇÃO GERAL - ALGORITMOS

mostrar_ranking()

# %%
# REGRESSÃO LOGISTICA - INICIO

logistic_census = LogisticRegression(random_state=1)
logistic_census.fit(x_census_treinamento,y_census_treinamento)

# %%
# REGRESSÃO LOGISTICA - PREVISOES

predict_log = logistic_census.predict(x_census_teste)
print('Previsão base de teste:\n', predict_log)
linha()
print('Gabarito base de teste:\n', y_census_teste)

# %%
# REGRESSÃO LOGISTICA - COMPARAÇÃO

resultados['Regressão Logistica'] = accuracy_score(y_census_teste, predict_log)
print('Accuracy Regressão Logistica:\n', accuracy_score(y_census_teste, predict_log))
linha()
print('Matrix de confusão Regressão Logistica:\n', confusion_matrix(y_census_teste, predict_log))
verifica_accuracy(accuracy_score(y_census_teste, predict_log))


# %%
# REGRESSÃO LOGISTICA - ANALISE DE PRECISÃO

cm = ConfusionMatrix(logistic_census)
cm.fit(x_census_treinamento.toarray(), y_census_treinamento)
cm.score(x_census_teste.toarray(), y_census_teste)

print(classification_report(y_census_teste , predict_log))

# %%
# COMPARAÇÃO GERAL - ALGORITMOS

mostrar_ranking()

# %%
# SVM - INICIO

svm_census = SVC(kernel='linear', random_state=1, C=2.00)
svm_census.fit(x_census_treinamento,y_census_treinamento)


# %%
# SVM - PREVISOES

predict_svm = svm_census.predict(x_census_teste)
print('Previsao base de teste:\n', predict_svm)
linha()
print('Gabarito base de teste:\n', y_census_teste)


# %%
# SVM - COMPARAÇÃO

resultados['SVM'] = accuracy_score(y_census_teste, predict_svm)
print('Accuracy SVM:\n', accuracy_score(y_census_teste, predict_svm))
linha()
print('Matrix de confusão SVM:\n', confusion_matrix(y_census_teste,predict_svm))
verifica_accuracy(accuracy_score(y_census_teste, predict_svm))


# %%
# SVM - ANALISE DE PRECISÃO

cm = ConfusionMatrix(svm_census)
cm.fit(x_census_treinamento.toarray(), y_census_treinamento)
cm.score(x_census_teste.toarray(), y_census_teste)

print(classification_report(y_census_teste, predict_svm))

# %%
# COMPARAÇÃO GERAL - ALGORITMOS

mostrar_ranking()

# %%
