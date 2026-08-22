# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle 
import Orange
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# %%

base_credit_risk = pd.read_csv('Data/risco_credito.csv')

x_credit_risk = base_credit_risk.iloc[:,0:4].values

y_credit_risk = base_credit_risk.iloc[:,4].values

# %%

labelencoder_historia = LabelEncoder()
labelencoder_divida = LabelEncoder()
labelencoder_garantia = LabelEncoder()
labelencoder_renda = LabelEncoder()
labelencoder_risco = LabelEncoder()

x_credit_risk[:,0] = labelencoder_historia.fit_transform(x_credit_risk[:,0])
x_credit_risk[:,1] = labelencoder_divida.fit_transform(x_credit_risk[:,1])
x_credit_risk[:,2] = labelencoder_garantia.fit_transform(x_credit_risk[:,2])
x_credit_risk[:,3] = labelencoder_renda.fit_transform(x_credit_risk[:,3])

# %%

with open ('credit_risk.pkl', mode='wb') as f:
    pickle.dump([x_credit_risk,y_credit_risk], f)

# %%

naive_risk = GaussianNB()
naive_risk.fit(x_credit_risk,y_credit_risk)

# %%
#hist boa (0), hist desconhecida (1), hist ruim (2); divida alta (0), divida baixa (1)
#garantia adequada (0), garantia nenhuma (1); renda > 35 (2); renda < 15 (0)
predict_naive = naive_risk.predict([[0,0,1,2],[2,0,0,0]])
predict_naive

#naive_risk.classes_
#naive_risk.class_count_
#naive_risk.class_prior_

# %%

figuras, eixos = plt.subplots(nrows= 1, ncols= 1)
risk_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
risk_tree = risk_tree.fit (x_credit_risk,y_credit_risk)

tree.plot_tree(risk_tree,
               feature_names= ['historia','divida','garantias','renda'],
               class_names= ['alto','moderado','baixo']
);

# %%

base_credit_risk = Orange.data.Table('Data/risco_credito_regras.csv')

#  %%

base_credit_risk.domain
cn2 = Orange.classification.rules.CN2Learner()
rules_credit_risk = cn2(base_credit_risk)
for regras in rules_credit_risk.rule_list:
    print (regras)

# %%

predict_rules  = rules_credit_risk([['boa', 'alta', 'nenhuma', 'acima_35'] , ['ruim', 'alta', 'adequada', '0_15']])
for i in predict_rules:
    print(base_credit_risk.domain.class_var.values[i])

#  %%