# %%

from sklearn import tree
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# %%

base = pd.read_excel('Data/dados_cerveja.xlsx')

# %%

labelencoder_copo = LabelEncoder()
labelencoder_espuma = LabelEncoder()
labelencoder_cor = LabelEncoder()

x_cervejas = base.iloc[:,:5].values
y_cervejas = base.iloc[:,5].values

x_cervejas [:,2] = labelencoder_copo.fit_transform(x_cervejas[:,2])
x_cervejas [:,3] = labelencoder_espuma.fit_transform(x_cervejas[:,3])
x_cervejas [:,4] = labelencoder_cor.fit_transform(x_cervejas[:,4])

# %%

model = tree.DecisionTreeClassifier(random_state=42,
                                    max_depth=2,
                                    min_samples_split=3)
model.fit (x_cervejas,y_cervejas)

# %%


tree.plot_tree(model, 
               feature_names = x_cervejas,
               class_names= model.classes_,
               filled= True)


 # %% 
