import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from  sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import seaborn as sb
import matplotlib.pyplot as plt

tabel_train_test=pd.read_csv("dataIn/hernia.csv",index_col=0)

variabile=list(tabel_train_test.columns)
predictori=variabile[:-1]
tinta=variabile[-1]

x_train,x_test,y_train,y_test=train_test_split(tabel_train_test[predictori],tabel_train_test[tinta],test_size=0.4)

#Creare model liniar
model_lda=LinearDiscriminantAnalysis()
model_lda.fit(x_train,y_train)

#Predictie model liniar
predictie_model_lda=model_lda.predict(x_train)
print(predictie_model_lda)

#Predictie in testul de aplicare
x_apply=pd.read_csv("dataIn/hernia_apply.csv",index_col=0)
predictie_test_aplicatie=model_lda.predict(x_apply[predictori])
print(predictie_test_aplicatie)

#Creare model Bayes
model_b=GaussianNB()
model_b.fit(x_train,y_train)
#Predictie model  in bayes
predictie_b_test=model_b.predict(x_apply)
print(predictie_b_test)

#Calcul axe discriminante
clase=model_lda.classes_
#nr de functii discriminante
m=len(clase)-1

#calcul scoruri model liniar
z=model_lda.transform(x_test)
df_z=pd.DataFrame(data=z,index=x_test.index,columns=["Z"+str(i+1)for i in range(m)])
df_z.to_csv("zz.csv")

#Desen distributie-->o sg axa
def plot_distributie(z,y,k=0,titlu="Distributie"):
    fig=plt.figure(titlu,figsize=(9,9))
    ax=fig.add_subplot(1,1,1)
    ax.set_title(titlu)
    ax=sb.kdeplot(x=z[:,k],hue=y,fill=True,ax=ax)

for i in range(m):
    plot_distributie(z,y_test,i,"Distributie")

#Desen scoruri in 2 axe
def plot_instante(z,y,clase,k1=0,k2=1,titlu="2Axe"):
    fig = plt.figure(titlu, figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    ax.set_xlabel("z"+str(k1+1))
    ax.set_ylabel("z"+str(k2+1))
    ax=sb.scatterplot(x=z[:,k1],y=z[:,k2],hue=y,hue_order=clase,ax=ax)

for i in range(m-1):
    for j in range(i+1,m):
        plot_instante(z,y_test,clase,i,j,"2Axe")
plt.show()

