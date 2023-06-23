import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hclust
from sklearn.decomposition import PCA

tabel_alcool=pd.read_csv("DateIN/alcohol.csv",index_col=0)
tabel_coduri=pd.read_csv("DateIN/CoduriTariExtins.csv",index_col=0)

def medie(linie):
    valori=linie[1:]
    cod=linie["Code"]
    medie=np.mean(valori)
    data=[]
    data.append(cod)
    data.append(medie)
    return pd.Series(data=data,index=["Cod","Medie"])

cerinta1=tabel_alcool.apply(func=medie,axis=1)
cerinta1.to_csv("cerinta1.csv")

merged=tabel_coduri.merge(tabel_alcool,left_index=True,right_index=True).groupby("Continent").mean(numeric_only=True)

def anMaxim(linie):
    valori = linie[:]
    index_max=np.argmax(valori)
    return pd.Series(data=valori.index[index_max],index=["An"])
cerinta2=merged.apply(func=anMaxim,axis=1)
cerinta2.to_csv("cerinta2.csv")

#Cluster
variabile=list(tabel_alcool.columns[1:])
instante=list(tabel_alcool.index)

def nan_change(x):
    assert isinstance(x,pd.DataFrame)
    for v in x.columns:
        if any(x[v].isna()):
            x[v].fillna(x[v].mean(),inplace=True)
        else:
            x[v].fillna(x[v].mode()[0], inplace=True)

nan_change(tabel_alcool)

x=tabel_alcool[variabile].values
# print(x)
n=len(instante)
m=len(variabile)
#Construim ierarhie
h=hclust.linkage(x,method="ward")
print(h)
#nr jonctiuni
p=n-1
print("jonctiuni",p)
#distanta maxima intre 2 clusteri
k_diff_max=np.argmax(h[1:,2]-h[:(p-1),2])
print("distanta",k_diff_max)
#nr clusteri
nr_clusteri=p-k_diff_max
print("nr_clusteri",nr_clusteri)

def dendograma(h,instante,prag=None,titlu="Dendograma"):
    fig=plt.figure(titlu,figsize=(9,9))
    ax=fig.add_subplot(1,1,1)
    ax.set_title(titlu)
    hclust.dendrogram(h,labels=instante,color_threshold=prag,ax=ax)

def partitie(h,nr_clusteri,p,instante,titlu="Partitie"):
    k_diff_max=p-nr_clusteri
    prag=(h[k_diff_max,2]+h[k_diff_max+1,2])/2
    dendograma(h,instante,prag,titlu)
    n=p+1
    c=np.arange(n)
    for i in range(n-nr_clusteri):
        k1=h[i,0]
        k2=h[i,1]
        c[c==k1]=n+i
        c[c==k2]=n+i
    coduri=pd.Categorical(c).codes
    return np.array(["c"+str(cod)for cod in coduri])

partitie_optima=partitie(h,nr_clusteri,p,instante,"Partitie Optima")

df_partitieOptima=pd.DataFrame(data={"Tara":instante,"Cluster":partitie_optima})
df_partitieOptima.to_csv("partitieOptima.csv")

#plot partitie in axe principale
model_acp=PCA(2)
z=model_acp.fit_transform(x)

def plot_partitie(z,partitie_optima,titlu):
    fig = plt.figure(titlu, figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    sb.scatterplot(x=z[:,0],y=z[:,1],hue=partitie_optima,hue_order=np.unique(partitie_optima),ax=ax,)



plot_partitie(z,partitie_optima,"In primele 2 axe")
plt.show()