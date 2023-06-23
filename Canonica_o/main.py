import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
import seaborn as sb
import matplotlib.pyplot as plt

tabel_industrie=pd.read_csv("dataIn/Industrie.csv",index_col=0)
tabel_populatie=pd.read_csv("dataIN/PopulatieLocalitati.csv",index_col=0)
variabile=list(tabel_industrie.columns[1:])

tabel_merged=tabel_populatie.merge(tabel_industrie, on="Siruta")
tabel_merged.to_csv("merged.csv")

def CAloc(linie,variabile):
    nrLocuitori=linie["Populatie"]
    valori=linie[variabile].values
    localitate=linie["Localitate_x"]
    cifra=valori/nrLocuitori
    data=list(cifra)
    data.insert(0,localitate)
    return pd.Series(data=data,index=[["Localitate"]+variabile])
cerinta1=tabel_merged.apply(func=CAloc,variabile=variabile,axis=1)
cerinta1.to_csv("cerinta1.csv")

#Cerinta2
tabelGrupat=tabel_merged.groupby("Judet").sum(numeric_only=True)
print(tabelGrupat)

def cifraMax(linie,variabile):
    valori=linie[variabile]
    index_max=np.argmax(valori)
    maxim=valori[index_max]
    activitate=valori.index[index_max]
    data=[]
    data.append(activitate)
    data.append(maxim)
    return pd.Series(data=data,index=["Acctivitate","Cifra de Afaceri"])

cerinta2=tabelGrupat.apply(func=cifraMax,variabile=variabile,axis=1)
cerinta2.to_csv("cerinta2.csv")

#Analiza Canonica

tabel_canonica=pd.read_csv("dataIN/DataSet_34.csv",index_col=0)
instante=list(tabel_canonica.index)
def nan_change(t):
    assert isinstance(t,pd.DataFrame)
    for v in t.columns:
        if any(t[v].isna()):
            t[v].fillna(t[v].mean(),inplace=True)
        else:
            t[v].fillna((t[v].mode()[0]),inplace=True)
nan_change(tabel_canonica)

variabileX=list(tabel_canonica.columns[:4])
print(variabileX)
variabileY=list(tabel_canonica.columns[4:])
print(variabileY)

x=tabel_canonica[variabileX].values
y=tabel_canonica[variabileY].values
print(x)
print(y)
n,p=np.shape(x)
q=np.shape(y)[1]
print(q)
m=min(p,q)

etichete_z=["Z"+str(i+1)for i in range(m)]
etichete_u=["U"+str(i+1)for i in range(m)]

#Construire model
model_cca=CCA(n_components=m)
model_cca.fit(x,y)

#Preluare valori,scoruri,radacinile Z si U
z,u=model_cca.transform(x,y)
df_z=pd.DataFrame(data=z,index=instante,columns=etichete_z)
df_z.to_csv("z.csv")
df_u=pd.DataFrame(data=u,index=instante,columns=etichete_u)
df_u.to_csv("u.csv")

def plot_corelatii(df_z,z1,z2,df_u,u1,u2,titlu="Plot"):
    fig=plt.figure(titlu,figsize=(9,9))
    ax=fig.add_subplot(1,1,1)
    ax.set_title(titlu)
    ax.set_xlabel(z1+"/"+z2)
    ax.set_ylabel(u2+"/"+u2)
    theta=np.arange(0,2*np.pi,0.01)
    ax.plot(np.cos(theta),np.sin(theta))
    ax.axhline(0)
    ax.axvline(0)
    ax.scatter(df_z[z1],df_z[z2],label="Spatiul X")
    ax.scatter(df_u[u1], df_u[u2], label="Spatiul Y")
    for i in range(len(df_z)):
        ax.text(df_z[z1].iloc[i],df_z[z2].iloc[i],df_z.index[i])
    for i in range(len(df_u)):
        ax.text(df_u[u1].iloc[i],df_u[u2].iloc[i],df_u.index[i])
    ax.legend()


plot_corelatii(df_z,"Z1","Z2",df_u,"U1","U2","Radacinile Z si U in acelasi plan")


#Calcul corelatii variabile observate
rxz=model_cca.x_loadings_
df_rxz=pd.DataFrame(data=rxz,index=variabileX,columns=etichete_z)
df_rxz.to_csv("rxz.csv")
ryu=model_cca.y_loadings_
df_ryu=pd.DataFrame(data=ryu,index=variabileY,columns=etichete_u)
df_ryu.to_csv("ryu.csv")

def corelograma(x,min,max,titlu="Corelgorama"):
    fig = plt.figure(titlu, figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    ax=sb.heatmap(data=x,vmin=min,vmax=max,cmap="bwr",annot=True,ax=ax)
    for i in range(len(x)):
        ax.set_xticklabels(labels=x.columns,ha="right",rotation=30)

corelograma(df_rxz,-1,1,"rxz Corelogram")
corelograma(df_z,-1,1,"Z Root Corelogram")
corelograma(df_ryu,-1,1,"ryu Corelogram")
corelograma(df_u,-1,1,"U Root Corelogram")
#plt.show()

#Calcul corelatii radacini canonice
r=np.diag(np.corrcoef(z,u,rowvar=False)[:m,m:])
df_r=pd.DataFrame(data={"Valoare":r})
df_r.to_csv("r.csv")

#Relevanta radacini oarecare
#Se face cu bartlett, mi a fost sila sa l mai invat
r2=r*r
p_values=(r2,n,p,q,m)
print(p_values)





