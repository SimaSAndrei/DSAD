import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sb
import matplotlib.pyplot as plt

#Citire din fisier csv
tabel_coduri=pd.read_csv("DateIN/CoduriTariExtins.csv",index_col=0)
tabel_mortalitate=pd.read_csv("DateIN/Mortalitate.csv",index_col=0)
#Aleg index_col ca fiind coloana comuna dintre tabele

#Cerinta1

#Daca doresc anumite coloane sa afisez fac asa
    #cerinta1=tabel_mortalitate.loc[tabel_mortalitate["RS"]<0,["RS","FR"]]
#Daca doresc tot:
cerinta1=tabel_mortalitate.loc[tabel_mortalitate["RS"]<0]
cerinta1.to_csv("cerinta1.csv")

#Cerinta2

cerinta2=tabel_coduri.merge(tabel_mortalitate,left_index=True,right_index=True).groupby("Continent").mean(numeric_only=True)
cerinta2.to_csv("cerinta2.csv")

#ACP
#Mai intai rezolv cerintele subiectului dupa care completez cu restul cerintelor posibile
variabile=list(tabel_mortalitate.columns[:])
instante=list(tabel_mortalitate.index)

#Metoda pentru inlocuire valori nefinite in valori finite-->Valabila la toate analizele

def nan_change(t):
    assert  isinstance(t,pd.DataFrame)
    for v in t.columns:
        if any(t[v].isna()):
            t[v].fillna(t[v].mean(),inplace=True)
        else:
            t[v].fillna(t[v].mode()[0],inplace=True)
nan_change(tabel_mortalitate)

x=tabel_mortalitate[variabile].values

x=(x-np.mean(x,axis=0))/np.std(x,axis=0)

#Construim modelul

model_acp=PCA()
model_acp.fit(x)

#Variante Componente principale . Afisare la consola si in csv
alpha=model_acp.explained_variance_
print(alpha)
df_alpha=pd.DataFrame(data={"Valoare":alpha},index=variabile)
df_alpha.to_csv("variante.csv")

#Scoruri
c=model_acp.transform(x)
scoruri=c/np.sqrt(alpha)
df_scoruri=pd.DataFrame(data=scoruri,index=instante,columns=["C"+str(i+1)for i in range(len(alpha))])
df_scoruri.to_csv("scoruri.csv")

#Graficul scorurilor in primele doua axe principale
    #Nu stiu daca se refera la plot componente sau corelatii(adica in cerc de cos si sin)

def plot_componente(x,var_x,var_y,titlu="Plot componente"):
    fig=plt.figure(titlu,figsize=(9,9))
    ax=fig.add_subplot(1,1,1)
    ax.set_title(titlu)
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.scatter(x[var_x],x[var_y])
    for i in range(len(x)):
        ax.text(x[var_x].iloc[i],x[var_y].iloc[i],x.index[i])
plot_componente(df_scoruri,"C1","C2","Plot scoruri")

def plot_corelatii(x,var_x,var_y,titlu="Plot corelatii"):
    fig=plt.figure(titlu,figsize=(9,9))
    ax=fig.add_subplot(1,1,1)
    ax.set_title(titlu)
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    theta=np.arange(0,2*np.pi,0.01)
    ax.plot(np.cos(theta),np.sin(theta),c="r")
    ax.scatter(x[var_x],x[var_y])
    for i in range(len(x)):
        ax.text(x[var_x].iloc[i],x[var_y].iloc[i],x.index[i])
plot_corelatii(df_scoruri,"C1","C2","Plot Corelatii scoruri")

#A ramas sa mai facem vectori proprii matrice de covarianta , componente corelatii, comunalitati si corelograma

def corelograma(x,min=-1,max=1,titlu="Corelograma"):
    fig = plt.figure(titlu, figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    ax=sb.heatmap(data=x,vmin=min,vmax=max,cmap="bwr",annot=True,ax=ax)
    for i in range(len(x)):
        ax.set_xticklabels(labels=x.columns,ha="right",rotation=30)
corelograma(df_alpha,-1,1,"Corelo Varianta")
corelograma(df_scoruri,-1,1,"Coreloa scoruri")

#Componente de corelatie

rxc=np.corrcoef(x,c,rowvar=False)[:len(alpha),len(alpha):]
df_rxc=pd.DataFrame(data=rxc,index=variabile,columns=["C"+str(i+1)for i in range(len(alpha))])
df_rxc.to_csv("rxc.csv")
corelograma(df_rxc)
plot_corelatii(df_rxc,"C1","C2")

#Comunalitati
comunalitati=np.cumsum(rxc*rxc,axis=1)
df_comunalitati=pd.DataFrame(data=comunalitati,index=variabile,columns=["C"+str(i+1)for i in range(len(alpha))])
df_comunalitati.to_csv("comunalitati.csv")
corelograma(df_comunalitati,titlu="Corelo Comunalitati")

plt.show()










