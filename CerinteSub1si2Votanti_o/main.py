import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
import seaborn as sb
import matplotlib.pyplot as plt

tabel_coduri=pd.read_csv("DateIN/Coduri_Localitati.csv",index_col=0)
tabel_vot=pd.read_csv("DateIN/VotBUN.csv",index_col=0)

#Cerinta1
#Procente vot
variabile=list(tabel_vot.columns[2:])
#print(variabile)
#print(tabel_vot["Votanti_LP"])


def participare(linie,variabile):
    valori=linie[variabile].values
    nrVotanti=linie["Votanti_LP"]
    #nrvoturi=np.sum(valori) asta era daca faceam per total
    #print(nrvoturi)
    procent=valori*100/nrVotanti
    print(procent)
    data=list(procent)
    localitate=linie["Localitate"]
    data.insert(0,localitate)
    return pd.Series(data=data,index=["Localitate"]+variabile)

cerinta1=tabel_vot.apply(func=participare,variabile=variabile,axis=1)
cerinta1.to_csv("cerint1.csv")

#Cerinta2
def participareJudet(linie,variabile):
    valori=linie[variabile].values
    nrVotanti=linie["Votanti_LP"]
    #nrvoturi=np.sum(valori) asta era daca faceam per total
    #print(nrvoturi)
    procent=valori*100/nrVotanti
    print(procent)
    data=list(procent)

    return pd.Series(data=data,index=variabile)

merged=tabel_vot.merge(tabel_coduri,left_index=True,right_index=True).groupby("Judet").sum(numeric_only=True)
print(merged)
merged.to_csv("combinat.csv")
cerinta2=merged.apply(func=participareJudet,variabile=variabile,axis=1)
cerinta2.to_csv("cerinta2.csv")

#Analiza Canonica

variabileX=list(tabel_vot.columns[2:7])
print(variabileX)
variabileY=list(tabel_vot.columns[7:])
print(variabileY)

def nan_replace(t):
    assert isinstance(t,pd.DataFrame)
    for v in t.columns:
        if any(t[v].isna()):
            t[v].fillna(t[v].mean(),inplace=True)
        else:
            t[v].fillna(t[v].mode()[0],inplace=True)
nan_replace(tabel_vot)

x=tabel_vot[variabileX].values
y=tabel_vot[variabileY].values

n,p=np.shape(x)
q=np.shape(y)[1]
m=min(p,q)
etichete_z=["Z"+str(i+1)for i in range(m)]
etichete_u=["U"+str(i+1)for i in range(m)]

#Creare model
model_cca=CCA(n_components=m)
model_cca.fit(x,y)

#Calcul scoruri, radacini Z si U
z,u=model_cca.transform(x,y)
df_z=pd.DataFrame(data=z,index=tabel_vot.index,columns=etichete_z)
df_z.to_csv("z.csv")
df_u=pd.DataFrame(data=u,index=tabel_vot.index,columns=etichete_u)
df_u.to_csv("u.csv")
#Calcul corelatii canonice
r=np.diag(np.corrcoef(z,u,rowvar=False)[:m,m:])
df_r=pd.DataFrame(data=r,columns=["Valoare Corelatie"],index=["Radacina"+str(i+1)for i in range(m)])
df_r.to_csv("r.csv")

#Plot instante primele 2 radacini
def plot_instante(tabel_z,z1,z2,tabel_u,u1,u2,titlu="Plot instante"):
    fig=plt.figure(titlu,figsize=(9,9))
    ax=fig.add_subplot(1,1,1)
    ax.set_title(titlu)
    ax.set_xlabel(z1+"/"+z2)
    ax.set_ylabel(u1+"/"+u2)
    theta=np.arange(0,2*np.pi,0.01)
    ax.plot(np.cos(theta),np.sin(theta),c="red")
    ax.axhline(0)
    ax.axvline(0)
    ax.scatter(tabel_z[z1],tabel_z[z2],label="Spatiul X",c="green")
    ax.scatter(tabel_u[u1], tabel_u[u2], label="Spatiul Y", c="blue")
    for i in range(len(tabel_z)):
        ax.text(tabel_z[z1].iloc[i],tabel_z[z2].iloc[i],tabel_z.index[i])
    for i in range(len(tabel_u)):
        ax.text(tabel_u[u1].iloc[i],tabel_u[u2].iloc[i],tabel_u.index[i])
    ax.legend()
plot_instante(df_z,"Z1","Z2",df_u,"U1","U2")


#Extra
#corelograma
def corelgorama(x,min,max,titlu="Corelo"):
    fig = plt.figure(titlu, figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    sb.heatmap(data=x,vmin=min,vmax=max,cmap="bwr",annot=True,ax=ax)
    for i in range(len(x)):
        ax.set_xticklabels(labels=x.columns,ha="right",rotation=30)
corelgorama(df_z,-1,1,"Z corelgoram")

#Corelatii variabile observate
xloadings=model_cca.x_loadings_
df_loadings=pd.DataFrame(data=xloadings,index=variabileX,columns=etichete_z)
df_loadings.to_csv("xloadings.csv",index_label="Categorie")
yloadings=model_cca.y_loadings_
df_loadingsy=pd.DataFrame(data=yloadings,index=variabileY,columns=etichete_u)
df_loadingsy.to_csv("yloadings.csv",index_label="Categorie")
plt.show()






