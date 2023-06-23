import pandas as pd
import factor_analyzer as fact
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

tabel_coduri=pd.read_csv("DateIN/Coduri_Localitati.csv",index_col=0)
tabel_voturi=pd.read_csv("DateIN/VotBUN.csv",index_col=0)

#Cerinta1
def vot_minim(linie):
    valori=linie[1:]
    index_min=np.argmin(valori)
    localitate=linie["Localitate"]
    #suma=np.sum(valori)
    data=[]
    data.append(localitate)
    data.append(valori.index[index_min])
    #data.append(suma)
    return pd.Series(data=data,index=["Localitate","Categorie"])

cerinta1=tabel_voturi.apply(func=vot_minim,axis=1)
cerinta1.to_csv("cerinta1.csv")

#Cerinta2

cerinta2=tabel_coduri.merge(tabel_voturi,left_index=True,right_index=True).groupby("Judet").mean(numeric_only=True)
cerinta2.to_csv("cerinta2.csv")

#Analiza Factoriala
variabile=list(tabel_voturi.columns[1:])
instante=list(tabel_voturi.index)


def nan_change(t):
    assert  isinstance(t,pd.DataFrame)
    for v in t.columns:
        if any(t[v].isna()):
            t[v].fillna(t[v].mean(),inplace=True)
        else:
            t[v].fillna((t[v].mode()[0]),inplace=True)
nan_change(tabel_voturi)
x=tabel_voturi[variabile].values
n,m=np.shape(x) #n->nr linii m->nrColoane
#Test bartlett
test_bartlett=fact.calculate_bartlett_sphericity(x)
print(test_bartlett)
if(test_bartlett[1]>0.1):
    print("Nu avem factori comuni")

#KMO
kmo=fact.calculate_bartlett_sphericity(x)
print(kmo)
if(kmo[1]<0.5):
    print("Nu avem factori comuni relevanti")

#Cream model
rotatie=""
if rotatie=="":
    model_fact=fact.FactorAnalyzer(n_factors=m,rotation=None)
else:
    model_fact = fact.FactorAnalyzer(n_factors=m, rotation=rotatie)

model_fact.fit(x)
#Scoruri
scoruri=model_fact.transform(x)
df_scoruri=pd.DataFrame(data=scoruri,index=variabile+["Total"],columns=["C"+str(i+1)for i in range (m)])
df_scoruri.to_csv("scoruri.csv")
#plot componente si plot corelatii
def plot_componente(x,var_x,var_y,titlu="Componente"):
    fig=plt.figure(titlu,figsize=(9,9))
    ax=fig.add_subplot(1,1,1)
    ax.set_title(titlu)
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.scatter(x[var_x],x[var_y])

    for i in range(len(x)):
        ax.text(x[var_x].iloc[i],x[var_y].iloc[i],x.index[i])


def plot_corelatii(x, var_x, var_y, titlu="Corelatii"):
    fig = plt.figure(titlu, figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    theta=np.arange(0,2*np.pi,0.01)
    ax.plot(np.cos(theta),np.sin(theta),c="r")
    ax.scatter(x[var_x], x[var_y])

    for i in range(len(x)):
        ax.text(x[var_x].iloc[i], x[var_y].iloc[i], x.index[i])


def corelograma(x, min=-1, max=1, titlu="Componente"):
    fig = plt.figure(titlu, figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    ax=sb.heatmap(data=x,vmin=min,vmax=max,cmap="bwr",annot=True,ax=ax)
    for i in range(len(x)):
        ax.set_xticklabels(labels=x.columns, ha="right",rotation=30)


plot_componente(df_scoruri,"C1","C2","Componente Scoruri C1 si C2")
plot_corelatii(df_scoruri,"C1","C2","Corelatii Scoruri C1 si C2")

#Varianta
alpha=model_fact.get_factor_variance()[0]
print(alpha)
df_alpha=pd.DataFrame(data={"Valoare":alpha},index=variabile)
df_alpha.to_csv("varianta.csv")
#Factori de corelatie
loadings=model_fact.loadings_
df_loadings=pd.DataFrame(data=loadings,index=variabile,columns=["C"+str(i+1)for i in range (len(alpha))])
df_loadings.to_csv("FactoriCorelatie.csv")
plot_corelatii(df_loadings,"C1","C2","Corelatii Factori")

#Comunalitati
comunalitati=model_fact.get_communalities()
df_comunalitati=pd.DataFrame(data={"Valoare":comunalitati},index=variabile)
df_comunalitati.to_csv("comunalitati.csv")
corelograma(df_comunalitati,-1,1,"Corelgorama Comunalitati")
plt.show()