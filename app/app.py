import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from gower import gower_matrix
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform


@st.cache_data
def calcular_distancia_gower(x, vars_categoricas):
    return gower_matrix(x, cat_features=vars_categoricas)

@st.cache_data
def calcular_squareform(distancia_gower):
    return squareform(distancia_gower, force='tovector')

@st.cache_data
def calcular_linkage(squareform):
    return linkage(squareform, method='complete')


def main():

    st.set_page_config(page_title = 'Online Shoppers', layout="wide", initial_sidebar_state='expanded')

    cwd = os.getcwd()
    data_file = cwd + '/app/online_shoppers_intention.csv'
    df = pd.read_csv(data_file)


    st.write("""## Dados originais

    A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses, 
    para posteriormente estudarmos a relação entre o design da página e o perfil do cliente - "Será que clientes com comportamento de navegação 
    diferentes possuem propensão a compra diferente?
    """)
    st.write(df)



    # O valor da variável 'SpecialDay' está sendo alterado para o tipo boolean
    # considerando True como data festiva os registros cujo valor é maior que zero.

    df_copy = df.copy()
    df_copy['SpecialDay'] = np.where(df_copy['SpecialDay'] > 0, True, False)



    # Variáveis qualitativas. Variáveis que indicam a característica da data.
    vars_quali = [
        'Month',
        'Weekend',
        'SpecialDay'
    ]

    # Variáveis quantitativas. Variáveis que descrevem o padrão de navegação na sessão.
    vars_quanti = [
        'Administrative',
        'Administrative_Duration',
        'Informational',
        'Informational_Duration',
        'ProductRelated',
        'ProductRelated_Duration',
    ]

    # aplicando StandardScaler nas variáveis quantitativas
    df_quanti = pd.DataFrame(StandardScaler().fit_transform(df_copy[vars_quanti]), columns=vars_quanti)

    # criando os dummies para as variáveis qualitativas
    df_quali = pd.get_dummies(df_copy[vars_quali], columns=vars_quali)

    # criando o DataFrame padronizado
    X_pad = pd.concat([df_quanti, df_quali], axis=1)



    st.write("""## Dados com tratamento

    Aqui os dados quantitativos foram tratados com StandardScaler e os dados qualitativos transformados em dummies
    """)
    st.write(X_pad)




    # O código abaixo cria uma lista com o nome das variáveis qualitativas após 
    # a padronização do DataFrame usando o get_dummies

    vars_quali_names = []
    for col_name in X_pad.columns:
        for quali_name in vars_quali:
            if col_name.startswith(quali_name):
                vars_quali_names.append(col_name)
                break



    # definindo quais variáveis do DataFrame são categóricas
    vars_categoricas = [(x in vars_quali_names) for x in X_pad.columns]
    

    # calcula a matriz de distâncias utilizando a distância de Gower.
    distancia_gower = calcular_distancia_gower(X_pad, vars_categoricas)


    # ajustando o formato da matriz de distâncias
    gdv = calcular_squareform(distancia_gower)


    # ligação entre os pontos com base na matriz de distâncias
    Z = calcular_linkage(gdv)



    # marcando no Dataframe os grupos nos quais os registros foram classificados
    # considerando a formação de 3 grupos (coluna "grupo3") ou 4 grupos (coluna "grupo4")

    X_pad['grupo3'] = fcluster(Z, 3, criterion='maxclust')
    X_pad['grupo4'] = fcluster(Z, 4, criterion='maxclust')

    df_copy['grupo3'] = X_pad['grupo3']
    df_copy['grupo4'] = X_pad['grupo4']




    # nome das colunas dos meses que foram criadas com o get_dummies
    meses = [n for n in X_pad.columns if (n.startswith('Month_'))]

    # colunas com as variáveis referentes a quantidade de acessos
    df_grupos = df[['Administrative', 'Informational', 'ProductRelated']].copy()

    # adiciona a coluna referente a data festiva
    df_grupos['SpecialDay'] = df_copy['SpecialDay']

    # adiciona a classificação de 3 ou 4 grupos
    df_grupos['grupo3'] = df_copy['grupo3']
    df_grupos['grupo4'] = df_copy['grupo4']

    # cria um DataFrame final
    df_grupos = pd.concat([df_grupos, X_pad[meses]], axis=1)



    grupo_com_4 = df_grupos.drop(columns=['grupo3'], axis=1).groupby('grupo4').sum()

    st.write("""## Avaliação dos grupos

    Análise dos dados agrupados em 4 e 3 grupos
    """)


    st.write('#### 4 grupos')
    st.write(grupo_com_4)


    grupo_com_3 = df_grupos.drop(columns=['grupo4'], axis=1).groupby('grupo3').sum()
    st.write('#### 3 grupos')
    st.write(grupo_com_3)



    # agrupando por grupo é calculando a média, mediana, desvio padrão e valor máximo para BounceRates
    # calculando a quantidade de Revenue
    df['grupo'] = X_pad['grupo3']


    st.write("""## Avaliação dos grupos com Dendrograma    """)
    fig, axs = plt.subplots(1, 1, figsize=(7,4))
    dendrogram(Z, truncate_mode='level', p=10, show_leaf_counts=True, ax=axs, color_threshold=.24)
    st.pyplot(fig, use_container_width=False)


if __name__ == '__main__':
	main()