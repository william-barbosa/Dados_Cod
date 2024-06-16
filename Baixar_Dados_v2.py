################################################################################
## Rotina criada por William Barbosa para o download dos dados do censo       ##
## agropecuário 2017/2018 atualizados                                         ##
################################################################################
import pandas as pd
import numpy as np
#import sidrapy
import time
import requests
from functools import reduce

np.set_printoptions(edgeitems=30, linewidth = 1000)
# pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)


# Função anônima para baixar os dados a partir do api do sidra
def baixa_dados(url):
  requisicao = requests.get(url,json=True)               # Baixar os dados
  requisicao = pd.DataFrame.from_dict(requisicao.json()) # Converte para data.frame
  requisicao.columns = requisicao.iloc[0]                # Renomeando as colunas
  return requisicao.iloc[1:, :]                          # Retorna o dataset

# Função utilizada para agregação
def categorizacao_uso_terra(x):
    if x in ['2. Non Forest Natural Formation','4. Non Vegetated Area','5. Water','6. Non Observed']:
        return 'Outros'
    else:
        return x


# Função para realizar um left join entre dois DataFrames com base na coluna 'id'
def left_join(df_left, df_right):
    return pd.merge(df_left, df_right, on='geocode', how='left')





# Importando os dados com os rótulos dos municípios pertencentes ao Cerrado (verificar onde realizei essa marcação)
UF = pd.read_csv('./Dados_V2/Municipios_Cerrado.csv')

# Criando o vetor de coluna com o código da UF
UF['UF'] = UF['CD_Muni'].astype(str).str[:2]

UF.head(8)

# Tabela de crédito 6895
start = time.time()
credito = baixa_dados('https://apisidra.ibge.gov.br/values/t/6895/n6/all/v/allxp/p/all/c829/46302/c12542/115947/c218/46502/c12517/113601/c12544/allxt/c220/110085/d/2/f/c')
credito.to_csv('./Dados_V2/credito.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start


# Tabela 6897 - VBP 2017
start = time.time()
VBP17 = baixa_dados("https://apisidra.ibge.gov.br/values/t/6897/n6/all/v/1999/p/all/c829/46302/c12547/114017/c218/46502/c12517/113601/d/2/f/c")
VBP17.to_csv('./Dados_V2/VBP17.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Tabela 1118 - VBP 2006
start = time.time()
VBP06 = baixa_dados('https://apisidra.ibge.gov.br/values/t/1118/n6/all/v/1999/p/all/c12547/114017/c12896/0/d/2/f/c')
VBP06.to_csv('./Dados_V2/VBP06.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start


# Tabela 6962 - Correção do solo
start = time.time()
correcao_solo = baixa_dados('https://apisidra.ibge.gov.br/values/t/6962/n6/all/v/183/p/all/c836/46531/c12549/46554/c798/47179/c220/110085/d/2/f/c')
correcao_solo.to_csv('./Dados_V2/correcao_solo.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Tabela 6874 - Total de bens e equipamentos
start = time.time()
bens_equip = baixa_dados('https://apisidra.ibge.gov.br/values/t/6874/n6/all/v/9572/p/all/c829/46302/c796/46567/c12603/45927/c220/110085/d/2/f/c')
bens_equip.to_csv('./Dados_V2/bens_equip.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Tabela 6888 - Trabalho
start = time.time()
trabalho = baixa_dados('https://apisidra.ibge.gov.br/values/t/6888/n6/all/v/185/p/all/c829/46302/c12578/112967/c12573/45929/c218/46502/c12517/113601/d/2/f/c')
trabalho.to_csv('./Dados_V2/trabalho.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Tabela 6879 - Área ocupada pela agropecuária
start = time.time()
areaHec_agropec = baixa_dados('https://apisidra.ibge.gov.br/values/t/6879/n6/all/v/184/p/all/c829/46302/c12517/113601/c12567/41151/c12894/46569/d/2/f/c')
areaHec_agropec.to_csv('./Dados_V2/areaHec_agropec.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Tabela 6879 - Assistência técnica
start = time.time()
assist_tec = baixa_dados('https://apisidra.ibge.gov.br/values/t/6879/n6/all/v/183/p/all/c829/46302/c12517/113601/c12567/113111/c12894/46569/d/2/f/c')
assist_tec.to_csv('./Dados_V2/assist_tec.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Tabela 6857 - Área irrigada
start = time.time()
area_irrigada_hect = baixa_dados('https://apisidra.ibge.gov.br/values/t/6857/n6/all/v/2373/p/all/c829/46302/c12604/118477/c12564/41145/c12771/45951/c309/10969/d/2/f/c')
area_irrigada_hect.to_csv('./Dados_V2/area_irrigada_hect.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Tabela 6852 - Agrotóxico
start = time.time()
agrotoxico = baixa_dados('https://apisidra.ibge.gov.br/values/t/6852/n6/all/v/all/p/all/c829/46302/c12521/111611/c12567/41151/c837/46544/c12603/45927/c220/110085/d/2/f/c')
agrotoxico.to_csv('./Dados_V2/agrotoxico.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Tabela 6899 - Despesas
start = time.time()
despesas = baixa_dados('https://apisidra.ibge.gov.br/values/t/6899/n6/all/v/1996/p/all/c829/46302/c210/113946/c218/46502/c12517/113601/d/2/f/c')
despesas.to_csv('./Dados_V2/despesas_totais.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Tabela 6899 - Despesas com novas culturas permanentes e silvicultura; formação de pastagens
start = time.time()
despesas_novas_pastagens = baixa_dados('https://apisidra.ibge.gov.br/values/t/6899/n6/all/v/1996/p/all/c829/46302/c210/45957,45958/c218/46502/c12517/113601/d/2/f/c')
despesas_novas_pastagens.to_csv('./Dados_V2/despesas_novas_pastagens.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Tabela 1301 - Área municipal (2010)
start = time.time()
area_municipal = baixa_dados('https://apisidra.ibge.gov.br/values/t/1301/n6/all/v/615/p/all/d/2/f/c')
area_municipal.to_csv('./Dados_V2/area_municipal.csv',sep=',',decimal='.',encoding='UTF-8',index=False)
time.time() - start

# Manipulação das informações de cobertura florestal
start = time.time()
Cerrado = pd.read_excel('./Dados_V2/tabela_geral_mapbiomas_col8_biomas_municipios.xlsx',sheet_name='COBERTURA_COL8.0')
time.time() - start
# 1986, 1987, 1988, 1989, 1990, 1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022

cerrado  =  Cerrado[['geocode', 'biome','level_1', 1985, 2017]].query("biome == 'Cerrado'") \
  .assign( Level_1 = Cerrado['level_1'].apply(categorizacao_uso_terra) )


cerrado = cerrado[['geocode', 'Level_1', 1985, 2017]].groupby(['geocode','Level_1'],as_index=False)\
  .agg(
    a1985 = (1985, 'sum'),
    a2017 = (2017, 'sum'),
  )\
    .melt(id_vars=['geocode', 'Level_1'],value_name='Area',var_name='Ano')\
    .pivot(index= ['geocode','Ano'], columns=['Level_1'], values="Area")\
    .reset_index()\
    .assign(Total = lambda d: d['1. Forest'] + d['3. Farming'] + d['Outros'],
            Percent = lambda d: d['1. Forest'] / d['Total'],
            Ano = lambda d: d['Ano'].str.replace("a",'', regex=True).astype(int)
            )\
    .groupby('geocode').apply(
      lambda df: df.assign(
        tx_desf=lambda x: ((x['1. Forest'] / x['1. Forest'].shift(1)) - 1)\
          .apply(lambda y: 0 if y > 0 else abs(y))
          )
      ).reset_index(drop=True)\
        .query("Ano==2017").assign(
      geocode = lambda d: d['geocode'].astype(str)
    ).rename(columns={'geocode':'CD_GEOCMU'})



cerrado.head(15)


cerrado.shape




## Realizar o join entre as bases de desflorestamento e demais variáveis

# Dados do censo

vbp17 = (pd.read_csv('./Dados_V2/VBP17.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'VBP17'})\
    .assign(
      VBP17 = lambda d: d['VBP17'].str.replace("-|X", '0', regex=True).astype(float)
      )[['geocode', 'VBP17']])

vbp06 = (pd.read_csv('./Dados_V2/VBP06.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'VBP06'})\
    .assign(
      VBP06 = lambda d: d['VBP06'].str.replace("-|X", '0', regex=True).astype(float)
    )[['geocode', 'VBP06']])

agrotoxico = (pd.read_csv('./Dados_V2/agrotoxico.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'AGROTOXICO'})\
    .assign(
      AGROTOXICO = lambda d: d['AGROTOXICO'].str.replace("-|X", '0', regex=True).astype(float)
    )[['geocode', 'AGROTOXICO']])

irrigacao = (pd.read_csv('./Dados_V2/area_irrigada_hect.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'IRRIGACAO'})\
    .assign(
      IRRIGACAO = lambda d: d['IRRIGACAO'].str.replace("-|X", '0', regex=True).astype(float)
    )[['geocode', 'IRRIGACAO']])

area_agropec = (pd.read_csv('./Dados_V2/areaHec_agropec.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'AREA_AGROPEC'})\
    .assign(
      AREA_AGROPEC = lambda d: d['AREA_AGROPEC'].str.replace("-|X", '0', regex=True).astype(float)
    )[['geocode', 'AREA_AGROPEC']])

assist_tec = (pd.read_csv('./Dados_V2/assist_tec.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'ASSIST_TEC'})\
    .assign(
      ASSIST_TEC = lambda d: d['ASSIST_TEC'].str.replace("-|X", '0', regex=True).astype(float)
    )[['geocode', 'ASSIST_TEC']])

bens_equip = (pd.read_csv('./Dados_V2/bens_equip.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'BENS_EQUIP'})\
    .assign(
      BENS_EQUIP = lambda d: d['BENS_EQUIP'].str.replace("-|X", '0', regex=True).astype(float)
    )[['geocode', 'BENS_EQUIP']])

correcao_solo = (pd.read_csv('./Dados_V2/correcao_solo.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'CORRECAO_SOLO'})\
    .assign(
       CORRECAO_SOLO = lambda d: d['CORRECAO_SOLO'].str.replace("-|X", '0', regex=True).astype(float)
    )[['geocode', 'CORRECAO_SOLO']])

credito = (pd.read_csv('./Dados_V2/credito.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'CREDITO'})\
    .assign(
      CREDITO = lambda d: d['CREDITO'].str.replace("-|X", '0', regex=True).astype(float),
      # CREDITO = lambda d: d['CREDITO'].replace("-", np.nan, regex=True).replace("", np.nan).astype(float)
    )\
      .groupby('geocode',as_index=False).agg(
        CREDITO = ('CREDITO', 'sum'))
      [['geocode', 'CREDITO']])

trabalho = (pd.read_csv('./Dados_V2/trabalho.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'TRABALHO'})\
    .assign(
      TRABALHO = lambda d: d['TRABALHO'].str.replace("-|X", '0', regex=True).astype(float)
    )[['geocode', 'TRABALHO']])

area_municipal = (pd.read_csv('./Dados_V2/area_municipal.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'AREA_MUNI'})#\
    # .assign(
    #   AREA_MUNI = lambda d: d['AREA_MUNI'].str.replace("-|X", '0', regex=True).astype(float)
    # )
    [['geocode', 'AREA_MUNI']])

despesas_tot = (pd.read_csv('./Dados_V2/despesas_totais.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'DESP_TOT'})\
    .assign(
      DESP_TOT = lambda d: d['DESP_TOT'].str.replace("-|X", '0', regex=True).astype(float)
    )[['geocode', 'DESP_TOT']])

despesas_novas_pastagens = (pd.read_csv('./Dados_V2/despesas_novas_pastagens.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'DESP_PAST'})\
    .assign(
      DESP_PAST = lambda d: d['DESP_PAST'].str.replace("-|X", '0', regex=True).astype(float)
    )\
    .groupby('geocode',as_index=False).agg(
        DESP_PAST = ('DESP_PAST', 'sum'))
          [['geocode', 'DESP_PAST']])


lista_df = [vbp17, vbp06, agrotoxico, irrigacao, area_agropec, assist_tec, bens_equip, correcao_solo, credito, trabalho, area_municipal, despesas_tot, despesas_novas_pastagens]

dados_final = reduce(left_join, lista_df)\
  .rename(columns={'geocode':'CD_GEOCMU'})\
    .assign(
      # CD_GEOCMU = lambda d: d['CD_GEOCMU'].astype(chr)
      CD_GEOCMU = lambda d: d['CD_GEOCMU'].astype(str)
    )

len(dados_final)
dados_final.shape

# Verifica o total de linhas que cada um dos df possui antes de realizar o join.
[len(df) for df in lista_df]


cerrado.to_csv('./Dados_V2/teste.csv',sep=',',decimal='.',encoding='UTF-8',index=False)


###########

# def verificar_faltantes(x, y):
#   X = x.groupby('geocode').size().reset_index(name='counts_X').sort_values(by='counts_X', ascending=False)
#   Y =y.groupby('geocode').size().reset_index(name='counts_Y').sort_values(by='counts_Y', ascending=False)

#   # Realizar o left join
#   final = X.merge(Y, on='geocode', how='left')

#   # Filtrar os casos em que 'counts_Y' está NaN (não há correspondência)
#   faltantes = final[final['counts_Y'].isna()]

#   return faltantes


# verificar_faltantes(area_municipal,despesas_novas_pastagens)

# # Verificar o total de municipior pertencentes ao cerrado

# cerrado[['CD_GEOCMU']].nunique()

# area_municipal[['geocode']].nunique()


# dados_final.query("CD_GEOCMU == '3538907'")
# Cerrado.query("geocode == 3538907")
# shp.query("CD_GEOCMU == 3500600")
# OS dois municipios faltantes não estão no shp!
# '3500600', '3538907'

################################################################################
###                     Tratamento de dados espaciais                       ####
################################################################################

import geopandas as gpd
import matplotlib as plt
dados_final['CD_GEOCMU'].dtype


# Caminho para o seu arquivo .shp
shp_file_path = 'C:/Users/William Barbosa/Documents/Desfl_Cerrado/Dados_V2/SHP/br_municipios/BRMUE250GC_SIR.shp'

# Importando o arquivo .shp
shp = gpd.read_file(shp_file_path)

# Realiza o join entre o shp e a base de dados
# shp_final = shp_final[~shp_final['CD_GEOCMU'].isin([3518701, 3520400,2605459])]

shp_final = shp.merge(dados_final, on='CD_GEOCMU')\
  .query("CD_GEOCMU not in [3518701, 3520400, 2605459]")

shp_final = shp_final.merge(cerrado,on='CD_GEOCMU')

# Filtrando somente os municípios do Cerrado
cerrado_mun = cerrado[['CD_GEOCMU']].assign(
  CD_GEOCMU = lambda d: d['CD_GEOCMU'].astype(str)
  )['CD_GEOCMU'].unique() # Lista de municípios pertencentes ao Cerrado

# Apenas dois municípios não estão na lista final
set(shp_final['CD_GEOCMU']).symmetric_difference(cerrado_mun)

# SHP final
shp_final1 = shp_final[shp_final['CD_GEOCMU'].isin(cerrado_mun)]

shp_final1 = shp_final1.drop('Ano',axis=1)

# Salvando o resultado
shp_final1.to_file('./Dados_V2/SHP/mapa_regressao_vf.shp')

len(np.unique(cerrado[['CD_GEOCMU']]))

for col in shp_final.columns:
    print(col + ": " + str(shp_final[col].dtype))

len(shp_final[['CD_GEOCMU']])

[len(df) for df in lista_df]


# Verificar e retornar o total de NA's em colunas que possuem NA's
for col in shp_final.columns:
    na_count = shp_final[col].isna().sum()
    if na_count > 0:
        print(f"{col}: {na_count} NA's")


shp_final.query('VBP06 != VBP06') # 2 casos = NA
shp_final.query('IRRIGACAO != IRRIGACAO') # 2 casos = NA
shp_final.query('CREDITO != CREDITO') # 2 casos = NA
shp_final.query('tx_desf != tx_desf') # 2 casos = NA

import matplotlib.pyplot as plt

def count_na_per_row(df):
    """
    Conta a quantidade de valores NA em cada linha de um DataFrame.

    Parâmetros:
    df (DataFrame): O DataFrame a ser verificado.

    Retorna:
    Series: Uma série com a quantidade de valores NA em cada linha.
    """
    #return df.apply(lambda row: row.isna().sum(), axis=1)
    return df.apply(lambda row: row.isna().all(), axis=1)


def plt_na(shp,var):
  dados = shp.assign(
      varplot = lambda d: d[var].isna()
    )
  color_map = {True: 'red', False: '#0000CD'}
  dados['color'] = dados['varplot'].map(color_map)

  fig, ax = plt.subplots(1, 1, figsize=(10, 6))
  dados.plot(ax=ax, color=dados['color'], edgecolor='0.8', linewidth=0.2)

  legend_labels = {'Com NA': 'red', 'Com valor': '#0000CD'}
  handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
               for label, color in legend_labels.items()]
  ax.legend(handles=handles, loc='upper left', title='Legenda')

    # Mostrar o plot
  plt.show()

plt_na(shp_final,'IRRIGACAO')
plt_na(shp_final,'tx_desf')

## Fazer um loop para criar uma flag que indique NA para cada uma das variáveis e linhas

na_counts = count_na_per_row(shp_final)
shp_final['na_counts'] = na_counts

shp_final.query('na_counts != 0')[['NM_MUNICIP',]]


################################################################################
import geopandas as gpd
import matplotlib.pyplot as plt
import pysal as py
from splot.mapping import vba_choropleth as maps
from matplotlib import colors
import contextily
import spreg
import xlsxwriter
#from .sqlite import head_to_sql, start_sql
import os
import libpysal
from mlxtend.preprocessing import standardize
import numpy as np

shp_final.plot()
plt.show()

# # LISA
# tx_desf = shp_final['tx_desf']
# tx_desf_lisa = pygeoda.local_moran(queen_w, tx_desf)

# Início da modelagem
db = libpysal.io.open('./Dados_V2/SHP/mapa_regressao_vf.dbf','r')
shp_final = gpd.read_file('./Dados_V2/SHP/mapa_regressao_vf.shp')

db.header

# Matriz queen
queen_w = pygeoda.queen_weights(shp_final)
# Criando a matriz de peso espacial
w = libpysal.weights.Queen.from_shapefile('./Dados_V2/SHP/mapa_regressao_vf.shp')
w.transform = 'r' # Padroniza na linha, de forma que a soma será 1


shp_final['VBP17']
X = []
#X.append(db.by_col("IRRIGAC"))
X.append(shp_final['VBP17'])

x_names = ['VBP17']

# Criando as variáveis
vr_fl = shp_final["tx_desf"]
y = np.array(vr_fl)
y.shape = (len(vr_fl),1)


# Estima um OLS com os testes

# Normalizar a matriz
X0 = X#standardize(X)

ols = spreg.OLS(y,X0,w=w, name_x=x_names, name_y='des1785', name_w = 'Matriz Queen',name_ds='Dados_Desma',spat_diag=True)

print(ols.summary)