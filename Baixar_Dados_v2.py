################################################################################
## Rotina criada por William Barbosa para o download dos dados do censo       ##
## agropecuário 2017/2018 atualizados                                         ##
################################################################################
import pandas as pd
import numpy as np
import sidrapy
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
        .query("Ano==2017")



cerrado.head(15)


## Realizar o join entre as bases de desflorestamento e demais variáveis

# Dados do censo

vbp17 = (pd.read_csv('./Dados_V2/VBP17.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'VBP17'})
  [['geocode', 'VBP17']])

vbp06 = (pd.read_csv('./Dados_V2/VBP06.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'VBP06'})
          [['geocode', 'VBP06']])

agrotoxico = (pd.read_csv('./Dados_V2/agrotoxico.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'AGROTOXICO'})
          [['geocode', 'AGROTOXICO']])

irrigacao = (pd.read_csv('./Dados_V2/area_irrigada_hect.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'IRRIGACAO'})
          [['geocode', 'IRRIGACAO']])

area_agropec = (pd.read_csv('./Dados_V2/areaHec_agropec.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'AREA_AGROPEC'})
          [['geocode', 'AREA_AGROPEC']])

assist_tec = (pd.read_csv('./Dados_V2/assist_tec.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'ASSIST_TEC'})
          [['geocode', 'ASSIST_TEC']])

bens_equip = (pd.read_csv('./Dados_V2/bens_equip.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'BENS_EQUIP'})
          [['geocode', 'BENS_EQUIP']])

correcao_solo = (pd.read_csv('./Dados_V2/correcao_solo.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'CORRECAO_SOLO'})
          [['geocode', 'CORRECAO_SOLO']])

credito = (pd.read_csv('./Dados_V2/credito.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'CREDITO'})\
    .assign(
      CREDITO = lambda d: d['CREDITO'].str.replace("-", '0', regex=True).astype(float),
      # CREDITO = lambda d: d['CREDITO'].replace("-", np.nan, regex=True).replace("", np.nan).astype(float)
    )\
      .groupby('geocode',as_index=False).agg(
        CREDITO = ('CREDITO', 'sum'))
      [['geocode', 'CREDITO']])

trabalho = (pd.read_csv('./Dados_V2/trabalho.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'TRABALHO'})
          [['geocode', 'TRABALHO']])

area_municipal = (pd.read_csv('./Dados_V2/area_municipal.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'AREA_MUNI'})
          [['geocode', 'AREA_MUNI']])

despesas_tot = (pd.read_csv('./Dados_V2/despesas_totais.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'DESP_TOT'})
          [['geocode', 'DESP_TOT']])


#CONFERIR AQUI E SOMAR AS LINHAS QUE ESTÃO A MAIS
despesas_novas_pastagens = (pd.read_csv('./Dados_V2/despesas_novas_pastagens.csv',sep=',',decimal='.')\
  .rename(columns={'Município (Código)':'geocode','Valor':'DESP_PAST'})
          [['geocode', 'DESP_PAST']])


lista_df = [vbp17, vbp06, agrotoxico, irrigacao, area_agropec, assist_tec, bens_equip]#,
            # correcao_solo, credito, trabalho, area_municipal, despesas_tot, despesas_novas_pastagens]

dados_final = reduce(left_join, lista_df)

dados_final = pd.merge(cerrado, VBP06.filter(items=['geocode','VBP06']), how='left', on='geocode')

dados_final = pd.merge(cerrado, vbp17.filter(items=['geocode','VBP17']), how='left', on='geocode')


UU = lista_df[1].shape(0)

# Verifica o total de linhas que cada um dos df possui antes de realizar o join.
[len(df) for df in lista_df]


cerrado.to_csv('./Dados_V2/teste.csv',sep=',',decimal='.',encoding='UTF-8',index=False)


resultado = []
for coluna in range(0,len(lista_df)):
  res = lista_df[coluna].shape(0)
  resultado.append(res)

len(resultado[3])

vbp['Tipo de produção (Código)'].unique()


###########



Cerrado.assign(sepal_length_ranges = Cerrado['2015'].apply('2015') )

Cerrado.assign( teste = Cerrado[2015] / Cerrado[2016])



despesas.query("`Município (Código)` == '1100015'")
Cerrado.query("geocode == 1100015")

