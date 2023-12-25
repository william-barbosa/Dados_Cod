################################################################################
## Rotina criada por William Barbosa para o download dos dados do censo       ##
## agropecuário 2017/2018 atualizados                                         ##
################################################################################
import pandas as pd
import numpy as np
import sidrapy
import time
import requests

# Funçã anônima para baixar os dados a partir do api do sidra
def baixa_dados(url):
  requisicao = requests.get(url,json=True)               # Baixar os dados
  requisicao = pd.DataFrame.from_dict(requisicao.json()) # Converte para data.frame
  requisicao.columns = requisicao.iloc[0]                # Renomeando as colunas
  return requisicao.iloc[1:, :]                          # Retorna o dataset

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



