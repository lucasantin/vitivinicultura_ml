#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import boto3
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.setLevel(logging.ERROR)

# Configurações do S3
bucket_name = 'bucketproject-vinicultura-ml'
prefix = input("Digite o caminho da pasta no S3 onde constam os dados: ")

# Inicializa o cliente S3
s3 = boto3.client('s3')

# Lista os objetos na pasta especificada
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Verifica se existem arquivos
if 'Contents' in response:
    dfs = []  # Lista para armazenar os DataFrames

    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('.csv'):  # Verifica se é um arquivo CSV
            # Lê o conteúdo do arquivo CSV
            obj_response = s3.get_object(Bucket=bucket_name, Key=key)
            csv_content = obj_response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))  # Cria DataFrame a partir do conteúdo CSV
            
            # Extrai o ano do nome do arquivo
            ano = key[-8:-4]  # Pega os últimos 4 caracteres antes de .csv
            df['Ano'] = ano  # Adiciona a coluna 'Ano' ao DataFrame
            
            dfs.append(df)  # Adiciona o DataFrame à lista

    # Consolida todos os DataFrames em um único DataFrame
    consolidated_df = pd.concat(dfs, ignore_index=True)

    # Remover duplicatas
    consolidated_df = consolidated_df.drop_duplicates()

    # Remover linhas com valores ausentes
    consolidated_df = consolidated_df.dropna()

    # Identificar variáveis com base no caminho
    if "exportacao" in prefix:
        categorical_cols = ['Países']
        numeric_cols = ['Quantidade (Kg)', 'Valor (US$)', 'Ano']
    elif "importacao" in prefix:
        categorical_cols = ['Países']
        numeric_cols = ['Quantidade (Kg)', 'Valor (US$)', 'Ano']
    elif "comercializacao" in prefix:
        categorical_cols = ['Produto']
        numeric_cols = ['Quantidade (L.)', 'Ano']
    elif "producao" in prefix:
        categorical_cols = ['Produto']
        numeric_cols = ['Quantidade (L.)', 'Ano']
    elif "processamento" in prefix:
        categorical_cols = ['Cultivar']
        numeric_cols = ['Quantidade (Kg)', 'Ano']

    print("Categóricas:", categorical_cols)
    print("Numéricas:", numeric_cols)
    
    # Converter colunas numéricas para o tipo correto
    for col in numeric_cols:
        consolidated_df[col] = pd.to_numeric(consolidated_df[col], errors='coerce')

    # Criar a coluna de data contendo apenas o ano como string
    consolidated_df['data'] = consolidated_df['Ano'].astype(str)

    # Definir a coluna de data como índice, convertendo para datetime apenas com o ano
    consolidated_df.set_index(pd.to_datetime(consolidated_df['data'], format='%Y'), inplace=True)


# In[ ]:


# Analisar o crescimento do valor unitário nos últimos anos
if "exportacao" in prefix or "importacao" in prefix:
    # Calcular o valor unitário
    consolidated_df['Valor Unitário (US$/Kg)'] = consolidated_df['Valor (US$)'] / consolidated_df['Quantidade (Kg)']
    growth_df = consolidated_df.groupby(['Ano', 'Países'])['Valor Unitário (US$/Kg)'].mean().reset_index()
    growth_df['Valor Unitário (US$/Kg)'] = pd.to_numeric(growth_df['Valor Unitário (US$/Kg)'], errors='coerce')

    # Identificar os 3 países com maior crescimento
    latest_year = growth_df['Ano'].max()
    previous_year = latest_year - 1

    growth_df_latest = growth_df[growth_df['Ano'] == latest_year]
    growth_df_previous = growth_df[growth_df['Ano'] == previous_year]

    merged_growth = growth_df_latest.merge(growth_df_previous, on='Países', suffixes=('_latest', '_previous'))
    merged_growth['Crescimento'] = merged_growth['Valor Unitário (US$/Kg)_latest'] - merged_growth['Valor Unitário (US$/Kg)_previous']

    # Após a identificação dos 5 países com maior crescimento do valor unitário
    top_countries = merged_growth.nlargest(5, 'Crescimento')[['Países', 'Crescimento']]
    print("Top 5 países com maior crescimento do valor unitário:")
    print(top_countries)

    # Calcular o crescimento previsto para o próximo ano
    # Considerando a previsão como o valor unitário mais recente + crescimento
    for index, row in top_countries.iterrows():
        latest_value = growth_df_latest[growth_df_latest['Países'] == row['Países']]['Valor Unitário (US$/Kg)'].values[0]
        predicted_growth = latest_value + row['Crescimento']
        top_countries.at[index, 'Crescimento Previsto'] = predicted_growth

    # Gráfico de barras dos 5 países com maior crescimento
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(top_countries))

    # Barras para crescimento atual
    plt.bar(x, top_countries['Crescimento'], width=bar_width, label='Crescimento Atual', color='skyblue')

    # Barras para crescimento previsto
    plt.bar([p + bar_width for p in x], top_countries['Crescimento Previsto'], width=bar_width, label='Crescimento Previsto', color='orange')

    # Configurações do gráfico
    plt.title('Top 5 Países com Maior Crescimento do Valor Unitário (US$/Kg)')
    plt.xlabel('Países')
    plt.ylabel('Crescimento (US$/Kg)')
    plt.xticks([p + bar_width / 2 for p in x], top_countries['Países'], rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()



# In[ ]:


# Exibir os dados consolidados
print(consolidated_df.head())


# In[ ]:


# Se o prefixo for "comercializacao" ou "producao"
if prefix in ["comercializacao", "producao"]:
    # Remover linhas onde a coluna 'Produto' contém a string 'Outros'
    if 'Produto' in consolidated_df.columns:
        consolidated_df = consolidated_df[~consolidated_df['Produto'].str.contains('Outros', na=False)]
    # Agregar os dados
    consolidated_df.replace('-', 0, inplace=True)
    consolidated_df = consolidated_df.groupby(['Ano', 'Produto'], as_index=False)['Quantidade (L.)'].sum()

    # Criar um DataFrame para Prophet
    results = []
    for produto in consolidated_df['Produto'].unique():
        df = consolidated_df[consolidated_df['Produto'] == produto][['Ano', 'Quantidade (L.)', 'Produto']]
        df.rename(columns={'Ano': 'ds', 'Quantidade (L.)': 'y'}, inplace=True)

        # Garantir que a coluna 'ds' seja datetime
        df['ds'] = pd.to_datetime(df['ds'].astype(str), format='%Y')
        
        model = Prophet()
        model.fit(df)
        
        future = model.make_future_dataframe(periods=5, freq='Y')  # Prever 5 anos
        forecast = model.predict(future)
        forecast['Produto'] = produto
        results.append(forecast[['ds', 'yhat', 'Produto']])

    # Consolidar resultados
    final_forecast = pd.concat(results)
    print(final_forecast.head())

    # Identificar os 5 produtos com maior crescimento
    top_products = (final_forecast
                 .groupby('Produto')['yhat']
                 .apply(lambda x: x.tail(10).mean())  # Calcule a média dos últimos 5 valores
                 .nlargest(5)  # Selecione os 5 produtos com maior média
                 .index)

    # Filtrar as previsões para os produtos principais
    top_forecast = final_forecast[final_forecast['Produto'].isin(top_products)]

    # Plotar os resultados
    plt.figure(figsize=(12, 6))
    for produto in top_products:
        product_data = top_forecast[top_forecast['Produto'] == produto]
        plt.plot(product_data['ds'], product_data['yhat'], label=produto)

    plt.title(f'Previsão de Crescimento dos Principais Produtos de Uva - {prefix}')
    plt.xlabel('Ano')
    plt.ylabel('Quantidade Prevista')
    plt.legend()
    plt.grid()
    plt.show()

