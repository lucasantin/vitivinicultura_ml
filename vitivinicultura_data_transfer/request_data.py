from app import create_app
import requests
import pandas as pd
import boto3
import os

app = create_app()

def call_api_and_upload(base_url, tipo, anos):
    s3 = boto3.client('s3')
    bucket_name = 'bucketproject-vinicultura-ml'
    
    for ano in anos:
        api_url = f"{base_url}/{tipo}/{ano}"
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            csv_file_path = f'extracao_{tipo}_{ano}.csv'
            df = pd.DataFrame(data["rows"], columns=data["headers"])
            df.to_csv(csv_file_path, index=False)

            s3_file_path = f'{tipo}/{csv_file_path}'

            try:
                s3.upload_file(csv_file_path, bucket_name, s3_file_path)
                print(f"Arquivo {csv_file_path} enviado com sucesso!")
                
                # Deletar o arquivo local após o upload
                os.remove(csv_file_path)
                
            except Exception as e:
                print(f"Erro ao enviar o arquivo {csv_file_path}: {e}")
        else:
            print(f"Erro ao chamar a API para o ano {ano}: {response.status_code} - {response.text}")

if __name__ == "__main__":
    base_url = input("Por favor, insira a base da URL: ")
    tipo = input("Por favor, insira o tipo (producao, processamento, importacao, exportacao, comercializacao): ")
    ano_inicial = int(input("Por favor, insira o ano inicial: "))
    ano_final = int(input("Por favor, insira o ano final: "))

    anos = range(ano_inicial, ano_final + 1)
    call_api_and_upload(base_url, tipo, anos)
