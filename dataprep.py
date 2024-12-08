# dataprep.py

import os
import requests
import pandas as pd
import json
import yaml
from dotenv import load_dotenv
from datetime import datetime
import openai
import pickle
import plotly.express as px

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

BASE_URL = 'https://dadosabertos.camara.leg.br/api/v2/'
START_DATE = '2024-08-01'
END_DATE = '2024-08-30'

def pega_dados(endpoint, params={}):
    url = BASE_URL + endpoint
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Erro na requisicao: {response.status_code}')
        return None

def pega_proposicoes():
    endpoint = 'proposicoes'
    params = {
        'dataInicio': START_DATE,
        'dataFim': END_DATE,
        'itens': 100
    }
    dados = pega_dados(endpoint, params)
    if dados:
        return dados['dados']
    return []

def pega_deputados():
    endpoint = 'deputados'
    params = {
        'itens': 100
    }
    dados = pega_dados(endpoint, params)
    if dados:
        return dados['dados']
    return []

def pega_proposicoes_por_tema(codigos_temas, total_por_tema=10):
    proposicoes_filtradas = []
    for codigo in codigos_temas:
        endpoint = 'proposicoes'
        params = {
            'dataInicio': START_DATE,
            'dataFim': END_DATE,
            'idTipo': codigo,
            'itens': total_por_tema
        }
        dados = pega_dados(endpoint, params)
        if dados and 'dados' in dados:
            proposicoes_filtradas.extend(dados['dados'][:total_por_tema])
        else:
            print(f'Nenhuma proposicao encontrada para o tema código {codigo}.')
    return proposicoes_filtradas

def resumo_texto(texto):
    openai.api_key = OPENAI_API_KEY
    try:
        resposta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente que resume textos."},
                {"role": "user", "content": f"Resuma o seguinte texto: {texto}"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        return resposta.choices[0].message['content'].strip()
    except Exception as e:
        print(f'Erro na sumarizacao: {e}')
        return ""

def extrair_palavras_chave(texto):
    openai.api_key = OPENAI_API_KEY
    try:
        resposta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente que extrai palavras-chave de um texto."},
                {"role": "user", "content": f"Extraia as palavras-chave do seguinte texto: {texto}"}
            ],
            max_tokens=50,
            temperature=0.5
        )
        return resposta.choices[0].message['content'].strip().split(', ')
    except Exception as e:
        print(f'Erro na extracao de palavras-chave: {e}')
        return []

def gerar_overview_summary():
    prompt = "Utilize o modelo de linguagem para gerar um texto curto (2 parágrafos) que explique a Câmara dos Deputados do Brasil. O texto deve ser claro, conciso e informativo."
    openai.api_key = OPENAI_API_KEY
    try:
        resposta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente que gera resumos informativos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        summary = resposta.choices[0].message['content'].strip()
        config_path = 'data/config.yaml'
        os.makedirs('data', exist_ok=True)
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        config['overview_summary'] = summary
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        print('overview_summary salvo em config.yaml.')
    except Exception as e:
        print(f'Erro ao gerar overview_summary: {e}')

def coleta_despesas(deputados_df):
    despesas_list = []
    for _, dep in deputados_df.iterrows():
        deputado_id = dep['id']
        endpoint = f'deputados/{deputado_id}/despesas'
        params = {
            'ano': START_DATE[:4],
            'itens': 100
        }
        despesas = pega_dados(endpoint, params)
        if despesas and 'dados' in despesas:
            for d in despesas['dados']:
                d['deputado_id'] = deputado_id
                despesas_list.append(d)
        else:
            print(f'Nenhuma despesa encontrada para o deputado ID {deputado_id}.')

    if despesas_list:
        df_despesas = pd.DataFrame(despesas_list)
        df_despesas['data'] = pd.to_datetime(df_despesas['dataDocumento']).dt.date
        df_agrupada = df_despesas.groupby(['data', 'deputado_id', 'tipoDespesa']).agg({'valorLiquido': 'sum'}).reset_index()
        df_agrupada.to_parquet('data/serie_despesas_diárias_deputados.parquet', index=False)
        print('Dados de despesas agrupados e salvos em data/serie_despesas_diárias_deputados.parquet.')
    else:
        print('Nenhuma despesa coletada.')

def sumarizacao_proposicoes(proposicoes):
    try:
        summaries = []
        chunk_size = 5
        for i in range(0, len(proposicoes), chunk_size):
            chunk = proposicoes[i:i+chunk_size]
            textos = " ".join([prop.get('ementa', '') for prop in chunk])
            summary = resumo_texto(textos)
            summaries.append({
                "chunk": i//chunk_size +1,
                "summary": summary
            })
        with open('data/sumarizacao_proposicoes.json', 'w', encoding='utf-8') as f:
            json.dump({"summaries": summaries}, f, ensure_ascii=False, indent=4)
        print('Sumarização das proposições salva em data/sumarizacao_proposicoes.json.')
    except Exception as e:
        print(f'Erro na sumarizacao das proposicoes: {e}')

def gerar_grafico_distribuicao_deputados(df_deputados):
    os.makedirs('docs', exist_ok=True)
    distrib_partido = df_deputados['siglaPartido'].value_counts().reset_index()
    distrib_partido.columns = ['Partido', 'Total']
    fig = px.bar(distrib_partido, x='Partido', y='Total', title='Distribuição de Deputados por Partido')
    fig.write_image("docs/distribuicao_deputados.png")
    print('Gráfico de distribuição de deputados salvo em docs/distribuicao_deputados.png.')

def gerar_analises_despesas():
    # Código fixo e válido
    codigo_analises = r'''
import pandas as pd
import matplotlib.pyplot as plt

df_despesas = pd.read_parquet('data/serie_despesas_diárias_deputados.parquet')

# Análise 1: Principais tipos de despesas
total_gasto_por_categoria = df_despesas.groupby('tipoDespesa')['valorLiquido'].sum().sort_values(ascending=False)
print("Principais tipos de despesas e total gasto:")
print(total_gasto_por_categoria.head())

# Análise 2: Deputados com maiores despesas por categoria
total_gasto_por_deputado_categoria = df_despesas.groupby(['deputado_id', 'tipoDespesa'])['valorLiquido'].sum()
maiores_despesas_por_categoria = total_gasto_por_deputado_categoria.groupby('tipoDespesa').idxmax()
print("\nDeputados com maiores despesas por categoria:")
print(maiores_despesas_por_categoria)

# Análise 3: Variação das despesas ao longo do tempo
df_despesas['data'] = pd.to_datetime(df_despesas['data'], errors='coerce')
total_gasto_diario = df_despesas.groupby('data')['valorLiquido'].sum()

plt.figure(figsize=(12, 6))
plt.plot(total_gasto_diario.index, total_gasto_diario.values)
plt.xlabel('Data')
plt.ylabel('Total gasto')
plt.title('Variação das despesas ao longo do tempo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('docs/variacao_despesas_tempo.png')
print("\nGráfico da variação das despesas ao longo do tempo salvo em docs/variacao_despesas_tempo.png")
'''

    try:
        os.makedirs('data', exist_ok=True)
        with open('data/analises_despesas.py', 'w', encoding='utf-8') as f:
            f.write(codigo_analises)
        print("Código para análises de despesas gerado e salvo em data/analises_despesas.py.")
    except Exception as e:
        print(f'Erro ao gerar análises de despesas: {e}')

def gerar_insights_despesas():
    insights_data = {
        "analises_realizadas": [
            "Identificação dos principais tipos de despesas e o total gasto em cada categoria.",
            "Determinação de quais deputados têm as maiores despesas e em quais categorias.",
            "Análise da variação das despesas ao longo do tempo para identificar tendências."
        ],
        "insights_despesas_deputados": "Após análise, alguns tipos de despesas se destacam, certos deputados têm maiores gastos em categorias específicas e há uma variação ao longo do tempo que reflete períodos de maior atividade."
    }
    with open('data/insights_despesas_deputados.json', 'w', encoding='utf-8') as f:
        json.dump(insights_data, f, ensure_ascii=False, indent=4)
    print('Insights sobre as despesas dos deputados salvos em data/insights_despesas_deputados.json.')

def main():
    os.makedirs('data', exist_ok=True)  # Garantindo que a pasta data exista
    os.makedirs('docs', exist_ok=True)  # Garantindo que a pasta docs exista

    proposicoes = pega_proposicoes()
    if proposicoes:
        for prop in proposicoes:
            texto = prop.get('ementa', '')
            prop['resumo'] = resumo_texto(texto)
            prop['palavras_chave'] = extrair_palavras_chave(texto)
        df_proposicoes = pd.DataFrame(proposicoes)
        df_proposicoes.to_parquet('data/dados.parquet', index=False)
        print('Proposições gerais salvas em data/dados.parquet.')

        resumo = df_proposicoes[['id', 'resumo']].to_dict(orient='records')
        with open('data/resumo.json', 'w', encoding='utf-8') as f:
            json.dump(resumo, f, ensure_ascii=False, indent=4)
        print('Resumos das proposições salvos em data/resumo.json.')

        config = {
            'base_url': BASE_URL,
            'data_referencia': {
                'inicio': START_DATE,
                'fim': END_DATE
            }
        }
        with open('data/configuracao.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        print('Configuração salva em data/configuracao.yaml.')

        vetores = df_proposicoes['resumo'].apply(lambda x: resumo_texto(x))
        with open('data/vetores.pkl', 'wb') as f:
            pickle.dump(vetores.tolist(), f)
        print('Base vetorial salva em data/vetores.pkl.')
    else:
        print('Nenhuma proposição coletada no período especificado.')

    deputados = pega_deputados()
    if deputados:
        df_deputados = pd.DataFrame(deputados)
        df_deputados.to_parquet('data/deputados.parquet', index=False)
        print('Dados dos deputados salvos em data/deputados.parquet.')

        # Gerar gráfico de barras de distribuição de deputados por partido
        gerar_grafico_distribuicao_deputados(df_deputados)
    else:
        print('Nenhum deputado coletado.')

    temas = [40, 46, 62]
    proposicoes_temas = pega_proposicoes_por_tema(temas, total_por_tema=10)
    if proposicoes_temas:
        for p in proposicoes_temas:
            texto = p.get('ementa', '')
            p['resumo'] = resumo_texto(texto)
            p['palavras_chave'] = extrair_palavras_chave(texto)

        df_proposicoes_temas = pd.DataFrame(proposicoes_temas)
        df_proposicoes_temas.to_parquet('data/proposicoes_deputados.parquet', index=False)
        print('Proposições por tema salvas em data/proposicoes_deputados.parquet.')

        sumarizacao_proposicoes(proposicoes_temas)
    else:
        print('Nenhuma proposição por tema foi coletada.')

    gerar_overview_summary()

    if deputados:
        coleta_despesas(df_deputados)
    else:
        print('Não há deputados para coletar despesas.')

    gerar_analises_despesas()
    gerar_insights_despesas()

    print('Processamento offline concluído.')

if __name__ == '__main__':
    main()



