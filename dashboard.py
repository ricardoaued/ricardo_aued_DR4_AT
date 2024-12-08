# dashboard.py

import streamlit as st
import pandas as pd
import json
import yaml
from PIL import Image
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import openai

def carregar_config(config_path='data/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def carregar_insights_distribuicao(insights_path='data/insights_distribuicao_deputados.json'):
    with open(insights_path, 'r', encoding='utf-8') as f:
        insights = json.load(f)
    return insights

def carregar_insights_despesas(insights_path='data/insights_despesas_deputados.json'):
    with open(insights_path, 'r', encoding='utf-8') as f:
        insights = json.load(f)
    return insights

def carregar_summarizacao_proposicoes(summarizacao_path='data/sumarizacao_proposicoes.json'):
    with open(summarizacao_path, 'r', encoding='utf-8') as f:
        sumarizacao = json.load(f)
    return sumarizacao

@st.cache_resource
def carregar_modelo_embedding(model_name="neuralmind/bert-base-portuguese-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def gerar_embedding(texto, tokenizer, model):
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:,0,:].numpy()
    return embedding

@st.cache_resource
def carregar_base_faiss(faiss_index_path='data/faiss_index.bin', metadata_path='data/metadata.json'):
    index = faiss.read_index(faiss_index_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

def consultar_base(query, tokenizer, model, index, metadata, k=5):
    emb = gerar_embedding(query, tokenizer, model)
    D, I = index.search(emb, k)
    resultados = [metadata[i] for i in I[0]]
    return resultados

def assistente_virtual_self_ask(query, tokenizer, model, index, metadata):
    resultados = consultar_base(query, tokenizer, model, index, metadata, k=5)
    # Monta o contexto com os documentos retornados
    contexto = "Aqui estão algumas informações relevantes dos dados:\n"
    for r in resultados:
        contexto += f"- {r.get('texto', 'Sem texto')}\n"

    # Prompt do sistema orientando o LLM sobre a técnica Self-Ask
    prompt_sistema = (
        "Você é um assistente especialista na Câmara dos Deputados. "
        "Você recebeu alguns documentos relevantes da busca vetorial. "
        "Primeiro, se pergunte internamente (Self-Ask) quais subperguntas você precisa responder antes de chegar à resposta final, "
        "use as informações do contexto para respondê-las mentalmente e só então forneça a resposta final ao usuário. "
        "Responda usando apenas as informações fornecidas no contexto, não invente dados."
    )

    # Prompt do usuário com a pergunta e o contexto
    prompt_usuario = f"Pergunta do usuário: {query}\n\n{contexto}\n\nForneça a melhor resposta possível com base nas informações acima."

    # Chamada ao LLM para gerar a resposta final
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        resposta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            max_tokens=300,
            temperature=0.5
        )
        resposta_final = resposta.choices[0].message['content'].strip()
        return resposta_final
    except Exception as e:
        print(f"Erro ao gerar resposta final: {e}")
        return "Desculpe, não foi possível gerar a resposta final no momento."

def main():
    st.set_page_config(page_title="Dashboard Deputados", layout="wide")
    st.title("Dashboard da Câmara dos Deputados")

    abas = st.tabs(["Overview", "Despesas", "Proposições"])

    # Aba Overview
    with abas[0]:
        st.header("Overview")
        st.write("""
            Bem-vindo ao Dashboard da Câmara dos Deputados do Brasil. 
            Aqui você encontrará análises detalhadas sobre as atividades legislativas, distribuição de deputados por partido, despesas e proposições tramitadas.
        """)

        config = carregar_config()
        overview_text = config.get('overview_summary', 'Resumo não disponível.')
        st.subheader("Resumo da Câmara dos Deputados")
        st.write(overview_text)

        st.subheader("Distribuição de Deputados por Partido")
        try:
            imagem = Image.open("docs/distribuicao_deputados.png")
            st.image(imagem, use_column_width=True, caption="Distribuição de Deputados por Partido")
        except FileNotFoundError:
            st.error("Gráfico de distribuição de deputados não encontrado.")

        st.subheader("Insights sobre a Distribuição de Deputados")
        insights_dist = carregar_insights_distribuicao()
        insights_text = insights_dist.get('insights_distribuicao_deputados', 'Insights não disponíveis.')
        st.write(insights_text)

    # Aba Despesas
    with abas[1]:
        st.header("Despesas dos Deputados")

        insights_desp = carregar_insights_despesas()
        insights_text = insights_desp.get("insights_despesas_deputados", "Nenhum insight sobre despesas disponível.")
        st.subheader("Insights sobre Despesas")
        st.write(insights_text)

        try:
            df_despesas = pd.read_parquet('data/serie_despesas_diárias_deputados.parquet')
            st.subheader("Seleção de Deputado")
            lista_deputados = df_despesas['deputado_id'].unique().tolist()
            deputado_selecionado = st.selectbox("Escolha um deputado:", options=lista_deputados)

            st.subheader("Série Temporal de Despesas")
            df_desp_dep = df_despesas[df_despesas['deputado_id'] == deputado_selecionado].copy()
            if not df_desp_dep.empty:
                df_desp_dep = df_desp_dep.sort_values('data')
                st.bar_chart(df_desp_dep.set_index('data')['valorLiquido'])
            else:
                st.info("Não há despesas para o deputado selecionado.")
        except FileNotFoundError:
            st.error("Arquivo de despesas não encontrado.")
        except Exception as e:
            st.error(f"Erro ao carregar despesas: {e}")

    # Aba Proposições
    with abas[2]:
        st.header("Proposições")

        try:
            df_proposicoes = pd.read_parquet('data/proposicoes_deputados.parquet')
            st.subheader("Tabela de Proposições")
            st.dataframe(df_proposicoes)
        except FileNotFoundError:
            st.error("Arquivo de proposições não encontrado.")
        except Exception as e:
            st.error(f"Erro ao carregar proposições: {e}")

        try:
            sumarizacao = carregar_summarizacao_proposicoes()
            summaries = sumarizacao.get("summaries", [])
            st.subheader("Resumo das Proposições (Sumarização por Chunks)")
            if summaries:
                for s in summaries:
                    st.write(f"**Chunk {s['chunk']}:** {s['summary']}")
            else:
                st.info("Nenhum resumo disponível.")
        except FileNotFoundError:
            st.error("Arquivo de sumarização não encontrado.")
        except Exception as e:
            st.error(f"Erro ao carregar sumarização das proposições: {e}")

        st.subheader("Assistente Virtual Especialista em Câmara dos Deputados")
        st.write("Faça uma pergunta sobre deputados, despesas ou proposições:")

        tokenizer, embedding_model = carregar_modelo_embedding()
        faiss_index, metadata = carregar_base_faiss('data/faiss_index.bin', 'data/metadata.json')

        query = st.text_input("Digite sua pergunta:")
        if query:
            resposta = assistente_virtual_self_ask(query, tokenizer, embedding_model, faiss_index, metadata)
            st.write(resposta)

if __name__ == '__main__':
    main()

