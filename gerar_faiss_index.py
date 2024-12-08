# gerar_faiss_index.py

import faiss
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

def gerar_embedding(texto, tokenizer, model):
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:,0,:].numpy()
    return embedding

def main():
    if not os.path.exists('data'):
        os.makedirs('data')

    # Carregar modelo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    metadata = []
    textos = []

    # Deputados
    if os.path.exists('data/deputados.parquet'):
        df_deputados = pd.read_parquet('data/deputados.parquet')
        for _, row in df_deputados.iterrows():
            texto = f"Deputado ID: {row['id']}, Nome: {row['nome']}, Partido: {row['siglaPartido']}"
            textos.append(texto)
            metadata.append({'id': row['id'], 'tipo': 'deputado', 'texto': texto})
    else:
        print("Arquivo data/deputados.parquet não encontrado. Continuando sem deputados...")

    # Proposições por tema
    if os.path.exists('data/proposicoes_deputados.parquet'):
        df_proposicoes = pd.read_parquet('data/proposicoes_deputados.parquet')
        for _, row in df_proposicoes.iterrows():
            texto = f"Proposição ID: {row['id']}, Tipo: {row.get('idTipo', '')}, Ementa: {row.get('ementa', '')}, Resumo: {row.get('resumo', '')}"
            textos.append(texto)
            metadata.append({'id': row['id'], 'tipo': 'proposicao', 'texto': texto})
    else:
        print("Arquivo data/proposicoes_deputados.parquet não encontrado. Sem proposições por tema...")

    # Sumarização das proposições
    if os.path.exists('data/sumarizacao_proposicoes.json'):
        with open('data/sumarizacao_proposicoes.json', 'r', encoding='utf-8') as f:
            sumarizacao = json.load(f)
        for summary in sumarizacao.get('summaries', []):
            texto = f"Sumarização Chunk {summary['chunk']}: {summary['summary']}"
            textos.append(texto)
            metadata.append({'id': f"sumarizacao_{summary['chunk']}", 'tipo': 'sumarizacao', 'texto': texto})
    else:
        print("Arquivo data/sumarizacao_proposicoes.json não encontrado. Sem sumarizações...")

    # Caso não haja textos, não criamos o índice
    if not textos:
        print("Nenhum dado encontrado para indexar no FAISS. Verifique a coleta de dados.")
        return

    embeddings = []
    for t in textos:
        emb = gerar_embedding(t, tokenizer, model)
        embeddings.append(emb[0])

    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, 'data/faiss_index.bin')

    with open('data/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("Índice FAISS e metadados salvos com sucesso.")

if __name__ == '__main__':
    main()
