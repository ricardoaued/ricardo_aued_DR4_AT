
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
