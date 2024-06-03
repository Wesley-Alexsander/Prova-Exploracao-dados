import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from unidecode import unidecode
from scipy.stats import mode, chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from plotly.subplots import make_subplots

# Leitura da base de dados
DadosCriminais = pd.read_csv("DadosCriminais.csv", low_memory=False, sep=";")
DadosCriminais = DadosCriminais[DadosCriminais['ANO_BO'] != 2021]

# Correções nos nomes dos departamentos e crimes
correcao_dpto = {
    'dipol - depto de inteligencia': 'Dipol ',
    "deinter 2 - campinas": "Deiter 2 ",
    "dope-depto op pol estrat." : "Dope ",
    "demacro": "Demacro ",
    "decap": "Decap "
}
DadosCriminais['NOME_DEPARTAMENTO'] = DadosCriminais['NOME_DEPARTAMENTO'].replace(correcao_dpto)

correcao_crimes = {
    'furto - outros': 'Furto',
    "roubo - outros": "Roubo",
    "lesao corporal dolosa": "Lesao corporal dolosa", 
    "furto de veiculo": "Furto de veiculo",
    "roubo de veiculo": "Roubo de veiculo",
    "lesao corporal culposa por acidade de transito": "Lesao corporal culposa por acidade de transito", 
    "lesao corporal dolosa": "Lesao corporal dolosa",
    "trafico de entorpecentes": "Trafico de entorpecentes"
}
DadosCriminais['NATUREZA_APURADA'] = DadosCriminais['NATUREZA_APURADA'].replace(correcao_crimes)

# Título da página
st.title("Análise de Dados Criminais")

# Crimes por anos
crimes_anos = DadosCriminais.groupby('ANO_BO').size().reset_index(name='QTD')
crimes_anos['%'] = round((crimes_anos['QTD'] / crimes_anos['QTD'].sum()) * 100, 2)

# Criar o gráfico de pizza usando Plotly
fig1 = px.pie(
    crimes_anos, 
    values='QTD', 
    names='ANO_BO', 
    title='Quantidades de Ocorrências por Ano', 
    labels={'QTD': 'Quantidade', 'ANO_BO': 'Ano'},
    color_discrete_map={
        '2022': 'rgb(28, 10, 248)',  # Azul para 2022
        '2023': 'rgb(255, 99, 71)',  # Vermelho para 2023
        '2024': 'rgb(60, 179, 113)'  # Verde opaco para 2024
    }
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig1)

# Texto explicativo para o primeiro gráfico
st.write("Este gráfico mostra a distribuição das ocorrências criminais ao longo dos anos, destacando a quantidade de ocorrências em cada ano.")

# Selecionar apenas três meses específicos
meses_selecionados = [1, 2, 3]  # Por exemplo, janeiro, fevereiro e março
dados_selecionados = DadosCriminais[DadosCriminais['MES_ESTATISTICA'].isin(meses_selecionados)]

# Agrupar os dados por ano, mês e contar as ocorrências
agrupados = dados_selecionados.groupby(['ANO_BO', 'MES_ESTATISTICA']).size().reset_index(name='count')

# Criar um texto de hover que inclui os cinco maiores crimes de cada mês
def criar_hover_text(ano, mes):
    dados_mes = dados_selecionados[(dados_selecionados['ANO_BO'] == ano) & (dados_selecionados['MES_ESTATISTICA'] == mes)]
    top_5_crimes = dados_mes['NATUREZA_APURADA'].value_counts().head(5)
    hover_text = '<br>'.join([f"{crime}: {count}" for crime, count in top_5_crimes.items()])
    return hover_text

agrupados['hover_text'] = agrupados.apply(lambda row: criar_hover_text(row['ANO_BO'], row['MES_ESTATISTICA']), axis=1)

# Definir as cores específicas para cada ano
colors = {
    2022: 'rgba(100, 149, 237, 0.6)',  # Azul para 2022
    2023: 'rgba(255, 99, 71, 0.6)',  # Vermelho para 2023
    2024: 'rgba(60, 179, 113, 0.6)'  # Verde opaco para 2024
}

# Criar o gráfico de barras usando Plotly
fig2 = go.Figure()

# Adicionar barras para cada mês e ano
for ano in [2022, 2023, 2024]:
    for mes in meses_selecionados:
        dados_mes_ano = agrupados[(agrupados['MES_ESTATISTICA'] == mes) & (agrupados['ANO_BO'] == ano)]
        if not dados_mes_ano.empty:
            show_legend = mes == 1  # Mostrar a legenda apenas para o primeiro mês
            fig2.add_trace(go.Bar(
                x=[mes],
                y=dados_mes_ano['count'],
                name=str(ano),
                marker_color=colors[ano],
                hovertemplate='<b>Mês:</b> %{x}<br><b>Ano:</b> %{customdata}<br><b>Contagem:</b> %{y}<br><b>Crimes:</b><br>%{hovertext}',
                customdata=[ano],
                hovertext=dados_mes_ano['hover_text'],
                showlegend=show_legend
            ))

# Ajustar o layout do gráfico
fig2.update_layout(
    title='Contagem de Ocorrências por Mês e Ano',
    xaxis=dict(
        tickmode='array',
        tickvals=meses_selecionados,
        ticktext=['Janeiro', 'Fevereiro', 'Março']  # Ajuste os nomes dos meses conforme necessário
    ),
    barmode='group',  # Garantir que as barras estejam lado a lado
    legend_title_text='Ano',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig2)

# Texto explicativo para o segundo gráfico
st.write("Este gráfico apresenta a contagem de ocorrências criminais para os meses de janeiro, fevereiro e março ao longo dos anos 2022, 2023 e 2024, destacando os cinco crimes mais frequentes em cada mês.")

# Selecionar as colunas 'ANO_BO' e 'MES_ESTATISTICA'
ocorrencias_mes = DadosCriminais[['ANO_BO', 'MES_ESTATISTICA']]

# Agrupar os dados por 'ANO_BO' e 'MES_ESTATISTICA' e calcular a quantidade de registros por mês
ocorrencias_agrupadas = ocorrencias_mes.groupby(['ANO_BO', 'MES_ESTATISTICA']).size().reset_index(name='QTD')

# Criar o gráfico de linhas usando Plotly
fig3 = px.line(
    ocorrencias_agrupadas,
    x='MES_ESTATISTICA',
    y='QTD',
    color='ANO_BO',
    title='Quantidade de Ocorrências por Mês e Ano',
    labels={'MES_ESTATISTICA': 'Mês', 'QTD': 'Quantidade', 'ANO_BO': 'Ano'},
    color_discrete_map={
        '2022': 'rgba(28, 10, 248, 0.6)',  # Azul para 2022
        '2023': 'rgba(255, 99, 71, 0.6)',  # Vermelho para 2023
        '2024': 'rgba(60, 179, 113, 0.6)'  # Verde opaco para 2024
    }
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig3)

# Texto explicativo para o terceiro gráfico
st.write("Este gráfico de linhas mostra a quantidade de ocorrências criminais por mês ao longo dos anos, permitindo identificar tendências sazonais e variações anuais nas ocorrências.")

# Filtrando os dados para três anos diferentes
dados_2012 = DadosCriminais[DadosCriminais['ANO_BO'] == 2022]
dados_2023 = DadosCriminais[DadosCriminais['ANO_BO'] == 2023]
dados_2024 = DadosCriminais[DadosCriminais['ANO_BO'] == 2024]

# Agrupar os dados por região e calcular a quantidade de registros por região para cada ano
ocorrencias_2022 = dados_2012.groupby('regiao').size().reset_index(name='QTD')
ocorrencias_2023 = dados_2023.groupby('regiao').size().reset_index(name='QTD')
ocorrencias_2024 = dados_2024.groupby('regiao').size().reset_index(name='QTD')

# Função para calcular porcentagem de ocorrências
def calcular_porcentagem(ocorrencias):
    total = ocorrencias['QTD'].sum()
    ocorrencias['Porcentagem'] = (ocorrencias['QTD'] / total) * 100
    return ocorrencias

# Aplicar a função para cada ano
ocorrencias_2019 = calcular_porcentagem(ocorrencias_2022)
ocorrencias_2020 = calcular_porcentagem(ocorrencias_2023)
ocorrencias_2021 = calcular_porcentagem(ocorrencias_2024)

# Labels e valores para os gráficos de pizza
labels_2022 = ocorrencias_2019['regiao']
values_2022 = ocorrencias_2019['QTD']

labels_2023 = ocorrencias_2020['regiao']
values_2023 = ocorrencias_2020['QTD']

labels_2024 = ocorrencias_2021['regiao']
values_2024 = ocorrencias_2021['QTD']

# Criar subplots com três gráficos de pizza
fig4 = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])

fig4.add_trace(go.Pie(labels=labels_2022, values=values_2022, hole=.4, name="2022"), 1, 1)
fig4.add_trace(go.Pie(labels=labels_2023, values=values_2023, hole=.4, name="2023"), 1, 2)
fig4.add_trace(go.Pie(labels=labels_2024, values=values_2024, hole=.4, name="2024"), 1, 3)

# Atualizar o layout do gráfico
fig4.update_layout(
    title_text="Regiões Mais Perigosas Anualmente",
    annotations=[dict(text='2022', x=0.11, y=0.5, font_size=20, showarrow=False),
                 dict(text='2023', x=0.5, y=0.5, font_size=20, showarrow=False),
                 dict(text='2024', x=0.89, y=0.5, font_size=20, showarrow=False)]
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig4)

# Texto explicativo para o quarto gráfico
st.write("Este gráfico de subplots mostra as regiões mais perigosas para os anos de 2022, 2023 e 2024, permitindo comparar a distribuição das ocorrências criminais em diferentes regiões ao longo dos anos.")

# Agrupar os dados por região e calcular a quantidade de registros por região
ocorrencias_por_regiao = DadosCriminais.groupby('regiao').size().reset_index(name='QTD')

# Calcular a porcentagem de cada região
total_ocorrencias = ocorrencias_por_regiao['QTD'].sum()
ocorrencias_por_regiao['Porcentagem'] = (ocorrencias_por_regiao['QTD'] / total_ocorrencias) * 100

# Labels e valores para o gráfico de pizza
labels = ocorrencias_por_regiao['regiao']
values = ocorrencias_por_regiao['QTD']

# Criar o gráfico de donut usando Plotly
fig5 = go.Figure(go.Pie(
    labels=labels,
    values=values,
    hole=.4,
    hoverinfo="label+percent",
    textfont_size=20
))

# Atualizar o layout do gráfico
fig5.update_layout(
    title_text="Regiões Mais Perigosas"
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig5)

# Texto explicativo para o quinto gráfico
st.write("Este gráfico de donut mostra a distribuição das ocorrências criminais por região, destacando as regiões mais perigosas com base na quantidade total de ocorrências.")

# Lista de anos para filtrar e criar subplots
anos = [2022, 2023, 2024]

# Mapeamento de naturezas de crimes para cores específicas
color_map = {
    'Furto': '#502dfc',  # Azul
    'Roubo': '#fd5836',  # laranja
    'Lesao corporal dolosa': '#75bf64',  # Verde
    'Furto de veiculo': '#9162f5',  # Magenta
    'Trafico de entorpecentes': 'orange',  # Laranja
    'Roubo de veiculo': 'cyan',  # Ciano
    'Lesao corporal culposa por acidente de transito': 'purple'  # Roxo
}

# Criar subplots
fig6 = make_subplots(
    rows=len(anos), cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=[f'Top Crimes por Região - {ano}' for ano in anos]
)

for i, ano in enumerate(anos):
    # Filtrar dados pelo ano
    dados_ano = DadosCriminais[DadosCriminais['ANO_BO'] == ano]

    # Agrupar e calcular a quantidade de registros por regiao e natureza do crime
    regioes_afetadas = dados_ano.groupby(['regiao', 'NATUREZA_APURADA']).size().reset_index(name='QTD')

    # Ordenar os crimes por quantidade dentro de cada região e pegar os 5 mais comuns
    top_5_crimes_por_regiao = regioes_afetadas.sort_values(['regiao', 'QTD'], ascending=[True, False]).groupby('regiao').head(5)

    # Calcular a porcentagem de cada tipo de crime dentro de cada região
    total_ocorrencias_por_regiao = top_5_crimes_por_regiao.groupby('regiao')['QTD'].sum().reset_index()
    top_5_crimes_por_regiao = top_5_crimes_por_regiao.merge(total_ocorrencias_por_regiao, on='regiao', suffixes=('', '_total'))
    top_5_crimes_por_regiao['Porcentagem'] = (top_5_crimes_por_regiao['QTD'] / top_5_crimes_por_regiao['QTD_total']) * 100

    # Adicionar as barras para cada região e natureza de crime
    for natureza in top_5_crimes_por_regiao['NATUREZA_APURADA'].unique():
        dados_filtrados = top_5_crimes_por_regiao[top_5_crimes_por_regiao['NATUREZA_APURADA'] == natureza]
        
        fig6.add_trace(go.Bar(
            x=dados_filtrados['regiao'],
            y=dados_filtrados['QTD'],
            name=f"{natureza}",
            marker_color=color_map.get(natureza, 'grey'),  # Usar cor específica ou cinza como padrão
            hovertemplate='<b>Região:</b> %{x}<br>' +
                          '<b>Quantidade:</b> %{y}<br>' +
                          '<b>Porcentagem:</b> %{customdata:.2f}%',
            customdata=dados_filtrados['Porcentagem'].values,
            showlegend=(i == 0)  # Mostrar legenda apenas no primeiro gráfico
        ), row=i+1, col=1)

# Ajustar o layout do gráfico
fig6.update_layout(
    title='Top Crimes por Região Separados por Ano',
    xaxis_title='Região',
    yaxis_title='Quantidade',
    barmode='group',  # Agrupar as barras por região
    legend_title='Natureza do Crime',
    height=1000  # Ajustar a altura do gráfico conforme necessário
)

# Forçar a exibição dos rótulos do eixo x em todos os gráficos
fig6.update_xaxes(showticklabels=True)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig6)

# Texto explicativo para o sexto gráfico
st.write("Este gráfico de subplots apresenta os crimes mais comuns por região para os anos de 2022, 2023 e 2024, com barras coloridas para diferentes naturezas de crimes.")

# Selecionar as colunas 'NATUREZA_APURADA' e 'regiao' e calcular a quantidade de registros por regiao e natureza do crime
regioes_afetadas = DadosCriminais[['NATUREZA_APURADA', 'regiao']]
ocorrencias_agrupadas_regioes = regioes_afetadas.groupby(['regiao', 'NATUREZA_APURADA']).size().reset_index(name='QTD')

# Ordenar os crimes por quantidade dentro de cada região e pegar os 5 mais comuns
top_5_crimes_por_regiao = ocorrencias_agrupadas_regioes.sort_values(['regiao', 'QTD'], ascending=[True, False]).groupby('regiao').head(5)

# Calcular a porcentagem de cada tipo de crime dentro de cada região
total_ocorrencias_por_regiao = top_5_crimes_por_regiao.groupby('regiao')['QTD'].sum().reset_index()
top_5_crimes_por_regiao = top_5_crimes_por_regiao.merge(total_ocorrencias_por_regiao, on='regiao', suffixes=('', '_total'))
top_5_crimes_por_regiao['Porcentagem'] = (top_5_crimes_por_regiao['QTD'] / top_5_crimes_por_regiao['QTD_total']) * 100

# Criar um gráfico de barras usando Plotly
fig7 = go.Figure()

# Adicionar as barras para cada região e natureza de crime
for natureza in top_5_crimes_por_regiao['NATUREZA_APURADA'].unique():
    dados_filtrados = top_5_crimes_por_regiao[top_5_crimes_por_regiao['NATUREZA_APURADA'] == natureza]
    
    fig7.add_trace(go.Bar(
        x=dados_filtrados['regiao'],
        y=dados_filtrados['QTD'],
        name=f"{natureza} ({dados_filtrados['Porcentagem'].round(2).astype(str).values[0]}%)",
        hovertemplate='<b>Região:</b> %{x}<br>' +
                      '<b>Quantidade:</b> %{y}<br>' +
                      '<b>Porcentagem:</b> %{customdata:.2f}%',
        customdata=dados_filtrados['Porcentagem'].values,
    ))

# Ajustar o layout do gráfico
fig7.update_layout(
    title='Top Crimes por Região',
    xaxis_title='Região',
    yaxis_title='Quantidade',
    barmode='group',  # Agrupar as barras por região
    legend_title='Natureza do Crime'
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig7)

# Texto explicativo para o sétimo gráfico
st.write("Este gráfico apresenta os crimes mais comuns por região, destacando os cinco principais crimes em cada região.")
