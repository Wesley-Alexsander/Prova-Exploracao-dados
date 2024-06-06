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

@st.cache_data # Adiciona cache à função de carregar dados
def carregar_dados():
    return pd.read_csv("DadosCriminais.csv", low_memory=False, sep=";")  # Substitua pelo seu caminho de arquivo ou função de carga

DadosCriminais = carregar_dados()
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


# ========================================================================================================================================= #
# Título da página
st.title("Análise de Dados Criminais")
st.write('-'*10)


st.subheader("Introdução")
# Texto explicativo para o primeiro gráfico
st.write("""Neste trabalho, escolhemos explorar a base de dados de crimes cometidos na cidade de São Paulo, fornecida por órgãos de segurança pública. A base de dados contém informações detalhadas sobre boletins de ocorrência registrados na cidade
         , incluindo a natureza do crime, a data e hora da ocorrência, a localização, e outras informações relevantes.""")


# ========================================================================================================================================= #
st.write('-'*10)

st.subheader("Análises:")

st.write("""No gráfico de pizza, observa-se que 2023 foi o ano com maior quantidade de crimes registrados, representando 44.3% das ocorrências. Já 2022 contou com 42% dos crimes
, enquanto 2024, até agora, apresenta a menor porcentagem com 13.7% pois não está com todos os meses de referência, mas ainda sim apresenta caracteristicas muito interessantes.""")

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

# =============================================================================================================================================== #
st.write("""Abaixo veremos as medidas estatisticas da nossa base filtradas pelos anos, podendo assim verificar a distribuição e o comportamento dos dados.""")

# Função para criar o DataFrame com estatísticas
@st.cache_data
def criar_tabela_estatisticas(df_filtrado):
    frequencias = df_filtrado['NATUREZA_APURADA'].value_counts()
    estatisticas_descritivas = round(frequencias.describe(), 2)
    variancia = round(frequencias.var(), 2)
    moda = df_filtrado['NATUREZA_APURADA'].mode()
    categoria_max = frequencias.idxmax()
    categoria_min = frequencias.idxmin()

    data = {
        'Medidas': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'variancia', 'moda'],
        'Valores': [
            estatisticas_descritivas['count'], estatisticas_descritivas['mean'], estatisticas_descritivas['std'],
            estatisticas_descritivas['min'], estatisticas_descritivas['25%'], estatisticas_descritivas['50%'],
            estatisticas_descritivas['75%'], estatisticas_descritivas['max'], variancia, moda.iloc[0]
        ],
        'Categoria': ['', '', '', categoria_min, '', '', '', categoria_max, '', '']
    }
    return pd.DataFrame(data, index=None)

# Streamlit application
ano_escolhido = st.selectbox('Escolha o ano:', [2022, 2023, 2024, 'Todos'])
if ano_escolhido == 'Todos':
    df_filtrado = DadosCriminais
else:
    df_filtrado = DadosCriminais[DadosCriminais['ANO_BO'] == ano_escolhido]

tabela_medidas = criar_tabela_estatisticas(df_filtrado)

st.subheader('Tabela de Medidas Estatísticas')
st.table(tabela_medidas)


# =============================================================================================================================================== #

st.write("-"*10)

# Texto explicativo para o primeiro gráfico
st.write("""No gráfico de barras, uma análise mensal revela padrões interessantes nos primeiros meses de cada ano. 
         Em 2023, janeiro começou com um alto número de crimes, mas houve uma leve redução em fevereiro
         , indicando uma possível resposta efetiva às estratégias de segurança implantadas após um início de ano desafiador. 
         Já 2024 mostra um panorama diferente: começou com um número relativamente baixo de ocorrências em janeiro, mas experimentou um aumento em fevereiro
         , sabendo que não temos informações completas do mês de fevereiro o gráfico sugere uma tendência de crescimento na criminalidade nesse período. 
         Essa tendencia será melhor apresentada nos gráficos posteriores""")

# Selecionar apenas três meses específicos
meses_selecionados = [1, 2, 3]  # Por exemplo, janeiro, fevereiro e março
dados_selecionados = DadosCriminais[DadosCriminais['MES_ESTATISTICA'].isin(meses_selecionados)]

# Agrupar os dados por ano, mês e contar as ocorrências
agrupados = dados_selecionados.groupby(['ANO_BO', 'MES_ESTATISTICA']).size().reset_index(name='count')

# Criar um texto de hover que inclui os cinco maiores crimes de cada mês
@st.cache_data
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

# ================================================================================================================================================= # 

# Texto explicativo para o segundo gráfico
st.write("""Em março de 2023, registrou-se um pico expressivo nas ocorrências, coincidindo com a realização do Carnaval
         , um período de grande aglomeração e festividades que historicamente contribui para um aumento nos índices de crimes
         , variando desde pequenos furtos até ocorrências mais graves. A natureza expansiva do Carnaval, que atrai multidões e gera ambientes menos controlados
         , é um fator crítico nesse aumento sazonal da criminalidade.""")

st.write("""Em 2022, a instabilidade nas ocorrências criminais está intimamente ligada aos efeitos residuais da pandemia de COVID-19. 
         Apesar de a situação sanitária ter começado a se estabilizar, o vírus ainda estava presente, influenciando o comportamento da população. 
         Nos primeiros meses do ano, observou-se uma queda nas ocorrências, em grande parte devido aos apelos e restrições para que as pessoas ficassem em casa. 
         Contudo, com a aproximação do Carnaval, muitos desobedeceram essas recomendações, resultando em um aumento na criminalidade notável de fevereiro para março. 
         Este fenômeno ressalta como eventos culturais significativos podem perturbar padrões de criminalidade, mesmo em contextos de saúde pública adversos.""")

st.write("""Para 2024, embora o ano tenha começado com uma menor incidência de crimes, observa-se uma tendência de aumento entre janeiro e fevereiro. 
         Este padrão inicial sugere uma possível escalada nas ocorrências à medida que o ano progride
         , necessitando de monitoramento contínuo para verificar se essa tendência se mantém ou se alterações nas políticas de segurança pública poderão mitigar tais elevações.""")

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

st.write('-'*10)
# =================================================================================================================================================================== #
# Texto explicativo para o terceiro gráfico
st.write("""De acordo com o gráfico apresentado, observamos que, em 2022, a Zona Leste foi a região com o maior índice de criminalidade
         , concentrando 28,8% do total de Boletins de Ocorrência registrados naquele ano. No entanto, nos anos subsequentes
         , houve uma mudança significativa na distribuição geográfica da criminalidade. Em 2023 e 2024, a Zona Leste foi superada pela Zona Sul
         , que registrou um aumento expressivo no número de boletins de ocorrência. Esse deslocamento destaca uma mudança notável na dinâmica da criminalidade na cidade
         , com a Zona Sul emergindo como o novo foco de maior preocupação em relação à segurança pública.""")


# Seleção do ano pelo usuário
ano_escolhido = st.selectbox('Escolha:', [2022, 2023, 2024, 'Todos'])


# Função para calcular porcentagem de ocorrências
@st.cache_data
def calcular_porcentagem(ocorrencias):
    total = ocorrencias['QTD'].sum()
    ocorrencias['Porcentagem'] = (ocorrencias['QTD'] / total) * 100
    return ocorrencias

@st.cache_data
def criar_pie_chart(ano):
    dados = DadosCriminais[DadosCriminais['ANO_BO'] == ano]
    ocorrencias = dados.groupby('regiao').size().reset_index(name='QTD')
    ocorrencias = calcular_porcentagem(ocorrencias)
    labels = ocorrencias['regiao']
    values = ocorrencias['QTD']
    return go.Pie(labels=labels, values=values, hole=.4, name=str(ano))

# Criar subplots com até três gráficos de pizza, dependendo da seleção
fig4 = make_subplots(rows=1, cols=3 if ano_escolhido == 'Todos' else 1, specs=[[{'type':'domain'}] * (3 if ano_escolhido == 'Todos' else 1)])

if ano_escolhido == 'Todos':
    fig4.add_trace(criar_pie_chart(2022), 1, 1)
    fig4.add_trace(criar_pie_chart(2023), 1, 2)
    fig4.add_trace(criar_pie_chart(2024), 1, 3)
    annotations = [
        dict(text='2022', x=0.11, y=0.5, font_size=20, showarrow=False),
        dict(text='2023', x=0.5, y=0.5, font_size=20, showarrow=False),
        dict(text='2024', x=0.89, y=0.5, font_size=20, showarrow=False)
    ]
else:
    ano = int(ano_escolhido)
    fig4.add_trace(criar_pie_chart(ano), 1, 1)
    annotations = [dict(text=str(ano), x=0.5, y=0.5, font_size=20, showarrow=False)]

fig4.update_layout(title_text="Regiões Mais Perigosas Anualmente", annotations=annotations)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig4)

# =================================================================================================================================================================== #
# Texto explicativo para o quarto gráfico
st.write("""Ao análisarmos a consolidação dos 3 anos, dados recentes indicam que a Zona Sul lidera em incidência criminal
         , com 29.4% do total de ocorrências registradas. 
         Este dado coloca a Zona Sul como a região mais crítica da metrópole em termos de segurança.""")

st.write("""Segue-se a Zona Leste e a Zona Central
         , com 27.8% e 19.2% das ocorrências, respectivamente. 
         Estas áreas também mostram uma concentração significativa de atividades criminais
         , refletindo a necessidade de atenção e intervenção estratégica. Por outro lado
         , as Zonas Norte e Oeste apresentam incidências relativamente menores, com 12% e 11.6% das ocorrências
         , sugerindo um perfil de segurança um pouco mais estável, mas ainda assim suscetível a desafios.""")

# Agrupar os dados por região e calcular a quantidade de registros por região
ocorrencias_por_regiao = DadosCriminais.groupby('regiao').size().reset_index(name='QTD')

# Calcular a porcentagem de cada região
total_ocorrencias = ocorrencias_por_regiao['QTD'].sum()
ocorrencias_por_regiao['Porcentagem'] = (ocorrencias_por_regiao['QTD'] / total_ocorrencias) * 100

# Ordenar os dados para o gráfico de funil
ocorrencias_por_regiao = ocorrencias_por_regiao.sort_values('QTD', ascending=False)

# Criar o gráfico de funil usando Plotly
fig5 = go.Figure(go.Funnel(
    y=ocorrencias_por_regiao['regiao'],
    x=ocorrencias_por_regiao['QTD'],
    textinfo="value+percent total",
    hoverinfo="name+percent total"
))

# Atualizar o layout do gráfico
fig5.update_layout(
    title_text="Regiões Mais Perigosas - Gráfico de Funil",
    xaxis_title='Quantidade de Ocorrências',
    yaxis_title='Região'  # Aqui foi corrigido de 'yashis_title' para 'yaxis_title'
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig5)

# =================================================================================================================================================================== #

st.write("""Abaixo veremos as medidas estatisticas da nossa base filtradas pelos anos, podendo assim verificar a distribuição e o comportamento dos dados.""")

# Função para criar o DataFrame com estatísticas
@st.cache_data
def criar_tabela_estatisticas_regioes(df_filtrado):
    frequencias_regioes = df_filtrado['regiao'].value_counts()
    estatisticas_descritivas_regioes = round(frequencias_regioes.describe(), 2)
    variancia_regioes = round(frequencias_regioes.var(), 2)
    moda = df_filtrado['regiao'].mode()
    categoria_max_regioes = frequencias_regioes.idxmax()
    categoria_min_regioes = frequencias_regioes.idxmin()

    data_regioes = {
        'Medidas': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'variancia', 'moda'],
        'Valores': [
            estatisticas_descritivas_regioes['count'], estatisticas_descritivas_regioes['mean'], estatisticas_descritivas_regioes['std'],
            estatisticas_descritivas_regioes['min'], estatisticas_descritivas_regioes['25%'], estatisticas_descritivas_regioes['50%'],
            estatisticas_descritivas_regioes['75%'], estatisticas_descritivas_regioes['max'], variancia_regioes, moda.iloc[0]
        ],
        'Categoria': ['', '', '', categoria_min_regioes, '', '', '', categoria_max_regioes, '', '']
    }
    return pd.DataFrame(data_regioes, index=None)

# Streamlit application
ano_regiao = st.selectbox('Escolha o ano das regiões:', [2022, 2023, 2024, 'Todos'])
if ano_regiao == 'Todos':
    df_filtrado = DadosCriminais
else:
    df_filtrado = DadosCriminais[DadosCriminais['ANO_BO'] == ano_regiao]

tabela_medidas = criar_tabela_estatisticas_regioes(df_filtrado)

st.subheader('Tabela de Medidas Estatísticas')
st.table(tabela_medidas)




# =================================================================================================================================================================== #
st.write('-'*10)

st.write("""Este gráfico fornece uma visão detalhada das diferenças regionais 
         em relação à incidência de crimes. Observa-se que a Zona Sul se destaca com as maiores taxas tanto de 
         furtos classificados quanto de roubos, indicando que essa região enfrenta os mais graves desafios de 
         segurança na cidade. Os furtos, que representam 67.01% das ocorrências gerais, e os roubos, com 26.63%
         , são especialmente predominantes na Zona Sul, demonstrando uma necessidade crítica de intervenções focadas.
    """)

st.write("""A Zona Central também mostra taxas elevadas de furtos, ficando atrás apenas da Zona Sul. 
         Esse alto índice de furtos nesta região central, que é um hub comercial e turístico
         , sublinha a importância de estratégias de prevenção e vigilância reforçada. Já a Zona Leste, 
         junto com a Zona Sul, registra altas incidências de roubos. 
         A significativa ocorrência destes crimes nas zonas Sul e Leste e em uma medida considerável na Central 
         sugere que estas áreas se beneficiariam de um aumento no patrulhamento policial 
         e na instalação de mais infraestrutura de segurança, como câmeras de vigilância.
    """)

# Texto explicativo para o sexto gráfico
st.write("""Além desses, outros crimes como "Lesão corporal culposa por acidente de trânsito"
         , "Roubo de veículo", e "Furto de veículo" aparecem em proporções menores, mas ainda significativas. 
         O tráfico de entorpecentes é o menos prevalente, marcando presença mais discreta em comparação 
         aos outros crimes listados.""")

# Interface do usuário para selecionar o ano
ano_escolhido_crime = st.selectbox('Escolha o ano dos crimes:', [2022, 2023, 2024, 'Todos'])

# Filtrar dados com base na escolha do ano
if ano_escolhido_crime != 'Todos':
    regioes_afetadas = DadosCriminais[(DadosCriminais['ANO_BO'] == ano_escolhido_crime)]
else:
    regioes_afetadas = DadosCriminais

# Selecionar as colunas e calcular a quantidade de registros por região e natureza do crime
regioes_afetadas = regioes_afetadas[['NATUREZA_APURADA', 'regiao', 'ANO_BO']]
ocorrencias_agrupadas_regioes = regioes_afetadas.groupby(['regiao', 'NATUREZA_APURADA']).size().reset_index(name='QTD')

# Ordenar os crimes por quantidade dentro de cada região e pegar os 5 mais comuns
top_crimes_por_regiao = ocorrencias_agrupadas_regioes.sort_values(['regiao', 'QTD'], ascending=[True, False]).groupby('regiao').head(5)

# Calcular a porcentagem de cada tipo de crime dentro de cada região
total_ocorrencias_por_regiao = top_crimes_por_regiao.groupby('regiao')['QTD'].sum().reset_index()
top_crimes_por_regiao = top_crimes_por_regiao.merge(total_ocorrencias_por_regiao, on='regiao', suffixes=('', '_total'))
top_crimes_por_regiao['Porcentagem'] = (top_crimes_por_regiao['QTD'] / top_crimes_por_regiao['QTD_total']) * 100

# Criar um gráfico de barras usando Plotly
fig6 = go.Figure()

# Adicionar as barras para cada região e natureza de crime
for natureza in top_crimes_por_regiao['NATUREZA_APURADA'].unique():
    dados_filtrados = top_crimes_por_regiao[top_crimes_por_regiao['NATUREZA_APURADA'] == natureza]
    
    fig6.add_trace(go.Bar(
        x=dados_filtrados['regiao'],
        y=dados_filtrados['QTD'],
        name=f"{natureza} ({dados_filtrados['Porcentagem'].round(2).astype(str).values[0]}%)",
        hovertemplate='<b>Região:</b> %{x}<br>' +
                      '<b>Quantidade:</b> %{y}<br>' +
                      '<b>Porcentagem:</b> %{customdata:.2f}%',
        customdata=dados_filtrados['Porcentagem'].values,
    ))

# Ajustar o layout do gráfico
fig6.update_layout(
    title='Top Crimes por Região',
    xaxis_title='Região',
    yaxis_title='Quantidade',
    barmode='group',  # Agrupar as barras por região
    legend_title='Natureza do Crime'
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig6)

# =================================================================================================================================================================== #
st.write("""Abaixo veremos as medidas estatisticas da nossa base filtradas pelos anos, podendo assim verificar a distribuição e o comportamento dos dados.""")
import streamlit as st
import pandas as pd

# Função para criar o DataFrame com estatísticas
@st.cache_data
def criar_tabela_estatisticas_regioes(df_filtrado):
    frequencias_regioes = df_filtrado['NATUREZA_APURADA'].value_counts()
    estatisticas_descritivas_regioes = round(frequencias_regioes.describe(), 2)
    variancia_regioes = round(frequencias_regioes.var(), 2)
    moda = df_filtrado['NATUREZA_APURADA'].mode()
    categoria_max_regioes = frequencias_regioes.idxmax()
    categoria_min_regioes = frequencias_regioes.idxmin()

    data_regioes = {
        'Medidas': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'variancia', 'moda'],
        'Valores': [
            estatisticas_descritivas_regioes['count'], estatisticas_descritivas_regioes['mean'], estatisticas_descritivas_regioes['std'],
            estatisticas_descritivas_regioes['min'], estatisticas_descritivas_regioes['25%'], estatisticas_descritivas_regioes['50%'],
            estatisticas_descritivas_regioes['75%'], estatisticas_descritivas_regioes['max'], variancia_regioes, moda.iloc[0]
        ],
        'Categoria': ['', '', '', categoria_min_regioes, '', '', '', categoria_max_regioes, '', '']
    }
    return pd.DataFrame(data_regioes, index=None)

# Streamlit application
regiao_selecionada = st.selectbox(
    'Escolha a região das ocorrências:', ['zona_Central', 'zona_Leste', 'zona_Norte', 'zona_Oeste', 'zona_Sul', 'Todos']
    )

if regiao_selecionada == 'Todos':
    df_filtrado = DadosCriminais
else:
    df_filtrado = DadosCriminais[DadosCriminais['regiao'] == regiao_selecionada]

tabela_medidas = criar_tabela_estatisticas_regioes(df_filtrado)

st.subheader('Tabela de Medidas Estatísticas')
st.table(tabela_medidas)

# ================================================================================================================================= #
st.write('-'*10)

st.subheader('Conclusões')

st.write("""Os dados apresentados nos gráficos permitem uma compreensão detalhada das tendências de criminalidade ao longo dos últimos três anos. 
         Em 2023, observou-se o pico nas ocorrências, indicando um ano particularmente desafiador em termos de segurança pública. 
         Em contraste, 2024 mostra uma diminuição significativa nas ocorrências criminais até agora, o que pode ser interpretado como um sinal positivo
         , possivelmente resultante de estratégias de segurança efetivas implementadas após o ápice em 2023. """)

st.write("""No entanto, a análise mensal de 2024 revela um aumento de ocorrências de janeiro para fevereiro
         , uma tendência que merece atenção para evitar uma escalada durante o resto do ano. Este aumento sugere que
         , apesar das melhorias gerais, existem dinâmicas específicas que podem estar influenciando a criminalidade nesse período inicial do ano.""")

st.write("""Os resultados indicam que a Zona Sul e a Zona Central de São Paulo são as regiões com maior incidência de criminalidade
         , especialmente em relação a furtos e roubos, que são os crimes mais comuns. Essas áreas apresentaram um aumento significativo de ocorrências durante os meses de fevereiro
         , coincidindo com o período do Carnaval, um evento sazonal que historicamente atrai um aumento na atividade criminal devido às grandes aglomerações.""")

st.write("""A prevalência de furtos nas regiões mais movimentadas da cidade pode ser atribuída à maior oportunidade de crimes contra propriedade em áreas densamente povoadas e turísticas. 
         O aumento dos crimes durante eventos sazonais sugere uma necessidade de reforço na segurança e vigilância durante esses períodos. 
         A implementação de postos de comando móveis e o uso intensificado de tecnologia de vigilância são estratégias que têm mostrado potencial para mitigar esses picos de criminalidade. """)

st.write("""Recomenda-se que as autoridades intensifiquem o patrulhamento nas regiões identificadas como de alto risco
         , especialmente durante eventos de grande porte que são conhecidos por atrair grande número de pessoas. 
         A alocação estratégica de recursos durante esses períodos pode incluir o aumento da visibilidade policial e a implementação de tecnologias avançadas de monitoramento. 
         Além disso, campanhas de conscientização pública antes desses eventos podem ajudar a reduzir a incidência de furtos, informando tanto residentes quanto visitantes sobre precauções de segurança. """)

st.write("""Este estudo resalta a importância de uma abordagem adaptativa e baseada em evidências para o planejamento de segurança pública
         , focando em intervenções específicas durante períodos de alto risco e em áreas de alta criminalidade.  """)

# ============================================================================================================================================================================================================ #
st.write('-'*10)

st.subheader('Referências')

st.write("""SSPSP, SP Transparência Números sem Mistérios Consulta, SSPSP 2024. Disponível em: https://www.ssp.sp.gov.br/estatistica/consultas""")
st.write("""IBGE, População, IBGE 2022. Disponível em: https://cidades.ibge.gov.br/brasil/sp/panorama""")


