import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import plotly.express as px
import io

def dataframe_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    return output.getvalue()

@st.cache_data
def carregar_dados():
    df = pd.read_csv(r'C:/Users/Mario Gouvea/OneDrive/Curso_Tera/docs_ok/df_dash.csv')
    
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['perna_preferida', 'estilo', 'Funcao'])

    caracteristicas = ['altura', 'peso', 'salario_anual', 'aceleracao', 'alcance_aereo', 'agressividade', 'agilidade', 'antecipacao', 'equilibrio', 'bravura', 'comanda_da_area', 'comunicacao', 'compostura', 'concentracao', 'cantos', 'cruzamento', 'decisao', 'determinacao', 'finta', 'excentricidade', 'finalizacao', 'primeiro_toque', 'imprevisibilidade', 'sofre_falta', 'jogo_de_maos', 'cabeceamento', 'impulsao', 'pontape', 'lideranca', 'remates_de_longe', 'lancamentos_longs', 'marcacao', 'aptidao_fisica', 'sem_bola', 'mano_a_mano', 'velocidade', 'Passing', 'marcacao_de_penalti', 'posicionamento', 'tendencia_socar', 'reflexos', 'tendencia_saida', 'resistencia', 'forca', 'desarme', 'trabalho_em_equipe', 'tecnica', 'arremesso', 'visao_de_jogo', 'work_rate']

    caracteristicas_norm = [f"{coluna}_norm" for coluna in caracteristicas]

    # Normalizando os dados
    scaler = MinMaxScaler()
    non_numeric_cols = []
    for coluna in caracteristicas:
        if pd.api.types.is_numeric_dtype(df_encoded[coluna]):
            df_encoded[f"{coluna}_norm"] = scaler.fit_transform(df_encoded[[coluna]])
        else:
            non_numeric_cols.append(coluna)

    return df_encoded, non_numeric_cols, caracteristicas_norm

df_norm, non_numeric_cols, caracteristicas_norm = carregar_dados()

def encontrar_jogadores_semelhantes(id_jogador_desejado, idade_min, idade_max, salario_min, salario_max, k, df_norm, nacionalidades_escolhidas):
    jogador_alvo = df_norm[df_norm['ID'] == id_jogador_desejado]

    # Filtrar jogadores pela posição ideal do jogador alvo
    posicao_ideal_alvo = jogador_alvo['posicao_ideal'].iloc[0]
    df_filtrado = df_norm[(df_norm['idade'] >= idade_min) & 
                          (df_norm['idade'] <= idade_max) & 
                          (df_norm['salario_anual'] >= salario_min) & 
                          (df_norm['salario_anual'] <= salario_max) &
                          (df_norm['posicao_ideal'] == posicao_ideal_alvo)]

    # Filtrar jogadores pela(s) nacionalidade(s) escolhida(s)
    if nacionalidades_escolhidas:
        df_filtrado = df_filtrado[df_filtrado['nacionalidade'].isin(nacionalidades_escolhidas)]


    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(df_filtrado[caracteristicas_norm])
    distancias, indices = knn.kneighbors(jogador_alvo[caracteristicas_norm])

    return df_filtrado.iloc[indices[0]]

# Streamlit
st.title("Sugestão de Jogadores Similares")

# Dropdown para seleção do jogador
jogador_nome = st.selectbox('Selecione o nome do jogador', df_norm['nome'].unique())
jogador_id = df_norm[df_norm['nome'] == jogador_nome]['ID'].iloc[0]

# Multiselect para seleção de nacionalidade(s)
nacionalidades_escolhidas = st.multiselect('Selecione as nacionalidades', df_norm['nacionalidade'].unique())

k = st.number_input('Quantidade de sugestões desejadas', min_value=1, max_value=28, value=10)

## SLIDER DE IDADE 
idade_min, idade_max = st.slider('Selecione o intervalo de idade', 
                                min_value=int(df_norm['idade'].min()), 
                                max_value=int(df_norm['idade'].max()), 
                                value=(int(df_norm['idade'].min()), int(df_norm['idade'].max())))


## SLIDER DE SALARIO
salario_min, salario_max = st.slider('Selecione o intervalo de salário', 
                                     min_value=int(df_norm['salario_anual'].min()), 
                                     max_value=int(df_norm['salario_anual'].max()), 
                                     value=(int(df_norm['salario_anual'].min()), int(df_norm['salario_anual'].max())))



# Botão para buscar
if st.button('Buscar Jogadores Semelhantes'):
    jogadores_semelhantes = encontrar_jogadores_semelhantes(jogador_id, idade_min, idade_max, salario_min, salario_max, k+1, df_norm, nacionalidades_escolhidas)
    jogadores_semelhantes = jogadores_semelhantes[jogadores_semelhantes['ID'] != jogador_id]
    st.subheader(f'Jogadores semelhantes ao jogador de ID {jogador_id}:')
    
    jogador_interesse = df_norm[df_norm['ID'] == jogador_id]
    tabela_comparativa = pd.concat([jogador_interesse, jogadores_semelhantes])

    df_norm['highlight'] = 'Outros'

        # Defina o valor de 'highlight' para o jogador-alvo como 'Jogador-Alvo'
    df_norm.loc[df_norm['ID'] == jogador_id, 'highlight'] = 'Jogador-Alvo'

        # Defina o valor de 'highlight' para os jogadores sugeridos como 'Sugeridos'
    for id_sugerido in jogadores_semelhantes['ID'].values:
        df_norm.loc[df_norm['ID'] == id_sugerido, 'highlight'] = 'Sugeridos'
        
        # Aqui termina o novo código
        
        # Agora, quando você gerar o scatterplot, use a coluna 'highlight' como o argumento color
    fig = px.scatter(df_norm, x='atributos_media_linha', y='salario_anual', 
                    title="Scatter plot of Salário vs. atributos", 
                    hover_data=['nome'], 
                    color='highlight')
    st.plotly_chart(fig)

    # Exibindo o dataframe de forma interativa com as colunas desejadas
    st.dataframe(tabela_comparativa[['ID', 'nome', 'idade', 'nacionalidade', 'Clube', 'divisao', 'valor_de_mercado', 'salario_anual', 'posicao_ideal', 'posicao', 'atributos_media_linha', 'atributos_media_goleiro', 'atributos_goleiro', 'atributos_tecnica', 'atributos_mentais', 'atributos_fisico']])
   
    # Adicionando botão de download para o DataFrame filtrado
    excel_data_filtered = dataframe_to_excel(jogadores_semelhantes[['ID', 'nome', 'idade', 'nacionalidade', 'Clube', 'divisao', 'valor_de_mercado', 'salario_anual', 'posicao_ideal', 'posicao', 'melhor_funcao', 'atributos_media_linha', 'atributos_media_goleiro', 'atributos_goleiro', 'atributos_tecnica', 'atributos_mentais', 'atributos_fisico']])
    st.download_button("Download tabela de jogadores indicados", data=excel_data_filtered, file_name="jogadores_indicados.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # Adicionando botão de download para o DataFrame completo
    excel_data_full = dataframe_to_excel(jogadores_semelhantes)
    st.download_button("Download tabela completa", data=excel_data_full, file_name="jogadores_completo.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
