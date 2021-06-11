import pandas as pd
import numpy as np

### Informações sobre os dados

def iqr(x):
    return x.quantile(q=0.75) - x.quantile(q=0.25)

# outlier > 75% +1.5IQR e < 25% -1.5IQR
def outlier_count(x):
    upper_out = x.quantile(q=0.75) + 1.5* iqr(x)
    lower_out = x.quantile(q=0.25) - 1.5* iqr(x)
    
    return len(x[x > upper_out]) + len(x[x < lower_out])

def remove_outlier(x):
    upper_out = x.quantile(q=0.75) + 1.5* iqr(x)
    lower_out = x.quantile(q=0.25) - 1.5* iqr(x)
    
    return x[(x <= upper_out) & (x >= lower_out)]


def df_information(df, id_cols=None):
    
    #mover id_cols
    if id_cols != None:
        df.drop(columns=id_cols)
    #dataframe vazio
    data_info = pd.DataFrame(np.random.randn(0, 12)* 0, 
                         columns=[ 'Num. de Observações',
                                   'Num. de Variáveis',
                                   'Num. Váriáveis numéricas',
                                   'Num. Variáveis de fator',
                                   'Num. Variáveis categoricas',
                                   'Num. Variáveis lógicas',
                                   'Num. Variáveis de data',
                                   'Num. Variáveis com variância zero (uniforme)',
                                   '% de variáveis com valores missing = 0%',
                                   '% de variáveis com valores missing <= 50%',
                                   '% de variáveis com valores missing > 50%',
                                   '% de variáveis com valores missing > 90%'])
    
    # informações dos dados
    data_info.loc[0, 'Num. de Observações'] = df.shape[0]
    data_info.loc[0, 'Num. de Variáveis'] = df.shape[1]
    data_info.loc[0, 'Num. Váriáveis numéricas'] = df.select_dtypes(include=np.number).shape[1]
    data_info.loc[0, 'Num. Variáveis de fator'] = df.select_dtypes(include='category').shape[1]
    data_info.loc[0, 'Num. Variáveis categoricas'] = df.select_dtypes(include='object').shape[1]
    data_info.loc[0, 'Num. Variáveis de data'] = df.select_dtypes(include='datetime64').shape[1]
    data_info.loc[0, 'Num. Variáveis com variância zero (uniforme)'] = df.loc[:, df.apply(pd.Series.nunique)== 1].shape[1]
    
    null_per = pd.DataFrame(df.isnull().sum()/df.shape[0])
    null_per.columns = ['null_per']
    
    data_info.loc[0, '% de variáveis com valores missing = 0%'] = null_per[null_per['null_per'] == 0].shape[0]* 100 / df.shape[1]
    data_info.loc[0, '% de variáveis com valores missing <= 50%'] = null_per[null_per['null_per'] <= 0.5].shape[0]* 100 / df.shape[1]
    data_info.loc[0, '% de variáveis com valores missing > 50%'] = null_per[null_per['null_per'] > 0.5].shape[0]* 100 / df.shape[1]
    data_info.loc[0, '% de variáveis com valores missing > 90%'] = null_per[null_per['null_per'] > 0.9].shape[0]* 100 / df.shape[1]
    
    #coloca os dados em formato de tabela
    data_info = data_info.transpose()
    data_info.columns = ['valor']
    data_info.fillna(0, inplace=True)
    data_info['valor'] = data_info['valor'].astype(int)
    
    return data_info


def num_sumary(df, id_cols=None):
    
    if id_cols != None:
        df_num = df.drop(columns=id_cols).select_dtypes(include=np.number)
    else:
        df_num = df.select_dtypes(include=np.number)
        
    columns = ['Valores negativos','Valores positivos', 
                'Valores iguais a zero', 'Valores únicos', 
                'Valores negativo infinito',
                'Valores positivo infinito',
                'Valores missing',
                'Outliers']
    index = df_num.columns
    
    data_info_num = pd.DataFrame(columns=columns, index=index)
    
    
    # Contagem das estatísticas
    for c in df_num.columns:
            data_info_num.loc[c, 'Valores negativos'] = df_num[df_num[c] < 0].shape[0]
            data_info_num.loc[c, 'Valores positivos'] = df_num[df_num[c] > 0].shape[0]
            data_info_num.loc[c, 'Valores iguais a zero'] = df_num[df_num[c] == 0].shape[0]
            data_info_num.loc[c, 'Valores únicos'] = len(df_num[c].unique())
            data_info_num.loc[c, 'Valores negativo infinito'] = df_num[df_num[c] == -np.inf].shape[0]
            data_info_num.loc[c, 'Valores positivo infinito'] = df_num[df_num[c] == np.inf].shape[0]
            data_info_num.loc[c, 'Valores missing'] = df_num[df_num[c].isnull()].shape[0]
            data_info_num.loc[c, 'Outliers'] = outlier_count(df_num[c])
            
    
    
    print('outlier > 75% +.5IQR e < 25% -1.5IQR')
    
    return data_info_num

def cat_sumary(df, id_cols=None):
    
    if id_cols != None:
        df_cat = df.drop(columns=id_cols).select_dtypes(include=['object', 'bool'])
    else:
        df_cat = df.select_dtypes(include=['object', 'bool'])
    
    columns = ['valores únicos', 'valores missing']
    index = df_cat.columns
    
    df_cat_info = pd.DataFrame(columns=columns, index=index)
    
     # Contagem das estatísticas
    for c in df_cat.columns:
        df_cat_info.loc[c, 'valores únicos'] = len(df_cat[c].unique())
        df_cat_info.loc[c, 'valores missing'] = df_cat[df_cat[c].isnull()].shape[0]
    
    df_cat_info.fillna(0, inplace=True)   
    return df_cat_info

def stats_num(df, id_cols=None):
    
    if id_cols != None:
        df_num = df.drop(columns=id_cols).select_dtypes(include=np.number)
    else:
        df_num = df.select_dtypes(include=np.number)
        
        
    df_stats_num = pd.DataFrame()
    
    df_stats_num = pd.concat([df_num.describe().transpose(),
                             pd.DataFrame(df_num.quantile(q=0.10)),
                             pd.DataFrame(df_num.quantile(q=0.90)),
                             pd.DataFrame(df_num.quantile(q=0.95))], axis=1)
    
    df_stats_num.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', '10%', '90%', '95%']
    
    return df_stats_num
