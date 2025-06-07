import numpy as np
import pandas as pd
from scipy.stats import kstest  # Prueba Kolmogorov-Smirnov

class AnalizadorDataFrame:
    """Clase para análisis exploratorio de DataFrames de pandas.
    
    Proporciona métodos para calcular estadísticas descriptivas, clasificar variables
    y generar resúmenes de columnas individuales.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Inicializa el analizador con un DataFrame.
        
        Parámetros:
        -----------
            df (pd.DataFrame): DataFrame a analizar.
        
        Atributos:
        ----------
            __df (pd.DataFrame): Copia del DataFrame original.
            __estadisticas (dict): Estadísticas básicas del DataFrame.
            __clasificacion_variables (dict): Clasificación de variables en cualitativas/cuantitativas.
        """
        self.df = df.copy()
        self.__clasificacion_variables = self.__clasificar_variables()
    
    # Getters y Setters
    @property
    def df(self) -> pd.DataFrame:
        """Obtiene el DataFrame actual.
        
        Retorna
        -------
            (pd.DataFrame) Copia del DataFrame almacenado.
        """
        return self.__df.copy()

    @property
    def clasificacion_variables(self) -> dict:
        """Obtiene la clasificación de variables del DataFrame.
        
        Retorna
        -------
        (dict) Diccionario con la clasificación de variables.
        """
        return self.__clasificacion_variables.copy()
    
    @df.setter
    def df(self, nuevo_df: pd.DataFrame):
        """Establece un nuevo DataFrame y recalcula las estadísticas.
        
        Parámetros
        ----------
        nuevo_df (pd.DataFrame): Nuevo DataFrame a analizar.
        """
        self.__df = nuevo_df.copy()
        self.__clasificacion_variables = self.__clasificar_variables()
    
    
    # Métodos internos
    def __clasificar_variables(self) -> dict:
        """Clasifica las variables en cualitativas y cuantitativas.
        
        Retorna
        -------
            (dict) Diccionario con listas de columnas clasificadas:
                - Cualitativas: Variables no numéricas
                - Cuantitativas: Variables numéricas
        """
        cualitativas = []
        cuantitativas = []
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                cuantitativas.append(col)
            else:
                cualitativas.append(col)
        
        return {
            'Cualitativas': cualitativas,
            'Cuantitativas': cuantitativas
        }
    
    def resumen_estadisticos(self) -> dict:
        """Calcula el resumen de algunos estadísticos de las variables cuantitativas.
        
        Retorna
        -------
            (dict) Diccionario con:
                - mínimo
                - primer cuartil (25%)
                - mediana (50%)
                - tercer cuartil (75%)
                - máximo
                - media
                - desviacion estandar
                para cada variable cuantitativa
        """
        resumen = {}
        
        for col in self.clasificacion_variables['Cuantitativas']:
            desc = self.df[col].describe(percentiles=[.25, .5, .75])
            resumen[col] = {
                'minimo': desc['min'],
                'primer_cuartil': desc['25%'],
                'mediana': desc['50%'],
                'tercer_cuartil': desc['75%'],
                'maximo': desc['max'],
                'media': desc['mean'],
                'desviacion_estandar': desc['std']
            }
        
        return resumen
    
    def clasificar_cualitativas(self, alpha: float = 0.2) -> dict:
        """Clasifica variables cualitativas en repetitivas o únicas.
        
        Parámetros
        ----------
            alpha (float): Umbral para clasificación (default 0.2).
                        Si nunq/nobs <= alpha, se considera repetitiva.
        
        Retorna
        -------
            (dict) Diccionario con:
                - repetitivas: Variables con valores repetidos (ej. géneros)
                - unicas: Variables con valores únicos (ej. nombres)
        """
        clasificacion = {'repetitivas': [], 'unicas': []}
        
        for col in self.clasificacion_variables['Cualitativas']:
            nunq = self.df[col].nunique()
            nobs = len(self.df[col].dropna())
            ratio = nunq / nobs if nobs > 0 else 0
            
            if ratio <= alpha:
                clasificacion['repetitivas'].append(col)
            else:
                clasificacion['unicas'].append(col)
        
        return clasificacion
    
    def analizar_columna(self, columna: str) -> dict:
        """Realiza análisis completo de una columna específica.
        
        Parámetros
        ----------
            columna (str): Nombre de la columna a analizar.
        
        Retorna
        -------
            (dict) Diccionario con:
                - nombre: Nombre de la columna
                - tipo: Tipo de datos
                - nulos: Cantidad de valores nulos
                - unicos: Cantidad de valores únicos
                - resumen_5_numeros (si es cuantitativa)
                - clasificacion_cualitativa (si es cualitativa)
        
        Raises
        ------
            ValueError
                Si la columna no existe en el DataFrame.
        """
        if columna not in self.df.columns:
            raise ValueError(f"Columna '{columna}' no encontrada")
        
        resultado = {
            'nombre': columna,
            'tipo': str(self.df[columna].dtype),
            'nulos': self.df[columna].isnull().sum(),
            'unicos': self.df[columna].nunique()
        }
        ratio = resultado['unicos'] / len(self.df[columna].dropna())
        resultado['clasificacion_cualitativa'] = 'repetitiva' if ratio <= 0.2 else 'única'

        if columna in self.clasificacion_variables['Cuantitativas']:
            desc = self.df[columna].describe(percentiles=[.25, .5, .75])
            resultado['resumen_estadisticos'] = {
                'minimo': desc['min'],
                'primer_cuartil': desc['25%'],
                'mediana': desc['50%'],
                'tercer_cuartil': desc['75%'],
                'maximo': desc['max']
            }
        
        
        return resultado
    
    def mostrar_estadisticas(self):
        return self.df.info()


    def correlacion_pearson_manual(self, col_x: str, col_y: str) -> float:
        """Calcula la correlación lineal de Pearson entre dos columnas numéricas.
        
        Parámetros:
        ----------
            col_x (str): Nombre de la primera columna (X).
            col_y (str): Nombre de la segunda columna (Y).
        
        Retorna:
        -------
            (float) Valor de la correlación de Pearson entre las dos variables.
        
        Raises:
        ------
            ValueError: Si alguna de las columnas no es cuantitativa o no existe.
        """
        if col_x not in self.clasificacion_variables['Cuantitativas'] or \
           col_y not in self.clasificacion_variables['Cuantitativas']:
            raise ValueError("Ambas columnas deben ser cuantitativas.")

        x = self.df[col_x].dropna()
        y = self.df[col_y].dropna()
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
        
        sum_xy = np.sum(x * y)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x ** 2)
        sum_y2 = np.sum(y ** 2)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        
        return numerator / denominator if denominator != 0 else np.nan

    def detectar_atipicos_tukey(self) -> dict:
        """Detecta valores atípicos por columna cuantitativa usando el criterio de Tukey.
        
        Retorna
        -------
        (dict) Diccionario con:
            - índices: Lista de posiciones con valores atípicos
            - Número de valores atípicos: Cantidad total por columna
        """
        resultado = {}
        
        for col in self.clasificacion_variables['Cuantitativas']:
            serie = self.df[col].dropna()
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1
            lim_inf = Q1 - 1.5 * IQR
            lim_sup = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lim_inf) | (self.df[col] > lim_sup)].index.tolist()
            resultado[col] = {
                "índices": outliers,
                "Número de valores atípicos": len(outliers)
            }
        
        return resultado
    
    def normalidad_ks(self, alpha: float = 0.05) -> dict:
        """
        Evalúa la normalidad de columnas numéricas utilizando Kolmogorov-Smirnov,
        devolviendo resultados separados por normalidad.
        
        Parámetros:
        -----------
            alpha (float): Nivel de significancia (default: 0.05)
            
        Retorna:
        --------
            dict: Diccionario estructurado con {
                'normal': DataFrame de columnas normales,
                'no_normal': DataFrame de columnas no normales,
                'resumen': estadísticas globales
            }
        """
        resultados = []
        for col in self.clasificacion_variables['Cuantitativas']:
            stat, p = kstest(self.df[col], 'norm')
            resultados.append({
                'Columna': col,
                'Estadístico': stat,
                'p-value': p,
                'Normal?': p > alpha
            })
        
        df_resultados = pd.DataFrame(resultados)
        
        return {
            'normal': df_resultados[df_resultados['Normal?']],
            'no_normal': df_resultados[~df_resultados['Normal?']],
            'resumen': {
                'total_columnas': len(resultados),
                'porcentaje_normal': (df_resultados['Normal?'].mean() * 100),
                'mejor_ajuste': df_resultados.loc[df_resultados['p-value'].idxmax(), 'Columna']
            }
        }

    def generar_cuadro_cor(self, metodo: str = 'spearman') -> pd.DataFrame:
        """
        Genera una matriz de correlación para las variables numéricas del DataFrame.
        
        Parámetros:
        -----------
            metodo (str): Método de correlación a utilizar. Opciones disponibles:
                - 'pearson': Correlación lineal estándar (requiere normalidad)
                - 'spearman': Correlación por rangos (no paramétrica, default)
                - 'kendall': Correlación por concordancia (para muestras pequeñas)
                
        Retorna:
        --------
            pd.DataFrame: Matriz cuadrada de correlaciones con las siguientes características:
                - Índice y columnas con los nombres de las variables numéricas
                - Valores numéricos entre -1 y 1 representando la fuerza y dirección
                - Diagonal principal con valores 1 (autocorrelación)
                
        Ejemplo de retorno:
            │       │ Var1  │ Var2  │ Var3  │
            ├───────┼───────┼───────┼───────┤
            │ Var1  │ 1.000 │ 0.456 │ -0.210│
            │ Var2  │ 0.456 │ 1.000 │ 0.789 │
            │ Var3  │-0.210 │ 0.789 │ 1.000 │
                
        Notas:
        ------
        1. Para datos no normales, se recomienda 'spearman' o 'kendall'
        2. Los valores NaN son excluidos automáticamente del cálculo por columna
        3. La significancia estadística no es calculada (solo magnitudes)
        """
        return self.df.corr(method=metodo)
    
    def calcular_correlaciones(self, metodo: str = 'pearson') -> pd.DataFrame:
        '''Calcula un cuadro de correlaciones con las variables cuantitativas. 

        Parámetros:
        -----------
            metodo (str): Método de correlación a utilizar. Opciones disponibles:
                - 'pearson': Correlación lineal estándar (requiere normalidad)
                - 'spearman': Correlación por rangos (no paramétrica, default)
                - 'kendall': Correlación por concordancia (para muestras pequeñas)

        Retorna:
        --------
            pd.DataFrame: Cuadro de correlaciones
        '''
        return self.df[self.__clasificacion_variables['Cuantitativas']].corr()