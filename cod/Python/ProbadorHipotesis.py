import pandas as pd
from scipy.stats import kstest  # Prueba Kolmogorov-Smirnov

class ProbadorHipotesis:

    def __init__(self, dataframe: pd.DataFrame):
        self.__dataframe = dataframe

    def normalidad_ks(self, alpha: float = 0.05):
        '''Evalúa la normalidad de columnas numéricas en un DataFrame utilizando la prueba de Kolmogorov-Smirnov.
        
        Parámetros:
        -----------
        dataframe (pd.DataFrame): DataFrame con las columnas numéricas a evaluar.
        alpha (float): Nivel de significancia (alpha = 0.05)

        Returns:
        --------
        resultados (pd.DataFrame): Resultados de la prueba para cada columna.
        '''
        resultados = []
        for col in self.__dataframe.select_dtypes(include=['float64', 'int64']).columns:
            stat, p = kstest(self.__dataframe[col], 'norm')
            resultados.append({
                'Columna': col,
                'Estadístico': stat,
                'p-value': p,
                'Normal?': p > alpha
            })
        return pd.DataFrame(resultados)

    def generar_cuadro_cor(self, metodo: str = 'spearman'):
        return self.__dataframe.corr(method = metodo)