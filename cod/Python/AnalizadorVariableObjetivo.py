import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, f_oneway
from analizador_data_frame.analizador_df import AnalizadorDataFrame

class AnalizadorVariableObjetivo:
    """Clase para análisis de relaciones entre variables y una variable objetivo binaria.
    
    Proporciona métodos para:
    - Calcular correlaciones entre variables predictoras y la variable objetivo
    - Medir la fuerza de relación entre variables y el objetivo
    - Analizar la distribución de la variable objetivo
    
    Atributos:
    ----------
    __y (pd.Series): Variable objetivo binaria (privada).
    __x (AnalizadorDataFrame): Instancia con variables predictoras (privada).
    """
    
    def __init__(self, df: pd.DataFrame, variable_objetivo: str):
        """Inicializa el analizador con un DataFrame y variable objetivo.
        
        Parámetros:
        -----------
            df (pd.DataFrame): DataFrame completo con los datos.
            variable_objetivo (str): Nombre de la columna binaria objetivo.
            
        Raises:
        -------
            ValueError: Si la variable objetivo no existe o no es binaria.
        """
        if variable_objetivo not in df.columns:
            raise ValueError(f"Variable objetivo '{variable_objetivo}' no encontrada en el DataFrame")
            
        if df[variable_objetivo].nunique() > 2:
            raise ValueError("La variable objetivo debe ser binaria")
            
        self.__y = df[variable_objetivo].copy()
        self.__x = AnalizadorDataFrame(df.drop(columns=[variable_objetivo]))
    
    # Getters para atributos privados
    @property
    def y(self):
        """Obtiene una copia de la variable objetivo.
        
        Retorna:
        --------
            pd.Series: Copia de la serie con la variable objetivo.
        """
        return self.__y.copy()
    
    @property
    def x(self):
        """Obtiene el analizador de variables predictoras.
        
        Retorna:
        --------
            AnalizadorDataFrame: Instancia con las variables predictoras.
        """
        return self.__x
    
    def eliminar_variables(self, variables: list):
        """Elimina variables especificadas del conjunto de predictores.
        
        Parámetros:
        -----------
            variables (list): Lista de nombres de columnas a eliminar.
        """
        current_df = self.__x.df
        updated_df = current_df.drop(columns=[v for v in variables if v in current_df.columns])
        self.__x = AnalizadorDataFrame(updated_df)
    
    def correlacion_objetivo(self) -> pd.DataFrame:
        """Calcula correlaciones entre variables cuantitativas y la objetivo.
        
        Retorna:
        --------
            pd.DataFrame: Resultados con:
                - 'Variable': Nombre de la variable predictora
                - 'Correlacion': Correlación de Pearson
                - 'p-value': Significancia estadística
                - 'Fuerza': Clasificación cualitativa de la fuerza
        """
        resultados = []
        
        for col in self.__x.clasificacion_variables['Cuantitativas']:
            # Eliminar pares con NA
            temp_df = pd.DataFrame({
                'x': self.__x.df[col],
                'y': self.__y
            }).dropna()
            
            if len(temp_df) > 0:
                corr, p_val = pointbiserialr(temp_df['x'], temp_df['y'])
                
                # Clasificar fuerza de correlación
                fuerza = self.__clasificar_fuerza_correlacion(abs(corr))
                
                resultados.append({
                    'Variable': col,
                    'Correlacion': corr,
                    'p-value': p_val,
                    'Fuerza': fuerza
                })
        
        return pd.DataFrame(resultados).sort_values('Correlacion', key=abs, ascending=False)
    
    def medir_relacion_objetivo(self) -> pd.DataFrame:
        """Mide la relación entre todas las variables y la objetivo.
        
        Para variables:
        - Cuantitativas: Usa Point-Biserial (equivalente a Pearson)
        - Cualitativas: Usa ANOVA (F-test)
        
        Retorna:
        --------
            pd.DataFrame: Resultados con:
                - 'Variable': Nombre de la variable
                - 'Estadistico': Valor del estadístico
                - 'p-value': Significancia
                - 'Fuerza': Clasificación cualitativa
        """
        resultados = []
        
        for col in self.__x.df.columns:
            temp_df = pd.DataFrame({
                'x': self.__x.df[col],
                'y': self.__y
            }).dropna()
            
            if len(temp_df) == 0:
                continue
                
            if col in self.__x.clasificacion_variables['Cuantitativas']:
                # Método para variables cuantitativas
                stat, p_val = pointbiserialr(temp_df['x'], temp_df['y'])
                fuerza = self.__clasificar_fuerza_correlacion(abs(stat))
                tipo = 'Point-Biserial'
            else:
                # Método para variables cualitativas
                groups = [g['y'].values for _, g in temp_df.groupby('x')]
                if len(groups) < 2:
                    continue
                    
                stat, p_val = f_oneway(*groups)
                fuerza = self.__clasificar_fuerza_anova(stat)
                tipo = 'ANOVA-F'
            
            resultados.append({
                'Variable': col,
                'Tipo': tipo,
                'Estadistico': stat,
                'p-value': p_val,
                'Fuerza': fuerza
            })
        
        return pd.DataFrame(resultados).sort_values('p-value')
    
    def distribucion_objetivo(self) -> dict:
        """Analiza la distribución de la variable objetivo.
        
        Retorna:
        --------
            dict: Diccionario con:
                - 'conteo': {
                    '0': conteo de ceros,
                    '1': conteo de unos,
                    'NA': conteo de nulos
                }
                - 'porcentajes': {
                    '0': porcentaje de ceros,
                    '1': porcentaje de unos,
                    'NA': porcentaje de nulos
                }
                - 'balance': Ratio entre clases (menor/mayor)
        """
        conteo = self.__y.value_counts(dropna=False)
        total = len(self.__y)
        
        resultado = {
            'conteo': {
                '0': conteo.get(0, 0),
                '1': conteo.get(1, 0),
                'NA': conteo.get(np.nan, 0)
            },
            'porcentajes': {
                '0': (conteo.get(0, 0) / total) * 100,
                '1': (conteo.get(1, 0) / total) * 100,
                'NA': (conteo.get(np.nan, 0) / total) * 100
            }
        }
        
        # Calcular balance (ignorando NAs)
        count_0 = resultado['conteo']['0']
        count_1 = resultado['conteo']['1']
        total_validos = count_0 + count_1
        
        if total_validos > 0:
            resultado['balance'] = min(count_0, count_1) / max(count_0, count_1)
        else:
            resultado['balance'] = np.nan
            
        return resultado
    
    # Métodos internos
    def __clasificar_fuerza_correlacion(self, r: float) -> str:
        """Clasifica la fuerza de una correlación.
        
        Parámetros:
        -----------
            r (float): Valor absoluto del coeficiente de correlación.
            
        Retorna:
        --------
            str: Clasificación cualitativa.
        """
        if r >= 0.7:
            return 'Muy Fuerte'
        elif r >= 0.5:
            return 'Fuerte'
        elif r >= 0.3:
            return 'Moderada'
        elif r >= 0.1:
            return 'Débil'
        else:
            return 'Muy Débil o Nula'
    
    def __clasificar_fuerza_anova(self, f: float) -> str:
        """Clasifica la fuerza basada en estadístico F de ANOVA.
        
        Parámetros:
        -----------
            f (float): Valor del estadístico F.
            
        Retorna:
        --------
            str: Clasificación cualitativa.
        """
        if f >= 10:
            return 'Muy Fuerte'
        elif f >= 5:
            return 'Fuerte'
        elif f >= 2:
            return 'Moderada'
        else:
            return 'Débil o Nula'