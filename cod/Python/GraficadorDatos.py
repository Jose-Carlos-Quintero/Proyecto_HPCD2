import matplotlib.pyplot as plt
import seaborn as sns

class GraficadorDatos:
    def __init__(self, df):
        """
        Inicializa una instancia del graficador con el DataFrame a utilizar.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con los datos que se desea visualizar.

        Retorna
        -------
        None
        """
        self.df = df.copy()
        
    @property
    def df(self):
        """
        Accede al DataFrame actual utilizado para las visualizaciones.

        Parámetros
        ----------
        None

        Retorna
        -------
        pd.DataFrame
        """
        return self.__df

    @df.setter
    def df(self, nuevo_df):
        """
        Asigna un nuevo DataFrame a la clase.

        Parámetros
        ----------
        nuevo_df : pd.DataFrame
            Nuevo DataFrame que se utilizará para las visualizaciones.

        Retorna
        -------
        None
        """
        self.__df = nuevo_df
        
    def __str__(self):
        """
        Retorna una representación textual del objeto con información resumida del DataFrame.

        Parámetros
        ----------
        None

        Retorna
        -------
        str
            Cadena con el número de filas y columnas del DataFrame.
        """
        return (f"ImportadorDatos desde '{self.__ruta}' con {self.__df.shape[0]} filas "
                f"y {self.__df.shape[1]} columnas.")


    def grafico_frecuencia_con_bono(self, columna, titulo = None):
        """
        Genera un gráfico de barras horizontales donde cada barra representa el porcentaje de observaciones
        en una categoría específica, y se anota el porcentaje de hogares con bono de vivienda (V21 = 1) dentro de cada categoría.

        Parámetros
        ----------
        columna : str
            Nombre de la variable categórica a graficar.
        titulo : str, optional
            Título personalizado del gráfico. Si no se proporciona, se genera automáticamente.

        Retorna
        -------
        None
        """
        df = self.df.copy()
        df['V21'] = df['V21'].astype(str)
    
        # Total de observaciones
        total = len(df)
    
        # % de personas en cada categoría respecto al total (longitud de la barra)
        freq_relativa = df[columna].value_counts(normalize = True).sort_index() * 100
    
        # % con bono (V21=1) dentro de cada categoría (etiqueta a mostrar)
        bono_por_categoria = (
            df[df['V21'] == "1"]
            .groupby(columna)
            .size()
            .div(df.groupby(columna).size())
            .fillna(0)
            .sort_index() * 100
        )
    
        # Plot
        plt.figure(figsize=(8, 5))
        sns.barplot(x = freq_relativa.values, y = freq_relativa.index, palette  =  'viridis')
        for i, (cat, v) in enumerate(bono_por_categoria.items()):
            plt.text(freq_relativa[cat] + 0.5, i, f"{v:.1f}%", va = 'center')
    
        plt.title(titulo or f"Distribución de {columna} y % con bono (V21)")
        plt.xlabel("Porcentaje del total")
        plt.ylabel(columna)
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.show()

    def grafico_caja_ingreso(self, variable_categorica):
        """
        Genera un gráfico de caja (boxplot) del ingreso (columna `ithb`) para cada categoría de la variable especificada,
        excluyendo valores extremos (outliers) mediante la regla de Tukey (±1.5·IQR).

        Parámetros
        ----------
        variable_categorica : str
            Nombre de la variable categórica por la cual segmentar el ingreso.

        Retorna
        -------
        None
        """
        df_filtrado = self.df.copy()
    
        # Calcular los cuartiles y IQR por categoría
        def quitar_outliers(grupo):
            q1 = grupo['ithb'].quantile(0.25)
            q3 = grupo['ithb'].quantile(0.75)
            iqr = q3 - q1
            filtro = (grupo['ithb'] >= q1 - 1.5 * iqr) & (grupo['ithb'] <= q3 + 1.5 * iqr)
            return grupo[filtro]
    
        df_filtrado = df_filtrado.groupby(variable_categorica, group_keys = False).apply(quitar_outliers)
    
        # Graficar sin valores extremos
        plt.figure(figsize = (8, 5))
        sns.boxplot(data = df_filtrado, x = variable_categorica, y = 'ithb', palette = 'Set2')
        plt.xticks(rotation = 45)
        plt.title(f"Ingreso por {variable_categorica} (sin outliers)")
        plt.tight_layout()
        plt.show()

    def graficar_todas(self):
        """
        Genera una serie de gráficos estándar que combinan análisis de frecuencia y visualización de ingresos.
        Incluye:
        - Gráficos de barras con frecuencia relativa y porcentaje con bono para:
          'ZONA', 'REGION', 'CondMig', 'V3', 'V15', 'V8'.
        - Boxplots del ingreso 'ithb' segmentado por:
          'CondMig', 'REGION', 'ZONA'.

        Parámetros
        ----------
        None

        Retorna
        -------
        None
        """
        self.grafico_frecuencia_con_bono("ZONA", "Distribución por Zona")
        self.grafico_frecuencia_con_bono("REGION", "Distribución por Región")
        self.grafico_frecuencia_con_bono("CondMig", "Condición Migrante")
        self.grafico_frecuencia_con_bono("V3", "Tipo de Pared")
        self.grafico_frecuencia_con_bono("V15", "Fuente de Electricidad")
        self.grafico_frecuencia_con_bono("V8", "Cantidad de cuartos para dormir de la vivienda")
        self.grafico_caja_ingreso("CondMig")
        self.grafico_caja_ingreso("REGION")
        self.grafico_caja_ingreso("ZONA")
