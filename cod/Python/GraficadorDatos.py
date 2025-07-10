import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
        self.__df = df.copy()
        
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
        return self.____df

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
        return (f"ImportadorDatos desde '{self.____ruta}' con {self.__df.shape[0]} filas "
                f"y {self.__df.shape[1]} columnas.")


    def grafico_frecuencia_con_bono(self, columna, titulo = None):
        """
        Genera un gráfico de barras horizontales donde cada barra representa 
        la cantidad (en miles) de observaciones en una categoría específica, y se anota 
        el porcentaje de hogares con bono de vivienda (V21 = 1) dentro de cada categoría.
    
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
        df = self.__df.copy()
        df['V21'] = df['V21'].astype(str)
    
        # Frecuencia absoluta ordenada de mayor a menor
        freq_absoluta = df[columna].value_counts().sort_values(ascending = False)
    
        # Porcentaje con bono por categoría
        bono_por_categoria = (
            df[df['V21'] == "1"]
            .groupby(columna)
            .size()
            .div(df.groupby(columna).size())
            .fillna(0) * 100
        )
    
        bono_por_categoria = bono_por_categoria.reindex(freq_absoluta.index)
    
        # Dividir los valores por 1000 para que el gráfico muestre miles
        valores_miles = freq_absoluta / 1000
    
        plt.figure(figsize = (10, 5))
        sns.barplot(x = valores_miles.values, y = valores_miles.index, palette = 'viridis')
    
        for i, (cat, v) in enumerate(bono_por_categoria.items()):
            plt.text(valores_miles[cat] + 0.1, i, f"{v:.1f}%", va = 'center', fontsize = 20)
    
        plt.title(titulo or f"Distribución de {columna} y % con bono (V21)", fontsize = 20)
        plt.xlabel("Cantidad de observaciones (en miles)", fontsize = 20)
        plt.ylabel(columna, fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.tight_layout()
        # plt.subplots_adjust(left=0.5)
        plt.xlim(0, valores_miles.max() * 1.1)
        plt.show()


    def grafico_caja_ingreso(self, variable_categorica: str, titulo: str = None):
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
        df_filtrado = self.__df.copy()
        df_filtrado["ithb"] = df_filtrado["ithb"] / 1_000_000

    
        # Calcular los cuartiles y IQR por categoría
        def quitar_outliers(grupo):
            q1 = grupo['ithb'].quantile(0.25)
            q3 = grupo['ithb'].quantile(0.75)
            iqr = q3 - q1
            filtro = (grupo['ithb'] >= q1 - 1.5 * iqr) & (grupo['ithb'] <= q3 + 1.5 * iqr)
            return grupo[filtro]
    
        df_filtrado = df_filtrado.groupby(variable_categorica, group_keys = False).apply(quitar_outliers)
    
        # Graficar sin valores extremos
        plt.figure(figsize = (10, 5))
        sns.boxplot(
            data = df_filtrado,
            y = variable_categorica,
            x = 'ithb',
            palette = "viridis",
            boxprops = dict(edgecolor = 'black'),
            whiskerprops = dict(color = 'black'),
            capprops = dict(color = 'black'),
            medianprops = dict(color = 'black'),
            flierprops = dict(markerfacecolor = 'gray', markeredgecolor = 'black')
        )
        plt.xlabel("Ingreso total del hogar bruto (millones de colones)", fontsize = 20)
        plt.ylabel(variable_categorica, fontsize = 20)
        plt.title(titulo or f"Ingreso por {variable_categorica} (sin outliers)", fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
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
