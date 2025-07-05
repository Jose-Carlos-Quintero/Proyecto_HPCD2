import matplotlib.pyplot as plt
import seaborn as sns

class GraficadorDatos:
    def __init__(self, df):
        self.df = df.copy()

    def grafico_barras(self, columna, titulo=None):
      
        # Calcular frecuencia relativa de bono por categoría
        df_agg = self.df.groupby(columna).agg(
            total=('V21', 'count'),
            con_bono=('V21', lambda x: (x == 1).sum())
        )
        df_agg['porcentaje_bono'] = (df_agg['con_bono'] / df_agg['total']) * 100
        df_agg = df_agg.reset_index().sort_values(by='total', ascending=False)
    
        # Plot
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(data=self.df, x=columna, order=df_agg[columna], palette='viridis')
        plt.xticks(rotation=45)
        plt.title(titulo or f"Distribución de {columna} con % de bono (V21)")
    
        # Anotar porcentajes arriba de cada barra
        for bar, pct in zip(ax.patches, df_agg['porcentaje_bono']):
            height = bar.get_height()
            ax.annotate(f"{pct:.1f}%", (bar.get_x() + bar.get_width()/2, height),
                        ha='center', va='bottom', fontsize=10, color='black')
    
        plt.tight_layout()
        plt.show()

    def grafico_caja_ingreso(self, variable_categorica):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, x=variable_categorica, y='ithb', palette='Set2')
        plt.xticks(rotation=45)
        plt.title(f"Ingreso por {variable_categorica}")
        plt.tight_layout()
        plt.show()

    def graficar_todas(self):
        self.grafico_barras("ZONA", "Distribución por Zona")
        self.grafico_barras("REGION", "Distribución por Región")
        self.grafico_barras("CondMig", "Condición Migrante")
        self.grafico_barras("V3", "Tipo de Pared")
        self.grafico_barras("V15", "Fuente de Electricidad")
        self.grafico_barras("V8", "Categoría de V8")
        self.grafico_caja_ingreso("CondMig")
        self.grafico_caja_ingreso("REGION")
        self.grafico_caja_ingreso("ZONA")
