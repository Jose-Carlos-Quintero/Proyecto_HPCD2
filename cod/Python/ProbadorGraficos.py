from GraficadorDatos import GraficadorDatos
import pandas as pd

base=pd.read_csv("~/Library/CloudStorage/OneDrive-UniversidaddeCostaRica/EMat/CA-0305/Proyecto Grupal/Proyecto_HPCD2/data/base.csv")
vis = GraficadorDatos(base)
vis.graficar_todas()

vis.grafico_barras('V15')
