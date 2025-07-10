from GraficadorDatos import GraficadorDatos
import pandas as pd

base=pd.read_csv("../../data/base.csv")
vis = GraficadorDatos(base)
# vis.graficar_todas()

vis.grafico_frecuencia_con_bono('V3', "Distribución del material de las paredes (V3) y % con bono")
vis.grafico_frecuencia_con_bono('V15', "Distribución de la fuente de energía (V15) y % con bono")
vis.grafico_frecuencia_con_bono('REGION', "Distribución de la región y % con bono")
vis.grafico_frecuencia_con_bono('TamViv', "Distribución del número de integrantes (TamViv) y % con bono")
vis.grafico_frecuencia_con_bono('R4A', "Distribución de la jefatura compartida (R4A) y % con bono")

vis.grafico_caja_ingreso("V22", "Distribución del ingreso según si se recibió o no el bono de vivienda (V22)")
