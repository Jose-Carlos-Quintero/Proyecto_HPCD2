from GraficadorDatos import GraficadorDatos
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv("../../data/base.csv")
vis = GraficadorDatos(base)
vis.codificar_binaria_a_categorica("V21")
# vis.graficar_todas()

vis.grafico_frecuencia_con_bono('V3', "Distribución del material de las paredes (V3) y % con bono")
plt.savefig("../../res/graphs/Distribución del material de las paredes y % con bono.png", dpi=300)
vis.grafico_frecuencia_con_bono('V15', "Distribución de la fuente de energía (V15) y % con bono")
plt.savefig("../../res/graphs/Distribución de la fuente de energía (V15) y % con bono.png", dpi=300)
vis.grafico_frecuencia_con_bono('REGION', "Distribución de la región y % con bono")
plt.savefig("../../res/graphs/Distribución de la región y % con bono.png", dpi=300)
vis.grafico_frecuencia_con_bono('TamViv', "Distribución del número de integrantes (TamViv) y % con bono")
plt.savefig("../../res/graphs/Distribución del número de integrantes y % con bono.png", dpi=300)
vis.grafico_frecuencia_con_bono('R4A', "Distribución de la jefatura compartida (R4A) y % con bono")
plt.savefig("../../res/graphs/Distribución de la jefatura compartida y % con bono.png", dpi=300)

vis.grafico_caja_ingreso("V21", "Distribución del ingreso según si se recibió o no el bono de vivienda (V21)")
plt.savefig("../../res/graphs/Distribución del ingreso según si se recibió o no el bono de vivienda.png", dpi=300)
