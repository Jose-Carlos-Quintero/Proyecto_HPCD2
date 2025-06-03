# -*- coding: utf-8 -*-
"""
Created on Thu May 29 21:32:16 2025

@author: José Carlos
"""

from ImportadorDatos import ImportadorDatos

enaho = ImportadorDatos("../data/enaho_2024.sav")

enaho.seleccionar_variables([
  "ZONA", "REGION", "V1", "V2A", "V3", "V4", "V5", "V6", "V6A", "V7A", "V7B", "V7C", "EFI",
  "V8", "HacApo", "V11", "V12", "V13A", "V13B", "V14A", "V14A1", "V14B", "V15", "V16",
  "V17A", "V17B1", "V17B2", "V17B3", "V17B4", "V17B5", "V17B6", "V18A", "V18A1", "V18C",
  "V18D", "V18E", "V18F", "V18F1", "V18G", "V18G1", "V18Q", "V18Q1", "V18H", "V18I",
  "V18I1", "V18J", "V18J1", "V18K", "V18K1", "V18L", "V18L1", "V18L2", "V18M", "V18M1",
  "V22", "V19", "V19B1", "V19B2", "V19B3", "V19B4", "V21", "R4A", "TamViv", "A4", "A5",
  "A6", "CondMig", "NivInst", "A22A"
]
)

enaho.factorizar_todo()

enaho.resumen_general()

# enaho.eliminar_na_columnas()

enaho.resumen_general()

y = enaho.contar_na_por_columna()

x = enaho.datos
