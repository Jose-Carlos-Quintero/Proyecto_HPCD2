#  Script dedicado a la limpieza de la base de datos de la ENAHO 2024

from ImportadorDatos import ImportadorDatos

enaho = ImportadorDatos("../data/enaho_2024.sav")

# Se seleccionan las variables demográficas y del hogar de la base de datos que, a simple vista, podrían tener sentido en el modelo

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

# Se le agregan las etiquetas a las variables
enaho.factorizar_todo()

#Se cuentan las variables que tienen más cantidad de valores na
enaho.contar_na_por_columna()

# Se seleccionan las variables con menor cantidad de datos nulos y las que se considera que podría tener importancia
enaho.eliminar_columnas(["V18A1", "V18F1", "V18G1", "V18Q1", "V18I1", "V18J1", "V18K1", "V18L1", "V18L2", "V18M1", "V19B1", "V19B2", "V19B3", "V19B4", ])

# Se eliminan los valores nulos
enaho.eliminar_na_columnas()

# Se revisan variables para ver cuales agrupar

enaho.resumen_categoria("V1")
enaho.refactorizar_variable("V1", {
    "En edificio (condominio vertical o apartamento)": "Otra",
    "Casa en condominio o residencial cerrado": "Otra",
    "Tugurio": "Otra",
    "Cuartería": "Otra",
    "Otro": "Otra"})

enaho.resumen_categoria("V2A")
enaho.refactorizar_variable("V2A", {
    "Propia totalmente pagada": "Propiedad",
    "Propia pagando a plazos": "Propiedad",
    "Alquilada": "Alquiler o cesión",
    "Otra tenencia (cedida, prestada)": "Alquiler o cesión",
    "En precario": "Tenencia precaria"})

enaho.resumen_categoria("V3")
enaho.refactorizar_variable("V3", {
    "Fibrocemento (Fibrolit, Ricalit)": "Otro",
    "Zinc": "Otro",
    "Material de desecho": "Otro",
    "Fibras naturales (bambú, caña, chonta)": "Otro",
    "Otro": "Otro"})

enaho.resumen_categoria("V4")
enaho.refactorizar_variable("V4", {
    "Entrepiso": "Otro",
    "Fibrocemento": "Otro",
    "Material de desecho": "Otro",
    "Fibras naturales (bambú, caña, chonta)": "Otro",
    "Otro": "Otro"})

enaho.resumen_categoria("V6")
enaho.refactorizar_variable("V6", {
    "Madera": "Otro",
    "No tiene (piso de tierra)": "Otro",
    "Otro": "Otro",
    "Material natural (bambú, caña, chonta)": "Otro"})

enaho.resumen_categoria("V8")
enaho.refactorizar_variable("V8", {
    1.0: "1 o ninguno",
    "Ninguno": "1 o ninguno",
    5.0: "5 o más",
    6.0: "5 o más",
    7.0: "5 o más",
    8.0: "5 o más",
    10.0: "5 o más"})

enaho.eliminar_columnas("HacApo") # Variable muy desbalanceada

enaho.resumen_categoria("V11")
enaho.refactorizar_variable("V11", {
    "Tubería fuera de la vivienda pero dentro del lote": "Otra ubicación",
    "No tiene por tubería": "Otra ubicación",
    "Tubería fuera del lote o edificio": "Otra ubicación"})

enaho.resumen_categoria("V12")
enaho.refactorizar_variable("V12", {
    "Empresa o cooperativa": "Otro",
    "Pozo": "Otro",
    "Río, quebrada o naciente": "Otro",
    "Lluvia u otro": "Otro"})

enaho.resumen_categoria("V13A")
enaho.refactorizar_variable("V13A", {
    "Conectado a tanque séptico con tratamiento (fosa filtro, biodigestor, etc.)": "Otro",
    "De hueco, de pozo negro o letrina": "Otro",
    "Otro sistema": "Otro",
    "No tiene": "Otro"
})

enaho.eliminar_columnas("V13B") # Variable muy desbalanceada
enaho.eliminar_columnas("V14A") # Variable muy desbalanceada

enaho.resumen_categoria("V14A1")
enaho.refactorizar_variable("V14A1", {
    3.0: "3 o más",
    4.0: "3 o más",
    5.0: "3 o más",
    6.0: "3 o más"
})

enaho.eliminar_columnas("V14B") # Variable muy desbalanceada

enaho.resumen_categoria("V15")
enaho.refactorizar_variable("V15", {
    "De la ESPH / JASEC": "Otra fuente o sin electricidad",
    "No hay luz eléctrica": "Otra fuente o sin electricidad",
    "Otra fuente de energía": "Otra fuente o sin electricidad",
    "De planta privada": "Otra fuente o sin electricidad"
})

enaho.resumen_categoria("V16")
enaho.refactorizar_variable("V16", {
    "Leña o carbón": "Otro o ninguno",
    "Ninguno (no cocina)": "Otro o ninguno"
})

enaho.resumen_categoria("V17A")
enaho.refactorizar_variable("V17A", {
    "La queman": "Otro método",
    "La botan en hueco o entierran": "Otro método",
    "La botan en lote baldío": "Otro método",
    "Otro": "Otro método"
})

enaho.eliminar_columnas("V17B6") # Variable muy desbalanceada

enaho.eliminar_columnas("V18A") # Variable muy desbalanceada

enaho.eliminar_columnas("V18H") # Variable muy desbalanceada y obsoleta

enaho.resumen_categoria("TamViv")
enaho.refactorizar_variable("TamViv", {
    6.0: "6 o más",
    7.0: "6 o más",
    8.0: "6 o más",
    9.0: "6 o más",
    10.0: "6 o más",
    11.0: "6 o más",
    12.0: "6 o más",
    13.0: "6 o más",
    14.0: "6 o más",
    15.0: "6 o más"
})

enaho.resumen_categoria("A5")
enaho.refactorizar_variable("A5", {
    "Menos de 1 año": "0 a 4",
    1.0: "0 a 4",
    2.0: "0 a 4",
    3.0: "0 a 4",
    4.0: "0 a 4",
    5.0: "5 a 12",
    6.0: "5 a 12",
    7.0: "5 a 12",
    8.0: "5 a 12",
    9.0: "5 a 12",
    10.0: "5 a 12",
    11.0: "5 a 12",
    12.0: "5 a 12",
    13.0: "13 a 17",
    14.0: "13 a 17",
    15.0: "13 a 17",
    16.0: "13 a 17",
    17.0: "13 a 17",
    18.0: "18 a 24",
    19.0: "18 a 24",
    20.0: "18 a 24",
    21.0: "18 a 24",
    22.0: "18 a 24",
    23.0: "18 a 24",
    24.0: "18 a 24",
    25.0: "25 a 34",
    26.0: "25 a 34",
    27.0: "25 a 34",
    28.0: "25 a 34",
    29.0: "25 a 34",
    30.0: "25 a 34",
    31.0: "25 a 34",
    32.0: "25 a 34",
    33.0: "25 a 34",
    34.0: "25 a 34",
    35.0: "35 a 44",
    36.0: "35 a 44",
    37.0: "35 a 44",
    38.0: "35 a 44",
    39.0: "35 a 44",
    40.0: "35 a 44",
    41.0: "35 a 44",
    42.0: "35 a 44",
    43.0: "35 a 44",
    44.0: "35 a 44",
    45.0: "45 a 59",
    46.0: "45 a 59",
    47.0: "45 a 59",
    48.0: "45 a 59",
    49.0: "45 a 59",
    50.0: "45 a 59",
    51.0: "45 a 59",
    52.0: "45 a 59",
    53.0: "45 a 59",
    54.0: "45 a 59",
    55.0: "45 a 59",
    56.0: "45 a 59",
    57.0: "45 a 59",
    58.0: "45 a 59",
    59.0: "45 a 59",
    60.0: "60 a 74",
    61.0: "60 a 74",
    62.0: "60 a 74",
    63.0: "60 a 74",
    64.0: "60 a 74",
    65.0: "60 a 74",
    66.0: "60 a 74",
    67.0: "60 a 74",
    68.0: "60 a 74",
    69.0: "60 a 74",
    70.0: "60 a 74",
    71.0: "60 a 74",
    72.0: "60 a 74",
    73.0: "60 a 74",
    74.0: "60 a 74",
    75.0: "75 o más",
    76.0: "75 o más",
    77.0: "75 o más",
    78.0: "75 o más",
    79.0: "75 o más",
    80.0: "75 o más",
    81.0: "75 o más",
    82.0: "75 o más",
    83.0: "75 o más",
    84.0: "75 o más",
    85.0: "75 o más",
    86.0: "75 o más",
    87.0: "75 o más",
    88.0: "75 o más",
    89.0: "75 o más",
    90.0: "75 o más",
    91.0: "75 o más",
    92.0: "75 o más",
    93.0: "75 o más",
    94.0: "75 o más",
    95.0: "75 o más",
    96.0: "75 o más",
    "97 años o más": "75 o más"
})

enaho.resumen_categoria("A6")
enaho.refactorizar_variable("A6", {
    "Separado(a)": "Separado/Divorciado/Viudo",
    "Divorciado(a)": "Separado/Divorciado/Viudo",
    "Viudo(a)": "Separado/Divorciado/Viudo"
})

enaho.resumen_categoria("NivInst")
enaho.refactorizar_variable("NivInst", {
    "Primaria incompleta": "Sin nivel de instrucción",
    "Ignorado": "Sin nivel de instrucción",
    "Primaria completa": "Primaria completa",
    "Secundaria académica incompleta": "Primaria completa",
    "Secundaria técnica incompleta": "Primaria completa",
    "Secundaria académica completa": "Secundaria completa",
    "Secundaria técnica completa": "Secundaria completa",
    "Educación superior de pregrado y grado": "Superior",
    "Educación superior de posgrado": "Superior"
})

enaho.codificar_binaria("V21", "Sí", "No")

# se revisa el analisis exploratorio con las variables recodificadas y se guarda la nueva base en un archivo .csv
enaho.generar_eda_html("V21")

enaho.generar_csv()