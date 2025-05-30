import pandas as pd
import pyreadstat

class ImportadorDatos:
    """
    Clase para importar y transformar datos desde un archivo SPSS (.sav).
    """

    def __init__(self, ruta_archivo):
        """
        Inicializa una instancia del importador con la ruta al archivo .sav.

        Parámetros
        ----------
        ruta_archivo : str
            Ruta al archivo SPSS (.sav) que se desea importar.
        """
        self.__ruta = ruta_archivo
        self.__datos = None
        self.__etiquetas = None

        self.__datos, self.__etiquetas = pyreadstat.read_sav(self.__ruta)

    @property
    def datos(self):
        """Accede al DataFrame de datos importados."""
        return self.__datos

    @datos.setter
    def datos(self, nuevo_df):
        """Asigna un nuevo DataFrame a los datos importados."""
        self.__datos = nuevo_df

    @property
    def etiquetas(self):
        """Accede al diccionario de etiquetas importadas desde el .sav."""
        return self.__etiquetas

    @etiquetas.setter
    def etiquetas(self, nuevas_etiquetas):
        """Asigna nuevas etiquetas."""
        self.__etiquetas = nuevas_etiquetas

    @property
    def ruta(self):
        """Accede a la ruta del archivo importado."""
        return self.__ruta

    @ruta.setter
    def ruta(self, nueva_ruta):
        """Asigna una nueva ruta y recarga los datos."""
        self.__ruta = nueva_ruta
        self.__datos, self.__etiquetas = pyreadstat.read_sav(self.__ruta)

    def __str__(self):
        """
        Retorna una representación textual resumida del objeto.

        Retorna
        -------
        str
        """
        return (f"ImportadorDatos desde '{self.__ruta}' con {self.__datos.shape[0]} filas "
                f"y {self.__datos.shape[1]} columnas.")

    def resumen_general(self):
        """Imprime dimensiones, tipos de variables y primeras filas del DataFrame."""
        print("Dimensiones:", self.__datos.shape)
        print("\nTipos de variables:")
        print(self.__datos.dtypes.value_counts())
        print("\nPrimeras filas:")
        print(self.__datos.head())

    def seleccionar_variables(self, lista_vars):
        """
        Filtra el DataFrame dejando solo las variables indicadas.

        Parámetros
        ----------
        lista_vars : list
            Lista de nombres de variables a conservar.
        """
        self.__datos = self.__datos[lista_vars]

    def factorizar_todo(self):
        """
        Reemplaza los códigos numéricos por sus etiquetas y convierte las columnas a tipo categoría
        para todas las variables que tengan etiquetas definidas en el archivo .sav.
        """
        etiquetas = self.__etiquetas.variable_value_labels

        for col in self.__datos.columns:
            if col in etiquetas:
                self.__datos[col] = self.__datos[col].replace(etiquetas[col])
                self.__datos[col] = self.__datos[col].astype("category")
