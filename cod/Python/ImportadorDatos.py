import pandas as pd
import pyreadstat

class ImportadorDatos:
    """
    Clase para importar y transformar datos desde un archivo SPSS (.sav).
    """

    def __init__(self, ruta_archivo: str):
        """
        Inicializa una instancia del importador con la ruta al archivo .sav.

        Parámetros
        ----------
        ruta_archivo : str
            Ruta al archivo SPSS (.sav) que se desea importar.
            
        Retorna
        -------
        None
        """
        self.__ruta = ruta_archivo
        self.__datos = None
        self.__etiquetas = None

        self.__datos, self.__etiquetas = pyreadstat.read_sav(self.__ruta)

    @property
    def datos(self):
        """Accede al DataFrame de datos importados.
        Parámetros
        ----------
        ruta_archivo : str
            Ruta al archivo SPSS (.sav) que se desea importar.
            
        Retorna
        -------
        pd.DataFrame
            dataframe con los datos
        """
        return self.__datos

    @datos.setter
    def datos(self, nuevo_df):
        """Asigna un nuevo DataFrame a los datos importados.
        
        Parámetros
        ----------
        nuevo_df : pd.DataFrame
            Nuevo dataframe que se quiere importar
            
        Retorna
        -------
        None
        """
        self.__datos = nuevo_df

    @property
    def etiquetas(self):
        """Accede al diccionario de etiquetas importadas desde el .sav.
        
        Parámetros
        ----------
        None
            
        Retorna
        -------
        pyreadstat.MetadataContainer
            Contenedor de datos del dataframe
        """
        return self.__etiquetas

    @etiquetas.setter
    def etiquetas(self, nuevas_etiquetas):
        """Asigna nuevas etiquetas.
        
        Parámetros
        ----------
        nuevas_etiquetas: pyreadstat.MetadataContainer
            nuevas etiquetas del df
            
        Retorna
        -------
        None
        
        """
        self.__etiquetas = nuevas_etiquetas

    @property
    def ruta(self):
        """Accede a la ruta del archivo importado.
        
        Parámetros
        ----------
        None
            
        Retorna
        -------
        str
            ruta del archivo de la base de datos
        
        """
        return self.__ruta

    @ruta.setter
    def ruta(self, nueva_ruta: str):
        """Asigna una nueva ruta y recarga los datos.
        
        Parámetros
        ----------
        nueva_ruta: str
            nueva ruta al archivo de la base de datos
            
        Retorna
        -------
        None
        """
        self.__ruta = nueva_ruta
        self.__datos, self.__etiquetas = pyreadstat.read_sav(self.__ruta)

    def __str__(self):
        """
        Retorna una representación textual resumida del objeto.
        
        Parámetros
        ----------
        None
            
        Retorna
        -------
        str
        """
        return (f"ImportadorDatos desde '{self.__ruta}' con {self.__datos.shape[0]} filas "
                f"y {self.__datos.shape[1]} columnas.")

    def seleccionar_variables(self, lista_vars: list):
        """
        Filtra el DataFrame dejando solo las variables indicadas.

        Parámetros
        ----------
        lista_vars : list
            Lista de nombres de variables a conservar.
            
        Retorna
        -------
        pd.DataFrame
            dataframe con las variables seleccionadas
        """
        self.__datos = self.__datos[lista_vars]
        
        return self.__datos

    def factorizar_todo(self):
        """
        Reemplaza los códigos numéricos por sus etiquetas y convierte las columnas a tipo categoría
        para todas las variables que tengan etiquetas definidas en el archivo .sav.
        
        Parámetros
        ----------
        None
            
        Retorna
        -------
        pd.DataFrame
            Dataframe con las variables factorizadas
        """
        etiquetas = self.__etiquetas.variable_value_labels

        for col in self.__datos.columns:
            if col in etiquetas:
                self.__datos[col] = self.__datos[col].replace(etiquetas[col])
                self.__datos[col] = self.__datos[col].astype("category")
        return self.__datos 
                
    def eliminar_na_columnas(self, columnas = None):
        """
        Elimina las filas del DataFrame que tienen valores NA en al menos una de las columnas especificadas.
    
        Parámetros
        ----------
        columnas : list
            Lista de nombres de columnas en las cuales se eliminarán filas con valores NA.
            
        Retorna
        -------
        pd.DataFrame
            Dataframe con las observaciones NA filtradas
        """
        if columnas is None:
            columnas = self.__datos.columns
        self.__datos = self.__datos.dropna(subset=columnas)
        
        return self.__datos
        
    def contar_na_por_columna(self):
        """
        Retorna un Series con la cantidad de valores NA por cada columna del DataFrame.
        
        Parámetros
        ----------
        None

        Retorna
        -------
        pd.Series
            Serie con el nombre de la columna como índice y la cantidad de NA como valor.
        """
        return self.__datos.isna().sum()
    
    def refactorizar_variable(self, columna: str, recodificacion: dict):
        """
        Refactoriza una variable categórica según un diccionario de recodificación.
    
        Parámetros
        ----------
        columna : str
            Nombre de la variable a refactorizar.
        recodificacion : dict
            Diccionario con los valores a transformar. Formato: {valor_actual: nuevo_valor}
            
        Retorna
        -------
        pd.DataFrame
            Dataframe con las variables especificadas refactorizadas
        """
        if columna not in self.__datos.columns:
            raise ValueError(f"La columna '{columna}' no existe en los datos.")
    
        self.__datos[columna] = self.__datos[columna].replace(recodificacion)
        self.__datos[columna] = self.__datos[columna].astype("category")

        return self.__datos

    def eliminar_columnas(self, columnas: list):
        """
        Elimina del DataFrame las columnas especificadas.

        Parámetros
        ----------
        columnas : list
            Lista de nombres de columnas que se desea eliminar.

        Retorna
        -------
        pd.DataFrame
            DataFrame actualizado sin las columnas eliminadas.
        """
        self.__datos = self.__datos.drop(columns = columnas, errors = "ignore")
        return self.__datos

    def resumen_categoria(self, variable: str):
        """
        Retorna una tabla con la frecuencia absoluta y relativa de una variable categórica.

        Parámetros
        ----------
        variable : str
            Nombre de la variable categórica que se desea resumir.

        Retorna
        -------
        resumen: pd.DataFrame
            Tabla con las categorías, frecuencias absolutas y relativas.
        """
        conteo = self.__datos[variable].value_counts(dropna=False)
        porcentaje = self.__datos[variable].value_counts(normalize=True, dropna=False).round(4)

        resumen = pd.DataFrame({
            "Categoria": conteo.index,
            "Frecuencia absoluta": conteo.values,
            "Frecuencia relativa": porcentaje.values
        })

        return resumen


