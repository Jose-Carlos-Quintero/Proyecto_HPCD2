# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:16:48 2025

@author: Federico Alberto Alfaro Chaverri
"""

import matplotlib.pyplot as plt # Gráficos
import numpy as np  # Funciones matemáticas
import pandas as pd  # Manejo de dataframes
#import os  # Rutas de los archivos
#from scipy.cluster.hierarchy import dendrogram, fcluster, linkage # Clusterin Jerarquico
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, 
    f1_score, precision_score, recall_score, 
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from AnalizadorVariableObjetivo import AnalizadorVariableObjetivo
# Eliminar evaluación de clustering
# Eliminar jerárquico


class ModeladorDatos():
    """Clase para manejar modelos de machine learning heredando de AnalizadorDataFrame.
    
    Realiza preprocesamiento automático de datos (escalado numérico y encoding categórico)
    y gestión de conjuntos de entrenamiento/prueba.
    """
    def __init__(self, df: pd.DataFrame, 
                 var_objetivo : str,
                 modelos : dict, 
                 test_size : float = 0.3, 
                 handle_unknown : str = 'ignore', 
                 sparse_output : bool = False,
                 remainder : str = 'passthrough'):
        """Inicializa el modelo con un DataFrame, configuraciones de preprocesamiento y modelos a evaluar.

        Parámetros
        ----------
        df : pd.DataFrame
            Conjunto de datos donde una de las columnas corresponde a la variable objetivo.
        var_objetivo : str
            Nombre de la variable objetivo dentro del DataFrame.
        modelos : dict
            Diccionario de modelos a evaluar (por ejemplo, {'log_reg': LogisticRegression()}).
        test_size : float, opcional
            Proporción del conjunto de prueba respecto al total (por defecto es 0.3).
        handle_unknown : str, opcional
            Comportamiento para categorías desconocidas en OneHotEncoder (por defecto es 'ignore').
        sparse_output : bool, opcional
            Si es True, OneHotEncoder devuelve una matriz dispersa (por defecto es False).
        remainder : str, opcional
            Indica qué hacer con las columnas no transformadas (por defecto es 'passthrough').
        
        Retorna
        -------
        None
        """

        self.__Y = df[var_objetivo].copy()
        self.__X = df.copy().drop(columns = [var_objetivo])
        self.__numerical_transformer = StandardScaler() 
        self.__categorical_transformer = OneHotEncoder(handle_unknown = handle_unknown, sparse_output = sparse_output)
        self.__test_size = test_size
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = (
            train_test_split(self.__X,self.__Y,test_size = test_size,random_state = 42)
        )
        self.__modelos = modelos
        self.__analizador = AnalizadorVariableObjetivo(pd.concat([self.__X_train, self.__y_train], axis = 1), 
                                                       var_objetivo)
        self.__procesador = self.__configurar_preprocesador(remainder)
        self.__X_escalado = self.__numerical_transformer.fit_transform(
            self.__X[self.__analizador.x.clasificacion_variables['Cuantitativas']]
            )


    #Métodos internos
    def __configurar_preprocesador(self, remainder: str) -> ColumnTransformer:
        """Configura el ColumnTransformer con los preprocesadores apropiados según el tipo de variable.

        Parámetros
        ----------
        remainder : str
            Indica qué hacer con las columnas no transformadas (por ejemplo, 'drop' o 'passthrough').
        
        Retorna
        -------
        ColumnTransformer:
            Objeto configurado que transforma variables numéricas y categóricas según corresponda.
        """

        # Obtener columnas válidas
        cuant_cols = self.__analizador.x.clasificacion_variables.get('Cuantitativas', [])
        cual_cols = self.__analizador.x.clasificacion_variables.get('Cualitativas', [])
        
        # Validar que haya al menos un tipo de variables
        if not cuant_cols and not cual_cols:
            raise ValueError(
                "No se encontraron variables numéricas ni categóricas válidas. "
                "Revise la clasificación de variables."
            )
        
        # Configurar transformers dinámicamente
        transformers = []
        if cuant_cols:
            transformers.append(('num', self.__numerical_transformer, cuant_cols))
        if cual_cols:
            transformers.append(('cat', self.__categorical_transformer, cual_cols))
        return ColumnTransformer(
            transformers = transformers,
            remainder = remainder, # opciones: drop o personalizado
            verbose_feature_names_out = False # False hace que no se le agregue el nombre del transformador, ejemplo: onehot_variable
        )
    
    #Métodos de clase
    def evaluar_modelos(self, ruta_excel: str = 'res/resultados_modelos.xlsx'):
        """Entrena y evalúa múltiples modelos de clasificación utilizando métricas estándar.

        Parámetros
        ----------
        ruta_excel: str
            Ruta del archivo Excel donde se guardarán los resultados
        
        Retorna
        -------
        df_resultados: pd.DataFrame
            DataFrame con métricas de evaluación para cada modelo, incluyendo Accuracy, AUC, Gini, Precision, Recall, F1 Score y las predicciones generadas.
        """


        results = []

        for name, classifier in self.__modelos.items():
            print(f"Training {name}...")
            model = Pipeline(steps = [
                ('preprocessor', self.__procesador),
                ('classifier', classifier)
            ])
            model.fit(self.__X_train, self.__y_train)
            accuracy = model.score(self.__X_test, self.__y_test)

            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(self.__X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(self.__y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                gini = 2 * roc_auc - 1
            else:
                roc_auc = None
                gini = None

            y_pred_class = model.predict(self.__X_test)
            precision = precision_score(self.__y_test, y_pred_class)
            recall = recall_score(self.__y_test, y_pred_class)
            f1 = f1_score(self.__y_test, y_pred_class)

            results.append({
                "Model": name,
                "Accuracy": accuracy,
                "ROC AUC": roc_auc,
                "Gini": gini,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Predictions": y_pred_class
            })
            
        df_resultados = pd.DataFrame(results)
        
        df_resultados.drop(columns = ["Predictions"]).to_excel(ruta_excel, index = False)


        return df_resultados


    def importancia_variables_modelos(self, nombre_modelo: str, modelo):
        '''
        Calcula la importancia de variables para distintos modelos.

        Parámetros:
        ------------
        nombre_modelo: str
            Nombre del modelo al que se le quiere estimar la importancia de las variables
        modelo: sklearn.pipeline.Pipeline
            modelo al que se le quiere estimar la importancia de las variables
        
        Retorna:
        --------
        dict
            Diccionario con nombre del modelo como clave y DataFrame de importancias como valor.
        '''

        importancias = {}

        print(f"Evaluando importancia para {nombre_modelo}...")
        
        model = modelo
        model.fit(self.__X_train, self.__y_train)
        feature_names = model.named_steps['preprocessor'].get_feature_names_out() #??? podría ser causante de error
        try:
            if hasattr(modelo.named_steps['classifier'], 'feature_importances_'):
                importancias[nombre_modelo] = pd.DataFrame({
                    'Variable': feature_names,
                    'Importancia': model.named_steps['classifier'].feature_importances_
                }).sort_values(by = 'Importancia', ascending = False)

            elif hasattr(modelo.named_steps['classifier'], 'coef_'):
                coef = model.named_steps['classifier'].coef_
                if coef.ndim == 2:
                    coef = coef[0]
                importancias[nombre_modelo] = pd.DataFrame({
                    'Variable': feature_names,
                    'Importancia': coef
                }).sort_values(by = 'Importancia', key = abs, ascending = False)

            else:
                print(f"Usando Permutation Importance para {nombre_modelo}...")
                r = permutation_importance(model, self.__X_test, self.__y_test, n_repeats = 10, random_state = 42, n_jobs = -1)
                importancias[nombre_modelo] = pd.DataFrame({
                    'Variable': feature_names,
                    'Importancia': r.importances_mean
                }).sort_values(by = 'Importancia', ascending = False)

        except Exception as e:
            print(f"No se pudo calcular la importancia para {nombre_modelo}: {e}")

        return importancias




        
    def evaluar_clusters(self, y_true: np.ndarray,
                        clusters_dict: dict,
                        ruta_excel: str):
        """Evalúa métricas de clasificación para diferentes resultados de clustering y guarda los resultados en un archivo Excel.

        Parámetros
        ----------
        y_true : np.ndarray
            Vector que contiene las clases verdaderas.
        clusters_dict : dict
            Diccionario donde las claves son nombres de métodos de clustering y los valores son los vectores de predicciones mapeadas.
            Ejemplo: {'clusters_complete_2': clusters_complete_mapped}
        ruta_excel : str
            Ruta absoluta donde se guardará el archivo de resultados en formato Excel.
        
        Retorna
        -------
        None
        """

        resultados = {
            "Clusters": [],
            "Matriz de Confusión": [],
            "Precisión": [],
            "Exactitud": [],
            "Recall": [],
            "F1": [],
            "AUC": [],
            "Gini": []
        }

        for nombre, predicciones in clusters_dict.items():
            cm = confusion_matrix(y_true, predicciones)
            precision = precision_score(y_true, predicciones)
            accuracy = accuracy_score(y_true, predicciones)
            recall = recall_score(y_true, predicciones)
            f1 = f1_score(y_true, predicciones)
            auc = roc_auc_score(y_true, predicciones)
            gini = 2 * auc - 1

            resultados["Clusters"].append(nombre)
            resultados["Matriz de Confusión"].append(cm)
            resultados["Precisión"].append(precision)
            resultados["Exactitud"].append(accuracy)
            resultados["Recall"].append(recall)
            resultados["F1"].append(f1)
            resultados["AUC"].append(auc)
            resultados["Gini"].append(gini)

        df_resultados_clusters = pd.DataFrame(resultados)

        # Guardar las métricas generales (sin matrices)
        df_resultados_clusters.drop(columns = "Matriz de Confusión").to_excel(ruta_excel, index = False)

        # Guardar matrices de confusión en hojas separadas
        with pd.ExcelWriter(ruta_excel, engine = "openpyxl", mode = 'a') as writer:
            for i, nombre in enumerate(resultados["Clusters"]):
                pd.DataFrame(resultados["Matriz de Confusión"][i]).to_excel(writer, sheet_name = f"Matriz_{nombre}", index = False)


    # --------------------------
    # Propiedades (Getters/Setters)
    # --------------------------
    
    @property
    def X(self):
        """Retorna el conjunto de variables predictoras (features).
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        pd.DataFrame:
            Conjunto de variables predictoras.
        """
        return self.__X
    
    @X.setter
    def X(self, value):
        """Asigna un nuevo conjunto de variables predictoras (features).
        
        Parámetros
        ----------
        value : pd.DataFrame
            DataFrame que contiene las variables predictoras.
        
        Retorna
        -------
        None
        """

        if not isinstance(value, pd.DataFrame):
            raise TypeError("Debe ser un DataFrame de pandas")
        self.__X = value


    @property
    def X_train(self):
        """Retorna el conjunto de variables predictoras para entrenamiento.
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        pd.DataFrame:
            Conjunto de variables predictoras del conjunto de entrenamiento.
        """
        return self.__X_train
    
    @X_train.setter
    def X_train(self, value):
        """Asigna el conjunto de variables predictoras para entrenamiento.
        
        Parámetros
        ----------
        value : pd.DataFrame
            DataFrame que contiene las variables predictoras del conjunto de entrenamiento.
        
        Retorna
        -------
        None
        """

        if not isinstance(value, pd.DataFrame):
            raise TypeError("Debe ser un DataFrame de pandas")
        self.__X_train = value

    @property
    def X_test(self):
        """Retorna el conjunto de variables predictoras para prueba.
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        pd.DataFrame:
            Conjunto de variables predictoras del conjunto de prueba.
        """
        return self.__X_test
    
    @X_test.setter
    def X_test(self, value):
        """Asigna el conjunto de variables predictoras para prueba.
        
        Parámetros
        ----------
        value : pd.DataFrame
            DataFrame que contiene las variables predictoras del conjunto de prueba.
        
        Retorna
        -------
        None
        """

        if not isinstance(value, pd.DataFrame):
            raise TypeError("Debe ser un DataFrame de pandas")
        self.__X_test = value

    @property
    def y_train(self):
        """Retorna la variable objetivo del conjunto de entrenamiento.
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        pd.Series:
            Serie con los valores de la variable objetivo para entrenamiento.
        """
        return self.__y_train
    
    @y_train.setter
    def y_train(self, value):
        """Asigna la variable objetivo del conjunto de entrenamiento.
        
        Parámetros
        ----------
        value : pd.Series
            Serie con los valores de la variable objetivo para entrenamiento.
        
        Retorna
        -------
        None
        """
        if not isinstance(value, pd.Series):
            raise TypeError("Debe ser una Series de pandas")
        self.__y_train = value

    @property
    def y_test(self):
        """Retorna la variable objetivo del conjunto de prueba.
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        pd.Series:
            Serie con los valores de la variable objetivo para prueba.
        """
        return self.__y_test
    
    @y_test.setter
    def y_test(self, value):
        """Asigna la variable objetivo del conjunto de prueba.
        
        Parámetros
        ----------
        value : pd.Series
            Serie con los valores de la variable objetivo para prueba.
        
        Retorna
        -------
        None
        """

        if not isinstance(value, pd.Series):
            raise TypeError("Debe ser una Series de pandas")
        self.__y_test = value

    @property
    def modelos(self):
        """Retorna el diccionario de modelos configurados para evaluación.
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        dict:
            Diccionario donde las claves son identificadores de modelos y los valores son instancias de clasificadores.
        """
        return self.__modelos
    
    @modelos.setter
    def modelos(self, value):
        """Asigna el diccionario de modelos a evaluar.
        
        Parámetros
        ----------
        value : dict
            Diccionario donde las claves son identificadores y los valores son instancias de clasificadores.
        
        Retorna
        -------
        None
        """

        if not isinstance(value, dict):
            raise TypeError("Debe ser un diccionario")
        self.__modelos = value

    @property
    def X_escalado(self):
        """Retorna el conjunto de variables cuantitativas escaladas.
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        np.ndarray:
            Arreglo con las variables cuantitativas ya transformadas mediante escalamiento.
        """
        return self.__X_escalado
    
    @property
    def procesador(self):
        """Retorna el preprocesador configurado para transformar variables numéricas y categóricas.
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        ColumnTransformer:
            Objeto configurado que aplica las transformaciones definidas para el conjunto de datos.
        """
        return self.__procesador
    
    @property
    def df(self):
        """Retorna el DataFrame original almacenado en la clase actual.
    
        Parámetros
        ----------
        None
    
        Retorna
        -------
        pd.DataFrame:
            Conjunto de datos original utilizado en el análisis.
        """
        return self.__df


    @df.setter
    def df(self, nuevo_df: pd.DataFrame, remainder = 'passthrough'):
        """Actualiza el DataFrame y recalcula todos los componentes dependientes.
        
        Parámetros:
        nuevo_df (pd.DataFrame): Nuevo DataFrame a analizar (última columna como target).
        """
        self.__X = nuevo_df[self.__var_objetivo].copy()
        self.__Y = nuevo_df.copy().drop(self.__var_objetivo)
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            self.__X,
            self.__Y,
            test_size = self.__test_size, 
            random_state = 42
        )
        
        self.__X_escalado = self.__numerical_transformer.fit_transform(self.__X)
        self.__analizador = AnalizadorVariableObjetivo(self.__X_train, self.__var_objetivo)
        self.__procesador = self.__configurar_preprocesador(remainder)
        
    def __str__(self):
        """Genera una representación legible del objeto ModeladorDatos.
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        str:
            Cadena con un resumen del conjunto de datos, incluyendo número de observaciones, nombres de las variables, tamaño de los conjuntos de entrenamiento y prueba, y modelos configurados.
        """
        info = [
            "ModeladorDatos resumen:",
            f"- Número de observaciones (total): {len(self.__X)}",
            f"- Variables predictoras: {list(self.__X.columns)}",
            f"- Variable objetivo: {self.__Y.name}",
            f"- Tamaño de entrenamiento: {len(self.__X_train)}",
            f"- Tamaño de prueba: {len(self.__X_test)}",
            f"- Modelos disponibles: {list(self.__modelos.keys())}"
        ]
        return "\n".join(info)
