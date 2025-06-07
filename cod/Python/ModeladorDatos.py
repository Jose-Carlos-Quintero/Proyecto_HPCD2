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
        '''
         Inicializa el modelo con un DataFrame.
        
        Parámetros
        ----------
        df (pd.DataFrame): DataFrame donde la última columna es la variable objetivo (Y).
        modelos (dict): Diccionario de modelos a evaluar (ej: {'log_reg': LogisticRegression()}).
        test_size (float): Proporción para el conjunto de prueba (default: 0.3).
        handle_unknown (str): Comportamiento para categorías desconocidas en OneHotEncoder (default: 'ignore').
        sparse_output (bool): Si True devuelve matrices sparse en OneHotEncoder (default: False).
        remainder (str): Qué hacer con columnas no especificadas (default: 'passthrough').
        '''
        self.__Y = df[var_objetivo].copy()
        self.__X = df.copy().drop(columns = [var_objetivo])
        self.__numerical_transformer = StandardScaler() 
        self.__categorical_transformer = OneHotEncoder(handle_unknown = handle_unknown, sparse_output = sparse_output)
        self.__test_size = test_size
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = (
            train_test_split(self.__X,self.__Y,test_size = test_size,random_state = 2)
        )
        self.__modelos = modelos
        self.__analizador = AnalizadorVariableObjetivo(pd.concat([self.__X_train, self.__y_train], axis=1), 
                                                       var_objetivo)
        self.__procesador = self.__configurar_preprocesador(remainder)
        self.__X_escalado = self.__numerical_transformer.fit_transform(self.__X[self.__analizador.x.clasificacion_variables['Cuantitativas']])
        #??? considerar si uso __X o __X_train, dependiendo de qué quiero en el supervisado

    #Métodos internos
    def __configurar_preprocesador(self, remainder: str) -> ColumnTransformer:
        """Configura el ColumnTransformer de manera robusta y elegante.
        
        Parámetros:
        -----------
            remainder (str): Qué hacer con columnas no especificadas
            
        Retornos:
        ---------
            ColumnTransformer: ColumnTransformer configurado
            
        Raises:
        -------
            ValueError: Si no hay variables para procesar
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
    def evaluar_modelos(self):
        '''Entrena y evalúa múltiples modelos de clasificación con métricas estándar.

        Parámetros:
        -----------
        procesador (sklearn.Pipeline o ColumnTransformer): Pipeline de preprocesamiento.
        X_train, X_test (pd.DataFrame o np.ndarray): Conjuntos de entrenamiento y prueba (features).
        y_train, y_test (pd.Series o np.ndarray): Conjuntos de entrenamiento y prueba (target).

        Returns:
        --------
        pd.DataFrame
            DataFrame con métricas de evaluación por modelo.
        '''

        results = []

        for name, classifier in self.__modelos.items():
            print(f"Training {name}...")
            model = Pipeline(steps=[
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

        return pd.DataFrame(results)


    def importancia_variables_modelos(self):
        '''
        Calcula la importancia de variables para distintos modelos.

        Parámetros:
        ------------
        procesador (sklearn.Pipeline o ColumnTransformer): Preprocesador aplicado antes del clasificador.
        X_train (pd.DataFrame): Variables independientes de entrenamiento.
        X_test (pd.DataFrame): Variables independientes de prueba.
        y_train (pd.Series): Variable dependiente de entrenamiento.
        y_test (pd.Series): Variable dependiente de prueba.
        feature_names (list): Lista de nombres de las variables después del preprocesamiento.

        Returns:
        --------
        dict
            Diccionario con nombre del modelo como clave y DataFrame de importancias como valor.
        '''

        importancias = {}

        for name, classifier in self.__modelos.items():
            print(f"Evaluando importancia para {name}...")

            model = Pipeline(steps=[
                ('preprocessor', self.__procesador),
                ('classifier', classifier)
            ])
            model.fit(self.__X_train, self.__y_train)
            feature_names = model.named_steps['preprocessor'].get_feature_names_out() #??? podría ser causante de error
            try:
                if hasattr(classifier, 'feature_importances_'):
                    importancias[name] = pd.DataFrame({
                        'Variable': feature_names,
                        'Importancia': model.named_steps['classifier'].feature_importances_
                    }).sort_values(by='Importancia', ascending=False)

                elif hasattr(classifier, 'coef_'):
                    coef = model.named_steps['classifier'].coef_
                    if coef.ndim == 2:
                        coef = coef[0]
                    importancias[name] = pd.DataFrame({
                        'Variable': feature_names,
                        'Importancia': coef
                    }).sort_values(by='Importancia', key=abs, ascending=False)

                else:
                    print(f"Usando Permutation Importance para {name}...")
                    r = permutation_importance(model, self.__X_test, self.__y_test, n_repeats=10, random_state=42, n_jobs=-1)
                    importancias[name] = pd.DataFrame({
                        'Variable': feature_names,
                        'Importancia': r.importances_mean
                    }).sort_values(by='Importancia', ascending=False)

            except Exception as e:
                print(f"No se pudo calcular la importancia para {name}: {e}")

        return importancias




        
    def evaluar_clusters(self, y_true: np.ndarray,
                        clusters_dict: dict,
                        ruta_excel: str):
        '''Evalúa métricas de clasificación para diferentes resultados de clustering y guarda los resultados en un archivo Excel.

        Parámetros:
        -----------
        y_true (np.ndarray): Vector de clases verdaderas.
        clusters_dict (dict): Diccionario con el nombre del método como clave y los vectores de predicciones mapeadas como valor.
                            Ejemplo: {'clusters_complete_2': clusters_complete_mapped}
        ruta_resultado_clusters (str): Ruta absoluta donde se guardará el archivo de resultados (`resultados_clusters.xlsx`).

        Returns:
        --------
        None
        '''

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
        df_resultados_clusters.drop(columns="Matriz de Confusión").to_excel(ruta_excel, index=False)

        # Guardar matrices de confusión en hojas separadas
        with pd.ExcelWriter(ruta_excel, engine="openpyxl", mode='a') as writer:
            for i, nombre in enumerate(resultados["Clusters"]):
                pd.DataFrame(resultados["Matriz de Confusión"][i]).to_excel(writer, sheet_name=f"Matriz_{nombre}", index=False)


    # --------------------------
    # Propiedades (Getters/Setters)
    # --------------------------
    
    @property
    def X(self):
        """DataFrame: Features."""
        return self.__X
    
    @X.setter
    def X_train(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Debe ser un DataFrame de pandas")
        self.__X = value

    @property
    def X_train(self):
        """DataFrame: Features de entrenamiento."""
        return self.__X_train
    
    @X_train.setter
    def X_train(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Debe ser un DataFrame de pandas")
        self.__X_train = value

    @property
    def X_test(self):
        """DataFrame: Features de prueba."""
        return self.__X_test
    
    @X_test.setter
    def X_test(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Debe ser un DataFrame de pandas")
        self.__X_test = value

    @property
    def y_train(self):
        """pd.Series: Target de entrenamiento."""
        return self.__y_train
    
    @y_train.setter
    def y_train(self, value):
        if not isinstance(value, pd.Series):
            raise TypeError("Debe ser una Series de pandas")
        self.__y_train = value

    @property
    def y_test(self):
        """pd.Series: Target de prueba."""
        return self.__y_test
    
    @y_test.setter
    def y_test(self, value):
        if not isinstance(value, pd.Series):
            raise TypeError("Debe ser una Series de pandas")
        self.__y_test = value

    @property
    def modelos(self):
        """dict: Diccionario de modelos configurados."""
        return self.__modelos
    
    @modelos.setter
    def modelos(self, value):
        if not isinstance(value, dict):
            raise TypeError("Debe ser un diccionario")
        self.__modelos = value

    @property
    def X_escalado(self):
        """np.array: Features escalados."""
        return self.__X_escalado
    
    @property
    def procesador(self):
        """ColumnTransformer: Procesador configurado."""
        return self.__procesador
    
    @property
    def df(self):
        """Devuelve el DataFrame almacenado en la clase padre (AnalizadorDataFrame)."""
        return super().__df  # Esto funciona si el padre tiene un getter para df  #??? no hay herencia

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
            test_size=self.__test_size, 
            random_state=2
        )
        
        self.__X_escalado = self.__numerical_transformer.fit_transform(self.__X)
        self.__analizador = AnalizadorVariableObjetivo(self.__X_train, self.__var_objetivo)
        self.__procesador = self.__configurar_preprocesador(remainder)