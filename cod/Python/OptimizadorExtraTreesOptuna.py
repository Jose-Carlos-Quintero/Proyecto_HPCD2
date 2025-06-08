import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ModeladorDatos import ModeladorDatos


class OptimizadorExtraTreesOptuna:
    """
    Clase que encapsula la optimización de hiperparámetros para ExtraTreesClassifier usando Optuna.
    """

    def __init__(self, df: pd.DataFrame, variable_objetivo: str, n_trials: int = 30):
        self.__modelador = ModeladorDatos(df, variable_objetivo, modelos = {})
        self.__variable_objetivo = variable_objetivo
        self.__n_trials = n_trials
        self.__study = None
        self.__modelo_final = None
        
    @property
    def modelador(self):
        """ModeladorDatos: objeto encargado del preprocesamiento y división de datos."""
        return self.__modelador

    @modelador.setter
    def modelador(self, value):
        if not isinstance(value, ModeladorDatos):
            raise TypeError("El valor debe ser una instancia de ModeladorDatos.")
        self.__modelador = value

    @property
    def variable_objetivo(self):
        """str: Nombre de la variable objetivo."""
        return self.__variable_objetivo

    @variable_objetivo.setter
    def variable_objetivo(self, value):
        if not isinstance(value, str):
            raise TypeError("El nombre de la variable objetivo debe ser una cadena de texto.")
        self.__variable_objetivo = value

    @property
    def n_trials(self):
        """int: Número de iteraciones de búsqueda de Optuna."""
        return self.__n_trials

    @n_trials.setter
    def n_trials(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("n_trials debe ser un entero positivo.")
        self.__n_trials = value

    @property
    def study(self):
        """optuna.Study: Estudio de Optuna con los resultados de la optimización."""
        return self.__study

    @study.setter
    def study(self, value):
        if not isinstance(value, optuna.Study) and value is not None:
            raise TypeError("El valor debe ser un objeto optuna.Study o None.")
        self.__study = value

    @property
    def modelo_final(self):
        """Pipeline: Modelo entrenado con los mejores hiperparámetros."""
        return self.__modelo_final

    @modelo_final.setter
    def modelo_final(self, value):
        if not isinstance(value, Pipeline) and value is not None:
            raise TypeError("modelo_final debe ser un sklearn.Pipeline o None.")
        self.__modelo_final = value
        
    def __str__(self):
        """Representación legible del objeto OptimizadorExtraTreesOptuna."""
        info = [
            "OptimizadorExtraTreesOptuna resumen:",
            f"- Variable objetivo: {self.__variable_objetivo}",
            f"- Número de ensayos (n_trials): {self.__n_trials}",
            f"- ¿Estudio Optuna creado?: {'Sí' if self.__study is not None else 'No'}",
            f"- ¿Modelo final entrenado?: {'Sí' if self.__modelo_final is not None else 'No'}",
            f"- Número de observaciones en entrenamiento: {len(self.__modelador.X_train)}",
            f"- Modelador asociado: {type(self.__modelador).__name__}"
        ]
        return "\n".join(info)


    def optimizar(self):
        """Ejecuta la optimización de hiperparámetros con Optuna.
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        str:
            Cadena de texto donde se muestran los mejores hiperparametros encontrados y el score obtenido con los mismos
        
        """

        def objetivo(trial):
            try:
                n_estimators = trial.suggest_int("n_estimators", 100, 500)
                max_depth = trial.suggest_int("max_depth", 5, 40)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
                max_features = trial.suggest_float("max_features", 0.3, 0.99)
                max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 20, 300)

                pipeline = Pipeline(steps=[
                    ('preprocessor', self.__modelador.procesador),
                    ('classifier', ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        max_leaf_nodes=max_leaf_nodes,
                        bootstrap=False,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])

                score = cross_val_score(
                    pipeline,
                    self.__modelador.X_train,
                    self.__modelador.y_train,
                    cv=3,
                    scoring='f1_macro',
                    n_jobs=-1
                ).mean()

                return score if score is not None else float('-inf')
            except Exception as e:
                print(f"Error en el ensayo: {e}")
                return float('-inf')

        self.__study = optuna.create_study(direction="maximize")
        self.__study.optimize(objetivo, n_trials=self.__n_trials)

        print("Mejores hiperparámetros encontrados:")
        print(self.__study.best_params)
        print("Mejor F1 (macro):", self.__study.best_value)

        return f"Parámetros: {self.__study.best_params}\nScore: {self.__study.best_value}"

        print("Mejores hiperparámetros encontrados:")
        print(self.__study.best_params)
        print("Mejor F1 (macro):", self.__study.best_value)
        
        return f"Parámetros: {self.__study.best_params}\nScore: {self.__study.best_value}"

    def entrenar_modelo_final(self):
        """Entrena el modelo ExtraTrees con los mejores hiperparámetros encontrados.
        
        """

        best_params = self.__study.best_params

        self.__modelo_final = Pipeline(steps=[
            ('preprocessor', self.__modelador.procesador),
            ('classifier', ExtraTreesClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                max_features=best_params['max_features'],
                max_leaf_nodes=best_params['max_leaf_nodes'],
                random_state = 42,
                n_jobs = -1
            ))
        ])

        self.__modelo_final.fit(self.__modelador.X_train, self.__modelador.y_train)
        
        return self.__modelo_final

    def evaluar_modelo_final(self, ruta_salida: str = "Resultado_modelo_optuna.xlsx") -> pd.DataFrame:
        """
        Evalúa el modelo entrenado para distintos umbrales de clasificación y guarda resultados en Excel.

        Parámetros
        ----------
        ruta_salida : str
            Ruta al archivo Excel donde se guardarán los resultados.

        Retorna
        -------
        df_resultaados: pd.DataFrame
            DataFrame con métricas para distintos umbrales.
        """

        y_pred_proba = self.__modelo_final.predict_proba(self.__modelador.X_test)[:, 1]
        resultados = []

        for i in range(11):
            umbral = i * 0.1
            y_pred_class = (y_pred_proba >= umbral).astype(int)

            accuracy = accuracy_score(self.__modelador.y_test, y_pred_class)
            precision = precision_score(self.__modelador.y_test, y_pred_class, zero_division = 0)
            recall = recall_score(self.__modelador.y_test, y_pred_class, zero_division = 0)
            f1 = f1_score(self.__modelador.y_test, y_pred_class, zero_division = 0)

            resultados.append((accuracy, precision, recall, f1))

        df_resultados = pd.DataFrame(resultados, columns = ['Accuracy', 'Precisión', 'Recall', 'F1'])
        df_resultados.to_excel(ruta_salida, index = False)

        return df_resultados
