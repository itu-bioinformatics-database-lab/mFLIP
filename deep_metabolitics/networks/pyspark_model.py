# from pyspark.ml.regression import LinearRegression
from tqdm import tqdm
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame

class PySparkMultiOutputRegressor:
    def __init__(self, base_model, feature_columns, target_columns, label_model):
        """
        Multi-output regression için PySpark modeli oluşturur.
        :param base_model: Kullanılacak temel regresyon modeli (Varsayılan: LinearRegression)
        """
        # self.base_model = base_model if base_model else LinearRegression()
        self.base_model = base_model
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.label_model = label_model
        self.models = {}
        n_targets = len(self.target_columns) 
        self.std_values = [float(self.label_model.stages[i].std[0]) for i in range(n_targets, 2 * n_targets)]


    def fit(self, X, y):
        """
        PySpark DataFrame kullanarak modeli eğitir.
        :param df: PySpark DataFrame
        :param feature_columns: Kullanılacak özellik sütunları
        :param target_columns: Tahmin edilecek hedef sütunları
        """

        # DataFrame'lere satır numarası ekleyerek eşleşmelerini garanti altına al
        features_df = X.withColumn("row_index", monotonically_increasing_id())
        targets_df = y.withColumn("row_index", monotonically_increasing_id())

        # İki DataFrame'i index üzerinden birleştir
        df = features_df.join(targets_df, on="row_index", how="inner")

        for target in self.target_columns:
            model = self.base_model.fit(df.select(self.feature_columns, f"scaled_{target}"))
            self.models[target] = model
        return self

    def transform(self, X: DataFrame):
        """
        PySpark DataFrame üzerinde tahmin yapar.
        :param df: PySpark DataFrame
        :return: Tahmin sonuçlarıyla yeni bir DataFrame
        """
        if not self.models:
            raise ValueError("Modelin önce fit() ile eğitilmesi gerekiyor!")

        df = X

        for target, model in tqdm(self.models.items()):
            df = model.transform(df).withColumnRenamed("prediction", f"prediction_{target}")

        return df

    def predict(self, X: DataFrame):
        """
        PySpark DataFrame üzerinde tahmin yapıp sadece sonuçları döndürür.
        :param df: PySpark DataFrame
        :return: Tahmin sonuçlarını içeren DataFrame
        """
        predictions_df = self.transform(X)
        return self.inverse_labels(predictions_df.select([f"prediction_{target}" for target in self.target_columns]).toPandas().applymap(lambda x: float(x[0])).to_numpy())

    def label_tonumpy(self, y: DataFrame):
        label_columns = [f"scaled_{target}" for target in self.target_columns]
        return self.inverse_labels(y.select(label_columns).toPandas().applymap(lambda x: float(x[0])).to_numpy())

    def inverse_labels(self, numpy_values):
        # n_targets = len(self.target_columns) 
        # std_values = [float(self.label_model.stages[i].std[0]) for i in range(n_targets, 2 * n_targets)]
        return numpy_values * self.std_values