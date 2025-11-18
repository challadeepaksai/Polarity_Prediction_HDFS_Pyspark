import json, traceback
from collections import Counter

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array, udf
from pyspark.sql.types import ArrayType, StringType, IntegerType, DoubleType

from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover,
    CountVectorizerModel, IDFModel,
    VectorAssembler, StringIndexerModel
)
from pyspark.ml.classification import (
    NaiveBayesModel, LogisticRegressionModel, OneVsRestModel
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

TRAIN_PATH = "hdfs://master:9000/user/deepakchalla/dataset/amazon_review_polarity_csv/train.csv"
TEST_PATH  = "hdfs://master:9000/user/deepakchalla/dataset/amazon_review_polarity_csv/test.csv"

TFIDF_DIR     = "hdfs://master:9000/user/deepakchalla/models/tfidf_models"
PIPELINES_DIR = "hdfs://master:9000/user/deepakchalla/models/pipelines"
MODELS_DIR    = "hdfs://master:9000/user/deepakchalla/models/models"

TEXT_COLS = ["title", "review"]

def log(msg):
    print(f"\n=====> {msg}")

def porter_stem_udf():
    def stem(tokens):
        if tokens is None:
            return []
        try:
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            return [ps.stem(t) for t in tokens if t is not None]
        except Exception:
            return tokens if tokens is not None else []
    return udf(stem, ArrayType(StringType()))

def safe_predict_and_rename(model, df, out_col):
    for c in ("prediction", "rawPrediction", "probability"):
        if c in df.columns:
            df = df.drop(c)
    transformed = model.transform(df)
    if "prediction" not in transformed.columns:
        raise RuntimeError("Model.transform did not produce 'prediction' column")
    transformed = transformed.withColumnRenamed("prediction", out_col)
    return transformed.withColumn(out_col, col(out_col).cast(IntegerType()))

majority_udf = udf(lambda arr: Counter(arr).most_common(1)[0][0] if arr else -1, IntegerType())

def main():
    try:
        spark = SparkSession.builder.appName("Eval_Final_NoIndexerTransform").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        log("Loading train and test CSVs")
        train = (spark.read.option("header","false").option("inferSchema","true").csv(TRAIN_PATH)
                 .withColumnRenamed("_c0","label")
                 .withColumnRenamed("_c1","title")
                 .withColumnRenamed("_c2","review")
                 .fillna({"title":"", "review":""}))
        test = (spark.read.option("header","false").option("inferSchema","true").csv(TEST_PATH)
                .withColumnRenamed("_c0","label")
                .withColumnRenamed("_c1","title")
                .withColumnRenamed("_c2","review")
                .fillna({"title":"", "review":""}))

        log(f"Train count: {train.count()}  Test count: {test.count()}")

        log("Applying RegexTokenizer + StopWordsRemover for each text column")
        for t in TEXT_COLS:
            tok = RegexTokenizer(inputCol=t, outputCol=f"{t}_tok", pattern="\\W+")
            stop = StopWordsRemover(inputCol=f"{t}_tok", outputCol=f"{t}_nostop")
            train = stop.transform(tok.transform(train))
            test  = stop.transform(tok.transform(test))

        log("Applying Porter stemmer (UDF)")
        stem = porter_stem_udf()
        for t in TEXT_COLS:
            train = train.withColumn(f"{t}_stem", stem(col(f"{t}_nostop")))
            test  = test.withColumn(f"{t}_stem", stem(col(f"{t}_nostop")))

        log("Loading CountVectorizerModel + IDFModel and transforming")
        for t in TEXT_COLS:
            cv_path = f"{TFIDF_DIR}/{t}_cv"
            idf_path = f"{TFIDF_DIR}/{t}_idf"
            log(f"Loading CV model: {cv_path}")
            cv_model = CountVectorizerModel.load(cv_path)
            log(f"Loading IDF model: {idf_path}")
            idf_model = IDFModel.load(idf_path)

            if f"{t}_stem" not in train.columns or f"{t}_stem" not in test.columns:
                raise RuntimeError(f"Missing expected stem column: {t}_stem")

            train = cv_model.transform(train)
            train = idf_model.transform(train)
            test  = cv_model.transform(test)
            test  = idf_model.transform(test)
            log(f"TF-IDF applied for {t}")

        log("Loading VectorAssembler and creating 'features'")
        assembler = VectorAssembler.load(f"{PIPELINES_DIR}/assembler")
        train = assembler.transform(train)
        test  = assembler.transform(test)

        log("Loading StringIndexerModel to read training label ordering")
        indexer = StringIndexerModel.load(f"{PIPELINES_DIR}/label_indexer")

        labels = None
        try:
            labels = indexer.labels
        except Exception:
            meta_path = f"{PIPELINES_DIR}/label_indexer/metadata/part-00000"
            try:
                meta_raw = spark.sparkContext.textFile(meta_path).collect()[0]
                meta = json.loads(meta_raw)
                labels = meta.get("labels", None)
            except Exception:
                labels = None

        if not labels:
            raise RuntimeError("Failed to extract label ordering from saved StringIndexerModel or metadata.")

        label_map = {str(l): int(i) for i, l in enumerate(labels)}
        log(f"Loaded label mapping (samples): {list(label_map.items())[:10]}")

        def map_label_fn(x):
            if x is None:
                return -1
            return label_map.get(str(x), -1)
        map_label_udf = udf(map_label_fn, IntegerType())

        log("Casting CSV label to string and mapping to indexed label (no transform())")
        train = train.withColumn("label_str", col("label").cast(StringType()))
        test  = test.withColumn("label_str", col("label").cast(StringType()))

        train = train.withColumn("label", map_label_udf(col("label_str"))).drop("label_str")
        test  = test.withColumn("label", map_label_udf(col("label_str"))).drop("label_str")

        train = train.select("features", "label")
        test  = test.select("features", "label")

        log("Label mapping complete. Schema: " + ", ".join(train.schema.fieldNames()))

        log("Loading trained models (naive_bayes, logistic_regression, svm_lsvc)")
        nb_model  = NaiveBayesModel.load(f"{MODELS_DIR}/naive_bayes")
        lr_model  = LogisticRegressionModel.load(f"{MODELS_DIR}/logistic_regression")
        svm_model = OneVsRestModel.load(f"{MODELS_DIR}/svm_lsvc")

        log("Running safe chained predictions: NB -> LR -> SVM")
        df1 = safe_predict_and_rename(nb_model, test, "nb_i")
        df2 = safe_predict_and_rename(lr_model, df1, "lr_i")
        df3 = safe_predict_and_rename(svm_model, df2, "svm_i")

        df_preds = df3.withColumn("ensemble_pred", majority_udf(array("nb_i", "lr_i", "svm_i")))

        log("Casting label & prediction columns to double for evaluator")
        df_preds = (df_preds
                    .withColumn("label", col("label").cast(DoubleType()))
                    .withColumn("nb_i", col("nb_i").cast(DoubleType()))
                    .withColumn("lr_i", col("lr_i").cast(DoubleType()))
                    .withColumn("svm_i", col("svm_i").cast(DoubleType()))
                    .withColumn("ensemble_pred", col("ensemble_pred").cast(DoubleType())))

        log("Computing metrics (accuracy, weightedPrecision, weightedRecall, f1)")
        acc_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
        rec_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
        f1_eval   = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

        def evaluate_for(colname):
            tmp = df_preds.withColumnRenamed(colname, "prediction")
            return {
                "accuracy":  acc_eval.evaluate(tmp),
                "precision": prec_eval.evaluate(tmp),
                "recall":    rec_eval.evaluate(tmp),
                "f1":        f1_eval.evaluate(tmp)
            }

        results = {
            "NaiveBayes": evaluate_for("nb_i"),
            "LogisticRegression": evaluate_for("lr_i"),
            "SVM": evaluate_for("svm_i"),
            "Ensemble": evaluate_for("ensemble_pred")
        }

        log("===== FINAL METRICS =====")
        for name, metrics in results.items():
            print(f"\nModel: {name}")
            print(f"  Accuracy : {metrics['accuracy']:.6f}")
            print(f"  Precision: {metrics['precision']:.6f}")
            print(f"  Recall   : {metrics['recall']:.6f}")
            print(f"  F1-score : {metrics['f1']:.6f}")

        spark.stop()

    except Exception:
        print("\nFATAL ERROR DURING EVALUATION:")
        traceback.print_exc()
        try:
            spark.stop()
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()
