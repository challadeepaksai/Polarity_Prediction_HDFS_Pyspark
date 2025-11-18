import os
from collections import Counter

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, array
from pyspark.sql.types import ArrayType, StringType, IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, CountVectorizer, IDF,
    VectorAssembler, StringIndexer
)
from pyspark.ml.classification import NaiveBayes, LogisticRegression, LinearSVC, OneVsRest
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



INPUT_PATH = "hdfs://master:9000/user/deepakchalla/dataset/amazon_review_polarity_csv/train.csv"
INPUT_FORMAT = "csv"
CSV_HAS_HEADER = False
CSV_INFER_SCHEMA = True

TEXT_COLS = ["title", "review"]
LABEL_COL = "label"

OUTPUT_BASE = "hdfs://master:9000/user/deepakchalla/models"

TFIDF_DIR = os.path.join(OUTPUT_BASE, "tfidf_models")
MODELS_DIR = os.path.join(OUTPUT_BASE, "models")
PIPELINES_DIR = os.path.join(OUTPUT_BASE, "pipelines")
ENSEMBLE_DIR = os.path.join(OUTPUT_BASE, "ensemble")
METADATA_DIR = os.path.join(OUTPUT_BASE, "metadata")

VOCAB_SIZE = 30000
MIN_DF = 2.0
NUM_FOLDS = 3
CV_PARALLELISM = 4
TRAIN_FRACTION = 0.8
SEED = 42



def porter_stem(tokens):
    if tokens is None:
        return []
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    return [ps.stem(t) for t in tokens if t is not None]

stem_udf = udf(porter_stem, ArrayType(StringType()))

def majority_vote_list(preds):
    if preds is None or len(preds) == 0:
        return -1
    preds_int = [int(x) for x in preds]
    return Counter(preds_int).most_common(1)[0][0]

majority_vote_udf = udf(majority_vote_list, IntegerType())


def read_input(spark):
    if INPUT_FORMAT == "csv":
        df = (
            spark.read
            .option("header", str(CSV_HAS_HEADER).lower())
            .option("inferSchema", str(CSV_INFER_SCHEMA).lower())
            .csv(INPUT_PATH)
        )
    elif INPUT_FORMAT == "json":
        df = spark.read.json(INPUT_PATH)
    else:
        df = spark.read.parquet(INPUT_PATH)
    return df


def save_model(obj, path):
    try:
        obj.write().overwrite().save(path)
    except:
        obj.save(path)



def main():
    print("\n=====> STARTING SPARK SESSION")
    spark = SparkSession.builder.appName("SparkNLP_Pipeline").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print("\n=====> READING DATASET:", INPUT_PATH)
    df = read_input(spark)

    print("\n=====> RAW SCHEMA:")
    df.printSchema()

    print("\n=====> RENAMING COLUMNS (label, title, review)")
    df = (
        df.withColumnRenamed("_c0", "label")
          .withColumnRenamed("_c1", "title")
          .withColumnRenamed("_c2", "review")
    )

    print("=====> FIXING NULL VALUES IN title/review + CAST TO STRING")
    from pyspark.sql.functions import col
    df = (
        df.withColumn("title", col("title").cast("string"))
          .withColumn("review", col("review").cast("string"))
          .fillna({"title": "", "review": ""})
    )

    print("\n=====> SCHEMA AFTER CLEANING:")
    df.printSchema()

    print("\n=====> TOKENIZING + REMOVING STOPWORDS")
    stages = []
    for tcol in TEXT_COLS:
        tok = RegexTokenizer(inputCol=tcol, outputCol=f"{tcol}_tok", pattern="\\W+")
        stop = StopWordsRemover(inputCol=f"{tcol}_tok", outputCol=f"{tcol}_nostop")
        stages += [tok, stop]

    print("=====> FITTING TOKENIZATION PIPELINE...")
    initial_pipeline = Pipeline(stages=stages)
    df_tokens = initial_pipeline.fit(df).transform(df)

    print("\n=====> APPLYING PORTER STEMMER")
    for tcol in TEXT_COLS:
        print(f"      -> Stemming column: {tcol}")
        df_tokens = df_tokens.withColumn(f"{tcol}_stem", stem_udf(col(f"{tcol}_nostop")))

    print("\n=====> GENERATING TF-IDF FEATURES")
    tfidf_vectors = []

    for tcol in TEXT_COLS:
        print(f"\n=====> FITTING CountVectorizer FOR COLUMN: {tcol}")
        cv = CountVectorizer(inputCol=f"{tcol}_stem", outputCol=f"{tcol}_tf",
                             vocabSize=VOCAB_SIZE, minDF=MIN_DF)
        cv_model = cv.fit(df_tokens)
        df_tokens = cv_model.transform(df_tokens)
        save_model(cv_model, f"{TFIDF_DIR}/{tcol}_cv")

        print(f"=====> FITTING IDF FOR COLUMN: {tcol}")
        idf = IDF(inputCol=f"{tcol}_tf", outputCol=f"{tcol}_tfidf")
        idf_model = idf.fit(df_tokens)
        df_tokens = idf_model.transform(df_tokens)
        save_model(idf_model, f"{TFIDF_DIR}/{tcol}_idf")

        tfidf_vectors.append(f"{tcol}_tfidf")

    print("\n=====> ASSEMBLING FEATURES")
    assembler = VectorAssembler(inputCols=tfidf_vectors, outputCol="features")
    df_features = assembler.transform(df_tokens)
    save_model(assembler, f"{PIPELINES_DIR}/assembler")

    print("\n=====> FITTING LABEL INDEXER")
    label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
    label_indexer_model = label_indexer.fit(df_features)
    df_final = label_indexer_model.transform(df_features)
    save_model(label_indexer_model, f"{PIPELINES_DIR}/label_indexer")

    df_final = df_final.select("features", "label_index")
    df_final = df_final.withColumnRenamed("label_index", "label")

    print("\n=====> SPLITTING INTO TRAIN/TEST")
    train_df, test_df = df_final.randomSplit([TRAIN_FRACTION, 1 - TRAIN_FRACTION], seed=SEED)

    print("      -> Train count:", train_df.count())
    print("      -> Test count :", test_df.count())

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    print("\n=====> TRAINING NAIVE BAYES (3-FOLD CV)")
    nb = NaiveBayes(featuresCol="features", labelCol="label")
    nb_cv = CrossValidator(
        estimator=nb,
        estimatorParamMaps=ParamGridBuilder().build(),
        evaluator=evaluator,
        numFolds=NUM_FOLDS,
        parallelism=CV_PARALLELISM
    )
    nb_model = nb_cv.fit(train_df).bestModel
    print("=====> SAVING NB MODEL")
    save_model(nb_model, f"{MODELS_DIR}/naive_bayes")

    print("\n=====> TRAINING LOGISTIC REGRESSION (3-FOLD CV)")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )
    lr_cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=lr_grid,
        evaluator=evaluator,
        numFolds=NUM_FOLDS,
        parallelism=CV_PARALLELISM
    )
    lr_model = lr_cv.fit(train_df).bestModel
    print("=====> SAVING LR MODEL")
    save_model(lr_model, f"{MODELS_DIR}/logistic_regression")

    print("\n=====> TRAINING SVM (LinearSVC via OneVsRest) (3-FOLD CV)")
    lsvc = LinearSVC(featuresCol="features", labelCol="label", maxIter=100)
    ovr = OneVsRest(classifier=lsvc)

    svm_grid = ParamGridBuilder().addGrid(lsvc.regParam, [0.01, 0.1]).build()
    svm_cv = CrossValidator(
        estimator=ovr,
        estimatorParamMaps=svm_grid,
        evaluator=evaluator,
        numFolds=NUM_FOLDS,
        parallelism=CV_PARALLELISM
    )

    svm_model = svm_cv.fit(train_df).bestModel
    print("=====> SAVING SVM MODEL")
    save_model(svm_model, f"{MODELS_DIR}/svm_lsvc")
    spark.stop()


if __name__ == "__main__":
    main()
