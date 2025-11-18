import webbrowser
from threading import Timer
from flask import Flask, render_template, request

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array
from pyspark.sql.types import ArrayType, StringType, IntegerType

from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover,
    CountVectorizerModel, IDFModel, VectorAssembler
)
from pyspark.ml.classification import (
    NaiveBayesModel, LogisticRegressionModel, OneVsRestModel
)

from collections import Counter


spark = SparkSession.builder \
    .appName("SentimentUI_Final") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


BASE = "hdfs://master:9000/user/deepakchalla/models"

CV_TITLE  = CountVectorizerModel.load(f"{BASE}/tfidf_models/title_cv")
IDF_TITLE = IDFModel.load(f"{BASE}/tfidf_models/title_idf")

CV_REVIEW  = CountVectorizerModel.load(f"{BASE}/tfidf_models/review_cv")
IDF_REVIEW = IDFModel.load(f"{BASE}/tfidf_models/review_idf")

ASSEMBLER = VectorAssembler.load(f"{BASE}/pipelines/assembler")

NB  = NaiveBayesModel.load(f"{BASE}/models/naive_bayes")
LR  = LogisticRegressionModel.load(f"{BASE}/models/logistic_regression")
SVM = OneVsRestModel.load(f"{BASE}/models/svm_lsvc")



def porter_stem(tokens):
    if tokens is None:
        return []
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    return [ps.stem(t) for t in tokens]

stem_udf = udf(porter_stem, ArrayType(StringType()))

majority = udf(lambda arr: Counter(arr).most_common(1)[0][0], IntegerType())


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    title = request.form["title"]
    review = request.form["review"]

    df = spark.createDataFrame([(title, review)], ["title", "review"])


    tok_title = RegexTokenizer(inputCol="title", outputCol="title_tok", pattern="\\W+")
    tok_rev   = RegexTokenizer(inputCol="review", outputCol="review_tok", pattern="\\W+")
    df = tok_title.transform(df)
    df = tok_rev.transform(df)

    stop_title = StopWordsRemover(inputCol="title_tok", outputCol="title_nostop")
    stop_rev   = StopWordsRemover(inputCol="review_tok", outputCol="review_nostop")
    df = stop_title.transform(df)
    df = stop_rev.transform(df)

    df = df.withColumn("title_stem", stem_udf(col("title_nostop")))
    df = df.withColumn("review_stem", stem_udf(col("review_nostop")))

    df = CV_TITLE.transform(df)
    df = IDF_TITLE.transform(df)

    df = CV_REVIEW.transform(df)
    df = IDF_REVIEW.transform(df)

    df = ASSEMBLER.transform(df)


    pred_nb  = NB.transform(df).collect()[0]["prediction"]
    pred_lr  = LR.transform(df).collect()[0]["prediction"]
    pred_svm = SVM.transform(df).collect()[0]["prediction"]

    votes = [pred_nb, pred_lr, pred_svm]
    final = max(set(votes), key=votes.count)

    sentiment = "Positive" if final == 1 else "Negative"

    return render_template("index.html",
                       prediction=sentiment,
                       title=title,
                       review=review)



def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

