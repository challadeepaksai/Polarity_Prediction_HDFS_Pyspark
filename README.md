# 🌟 Amazon Review Polarity Prediction using PySpark and Hadoop

**A Big Data and Machine Learning Project to Predict the Polarity (Positive/Negative) of Amazon Customer Reviews.**

---

## 📝 Overview

This project serves as a hands-on learning experience to implement a complete Machine Learning pipeline on a **Big Data stack**. It uses **PySpark** for distributed data processing and model training, and is designed to run on a **Hadoop cluster** (leveraging HDFS for storage and Yarn for resource management).

The core task is to classify Amazon product reviews as either positive or negative, utilizing a dataset sourced from Kaggle.

---

## 💾 Dataset

The dataset used for this project is the **Amazon Reviews** dataset available on Kaggle.

* **Source:** `https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data`

The raw data is expected to be loaded onto the **Hadoop Distributed File System (HDFS)** before running the PySpark jobs.

---
---

## ⚙️ Setup and Prerequisites

### Core Technology Stack

* **Python 3.10**
* **Apache Spark** (configured to run in cluster mode,on Yarn)
* **Apache Hadoop** (with HDFS and Yarn services running)

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Project Directory]
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Ingestion:**
    Ensure the Amazon reviews dataset is uploaded to your HDFS cluster.
    ```bash
    # Example command to upload data to HDFS
    hdfs dfs -put /local/path/to/reviews.csv /user/hadoop/amazon_reviews/
    ```

---

## 💻 Usage

### 1. Model Training and Evaluation (PySpark/Yarn Jobs)

The PySpark scripts (`script.py` and `eval.py`) handle all distributed ML tasks, including text processing (Tokenization, Stop Word Removal, TF-IDF) and model training using Spark MLlib.

To execute a PySpark job (Training or Evaluation) on the cluster:

```bash
# The place_job.sh script will use spark-submit to push the job to Yarn.
# You will likely need to pass arguments specifying the HDFS input/output paths.

# Example to start Training:
./place_job.sh script.py