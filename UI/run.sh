spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 1 \
    --executor-memory 5G \
    --driver-memory 3G \
    --executor-cores 1 \
    app.py
