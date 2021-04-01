import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import datediff
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler


def init_spark():
    spark = SparkSession \
        .builder \
        .appName('Soen 471') \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def get_data_as_dataframe():
    schema = StructType([StructField("ID", IntegerType(), True),
                     StructField("name", StringType(), True), 
                     StructField("category", StringType(),True), 
                     StructField("main_category", StringType(),True),
                     StructField("currency", StringType(), True),
                     StructField("deadline", DateType(), True),
                     StructField("goal", IntegerType(), True), 
                     StructField("launched", DateType(), True), 
                     StructField("pledged", DoubleType(), True), 
                     StructField("state", StringType(), True),
                     StructField("backers", IntegerType(), True),
                     StructField("country", StringType(), True),
                     StructField("usd pledged", DoubleType(), True),
                     StructField("usd_pledged_real", DoubleType(), True),
                     StructField("usd_goal_real", DoubleType(), True)])
    
    spark = init_spark()
    df = spark.read.format("csv") \
        .option("header", True) \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .schema(schema) \
        .load('../data/ks-projects-soen471.csv')
    return df


def get_clean_data():
    columns_to_drop = ['usd_pledged_real', 'usd pledged', 'backers', 'pledged', 'goal', 'currency', 'deadline', 'launched']
    
    df = get_data_as_dataframe()
    cleaned = df.filter((df.state == 'successful') | (df.state == 'failed')) \
                .withColumn('duration_in_days',  datediff(df['deadline'], df['launched'])) \
                .drop(*columns_to_drop)

    cleaned.show()

    return cleaned


def data_preparation():
    df = get_clean_data()
    cols = df.columns
    categoricalColumns = ['category', 'main_category', 'country']
    stages = []
    
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]

    label_stringIdx = StringIndexer(inputCol = 'state', outputCol = 'label')
    stages += [label_stringIdx]

    numericCols = ['usd_goal_real', 'duration_in_days']
    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    selectedCols = ['label', 'features'] + cols
    df = df.select(selectedCols)

    return df


def decision_tree_classifier():
    df = data_preparation()
    train, test = df.randomSplit([0.7, 0.3])

    dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=5, impurity="gini")
    dtModel = dt.fit(train)
    predictions = dtModel.transform(test)
    predictions.show(30)

    evaluate_predictions(predictions)


def random_forest_classifier():
    df = data_preparation()
    train, test = df.randomSplit([0.7, 0.3])

    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=5, impurity="gini")
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)
    predictions.show(30)

    evaluate_predictions(predictions)


def evaluate_predictions(predictions):
    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Accuracy = %g " % accuracy)


def start():
    random_forest_classifier()
