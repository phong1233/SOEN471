import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import OneHotEncoder, StringIndexer


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
    df = get_data_as_dataframe()
    cleaned = df.filter((df.state == 'successful') | (df.state == 'failed'))
    return cleaned


def decision_tree_classifier():
    data = get_clean_data()

    # (training, test) = data.randomSplit([0.8, 0.2])
    
    # model = DecisionTree.trainClassifier(training, numClasses=2, categoricalFeaturesInfo={},
    #                                  impurity='gini', maxDepth=5, maxBins=32)
    return


def start():
    decision_tree_classifier()
