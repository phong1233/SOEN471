from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, \
    MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from data_preparation import data_preparation


SEED = 40099004


def decision_tree_classifier():
    df = data_preparation()
    train, test = df.randomSplit([0.7, 0.3], seed=SEED)

    dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=10, impurity="gini", seed=SEED)
    dtModel = dt.fit(train)
    predictions = dtModel.transform(test)

    evaluate_predictions(predictions)

    return predictions, dtModel


def random_forest_classifier():
    df = data_preparation()
    train, test = df.randomSplit([0.7, 0.3], seed=SEED)

    rf = RandomForestClassifier(featuresCol='features', labelCol='label', maxDepth=10, impurity="gini", seed=SEED)
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)

    evaluate_predictions(predictions)

    return predictions, rfModel


def gradient_boosted_tree_classifier():
    df = data_preparation()
    train, test = df.randomSplit([0.7, 0.3], seed=SEED)

    gb = GBTClassifier(featuresCol='features', labelCol='label', maxDepth=10, seed=SEED)
    gbModel = gb.fit(train)
    predictions = gbModel.transform(test)

    evaluate_predictions(predictions)

    return predictions, gbModel


def gradient_boosted_tree_classifier_with_cross_validation():
    df = data_preparation()

    train, test = df.randomSplit([0.7, 0.3], seed=SEED)

    gbt = GBTClassifier()
    evaluator = BinaryClassificationEvaluator()

    paramGrid = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [2, 5, 10])
                 .addGrid(gbt.maxIter, [5, 100])
                 .build())
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5, seed=SEED)

    cvModel = cv.fit(train)
    predictions = cvModel.transform(test)

    evaluate_predictions(predictions)

    return predictions, cvModel


def multi_layer_perception_classifier():
    df = data_preparation()
    train, test = df.randomSplit([0.7, 0.3], seed=SEED)

    layers = [196, 3, 4, 2]

    mlp = MultilayerPerceptronClassifier(labelCol='label', featuresCol='features', maxIter=100, layers=layers,
                                         blockSize=128, seed=SEED)
    mlpModel = mlp.fit(train)
    predictions = mlpModel.transform(test)

    evaluate_predictions(predictions)

    return predictions, mlpModel


def evaluate_predictions(predictions):
    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    print('F1 score:', evaluator.evaluate(predictions))

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Accuracy = %g " % accuracy)


def start():
    return decision_tree_classifier()
