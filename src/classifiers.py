from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, \
    MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from data_preparation import data_preparation


def decision_tree_classifier():
    df = data_preparation()
    train, test = df.randomSplit([0.7, 0.3])

    dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=5, impurity="gini")
    dtModel = dt.fit(train)
    predictions = dtModel.transform(test)

    evaluate_predictions(predictions)


def random_forest_classifier():
    df = data_preparation()
    train, test = df.randomSplit([0.7, 0.3])

    rf = RandomForestClassifier(featuresCol='features', labelCol='label', maxDepth=5, impurity="gini")
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)

    evaluate_predictions(predictions)


def gradient_boosted_tree_classifier():
    df = data_preparation()
    train, test = df.randomSplit([0.7, 0.3])

    gb = GBTClassifier(featuresCol='features', labelCol='label', maxDepth=5)
    gbModel = gb.fit(train)
    predictions = gbModel.transform(test)

    evaluate_predictions(predictions)


def gradient_boosted_tree_classifier_with_cross_validation():
    df = data_preparation()

    train, test = df.randomSplit([0.7, 0.3])

    gbt = GBTClassifier()
    evaluator = BinaryClassificationEvaluator()

    paramGrid = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [2, 5, 10])
                 .addGrid(gbt.maxIter, [5, 100])
                 .build())
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

    cvModel = cv.fit(train)
    predictions = cvModel.transform(test)

    evaluate_predictions(predictions)


def multi_layer_perception_classifier():
    df = data_preparation()
    train, test = df.randomSplit([0.7, 0.3])

    layers = [196, 3, 4, 2]

    mlp = MultilayerPerceptronClassifier(labelCol='label', featuresCol='features', maxIter=100, layers=layers,
                                         blockSize=128)
    mlpModel = mlp.fit(train)
    predictions = mlpModel.transform(test)

    evaluate_predictions(predictions)


def evaluate_predictions(predictions):
    predictions.show()

    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Accuracy = %g " % accuracy)


def start():
    gradient_boosted_tree_classifier_with_cross_validation()
