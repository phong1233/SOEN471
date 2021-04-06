from classifiers import start
from dataset_analysis import dataset_analysis
from data_analysis import print_features_importance

if __name__ == "__main__":
    #dataset_analysis()

    predictions, model = start()

    print_features_importance(model.featureImportances, predictions, 'features')