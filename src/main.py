from classifiers import start
from dataset_analysis import dataset_analysis, print_class_balance
from data_analysis import print_features_importance

if __name__ == "__main__":
    #dataset_analysis()
    print_class_balance()

    #predictions, model = start()

    #print_features_importance(model.featureImportances, predictions, 'features')