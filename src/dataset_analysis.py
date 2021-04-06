import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data_preparation import get_clean_data


def dataset_analysis():
    df = get_clean_data()

    categories_failed = df.filter(df.state == 'failed'). \
        groupBy('category'). \
        count(). \
        orderBy('count', ascending=False). \
        take(15)
    categories_failed_count = [r['count'] for r in categories_failed]
    categories_failed_cat = [r['category'] for r in categories_failed]

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(categories_failed_cat, categories_failed_count)
    plt.xticks(rotation=90)

    ax.set_ylabel('Num of projects')
    ax.set_title('Top 15 failed categories')
    plt.show()

    categories_successful = df.filter(df.state == 'successful'). \
        groupBy('category'). \
        count(). \
        orderBy('count', ascending=False). \
        take(15)

    categories_successful_count = [r['count'] for r in categories_successful]
    categories_successful_cat = [r['category'] for r in categories_successful]

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(categories_successful_cat, categories_successful_count)
    plt.xticks(rotation=90)

    ax.set_ylabel('Num of projects')
    ax.set_title('Top 15 successful categories')
    plt.show()

    categories_general = df.groupBy('category').count().orderBy('count', ascending=False).take(25)
    categories_general_count = [r['count'] for r in categories_general]
    categories_general_cat = [r['category'] for r in categories_general]

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(categories_general_cat, categories_general_count)
    plt.xticks(rotation=90)

    ax.set_ylabel('Num of projects')
    ax.set_title('Top 25 categories')
    plt.show()
