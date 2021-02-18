# SOEN471

# Abstract

Starting a company from scratch is no easy feat, entrepreneurs have to build a product, gain clients and backers. This is where many projects fail to see the light, they are missing the funds necessary to expand and establish a grossing business. The website [Kickstarter](https://www.kickstarter.com/) makes this task easier by allowing users to support, give money to, projects that they deem worthy. Unfortunately, not all projects succeed in achieving their fund goal.
Using the [Webrobots scrapper dataset of Kickstarter projects](https://webrobots.io/kickstarter-datasets/), we aim to use supervised learning, using specific characteristics, to determine if a project will be deemed to succeed or be forgotten in the abyss of failed Kickstarter projects.

# Introduction

## Context

In the age of technology, it is as easy as ever to start a project. There is a huge variety of projects that are started everyday. Although they may vary, most projects need a fair amount of funding in order to get started.  As such many projects are looking for funds through websites such as Kickstarter every day.

## Problem

Unfortunately, looking at the statistics available on Kickstarter, we can observe that the vast majority of projects are unsuccessfully funded and end up forgotten. Over 514,634 launched projects, only 196,396 projects are successfully funded, only 38.16% of projects succeed. But, projects do not fail necessarily because they are bad. There are multiple examples of projects that failed to get the necessary funds on Kickstarter but still ended up getting the necessary funds after a relaunch. There are also many terrible projects that end up not only meeting the funding objectives but also exceeding them by a large margin. For example, in 2016, a potato salad received a total funding of $55,492 pledging only $10 to begin with. As such it is hard to predict whether or not a project has everything in order to maximise the possibility of getting funded successfully.

## Objective

Our objective is to create a prediction system using supervised learning that will predict whether or not a project is likely to meet its funding goal. Using two distinct supervised learning techniques, we will compare the outcomes of those techniques to study the differences between them and ultimately determine which one is the best.

## Related Work

The idea of analysing Kickstarter projects for determining if a project will be successful or not is not a novel idea, it has been done by multiple independent researchers. We hope to achieve better results using the techniques learned in class. Additionally, we will use Apache Spark to compute our models and predictions, something that has not been done a lot in the past.

# Materials and Methods

## Dataset

The dataset we will use is from a company named Web Robots. They are a web crawling company that creates datasets in order to solve big data challenges. The dataset that we will be using throughout this project is the [Kickstarter dataset](https://webrobots.io/kickstarter-datasets/). It is free to use and contains a lot of data (1Gb). Since Web Robots were crawling the Kickstarter website since 2015, we will only use the most recent data available which correspond to the latest scrape, 2021-02-11. It is available in either a JSON format or a CSV format. For this project we will use the CSV format. This dataset contains roughly 210 000 projects. The label that we would like to predict is the Status which in our dataset is either failed, successful, live or canceled. For the purpose of our prediction we will only try to predict the labels: successful and failed by only keeping the failed and the successful projects. We will mainly look at the following columns: category, converted_pledged_amount, country, created_at, deadline, goal, staff_pick and spotlight. There are more columns available but those are the main ones that we will be using.

## Methods

To analyse this dataset, we will use two different classification algorithms that will determine if a project, using the characteristics above, can be classed as successful or a failure.
The first technique we will use is the [Decision tree classifier](https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-trees). This algorithm is a perfect choice to create a baseline for our two technique’s performance because it is simple and easy to understand and interpret. A decision tree can be used to visually and explicitly represent decisions and decision making, which will be interesting in the scope of our project. We will be able to understand what makes a project successful or not.

The second technique we will use is the [Multi-layer perceptron classifier](https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier). This classifier is based on a feedforward neural network which is a neural network where the information moves in only one direction, from the input nodes toward the output nodes. The connections between nodes do not form a cycle. This technique is quite advanced and we believe it should bring better results than our baseline decision tree.

Once we have used both methods to compute our predictions, we will be able to compare them in how they operate and decide which one is the more suited to accurately predict if a project will be successful or not.

# Contributors

| Name                   | Github                                                |
|------------------------|-------------------------------------------------------|
| Phong Le               | (phong1233)[https://github.com/phong1233]             |
| Sébastien Blain-Nadeau | (sebastien-blain)[https://github.com/sebastien-blain] |
