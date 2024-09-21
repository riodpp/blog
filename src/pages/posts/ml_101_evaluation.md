---
layout: ../../layouts/post.astro
title: "Metrics to Evaluate Binary Classifier"
pubDate: 2024-09-10
description: "In this Article, I want to share about metrics that we can use to evaluate binary classifier"
author: "riodpp"
excerpt: I have learn some of concept about classification. Now, I learn  about how to evaluate binary classifier. Is it enough to just use the accuracy. Is there any other metrics that we can use?
image:
  src:
  alt:
tags: ["ml", "ml engineer"]
---

In the previous article we already learn about binary classifier. For summary, a binary classifier is a type of machine learning model that categorizes data into one of two distinct classes. It is used in scenarios where the outcome is binary, meaning there are only two possible labels or categories, such as "spam" vs. "not spam" in email filtering, "positive" vs. "negative" in sentiment analysis, or "disease" vs. "no disease" in medical diagnosis.

We also calculate the accuracy of our model with counting proportion of correctly predicted instance from total number of instance.

\[ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \]

But, is it enough to just using accuracy to evaluate binary classifier? 
To answer this, consider a dataset for detecting a rare disease where only 1% of the samples are positive (disease) and 99% are negative (no disease). Let's say we have 10,000 samples:

- Positive samples (disease): 100
- Negative samples (no disease): 9,900

Suppose we have a binary classifier that always predicts "no disease" (negative). The accuracy of this classifier would be:

\[ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} = \frac{9,900}{10,000} = 99\% \]

Despite the high accuracy, this classifier is useless because it never identifies the positive cases. This is why accuracy is not a suitable metric for imbalanced datasets. Instead, metrics like precision, recall, F1 score, and AUC-ROC provide a better evaluation of the model's performance in such scenarios. But, before we dive to other metrics, I want to share about confussion table for the bridge to the other metrics.

## Confussion Table
Confussion table used to evaluate the performance of a classification model. It provides a detailed breakdown of the model's predictions compared to the actual outcomes, allowing you to see not only the number of correct and incorrect predictions but also the types of errors made.

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)  | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN)  |

Definitions
True Positive (TP): The number of instances correctly predicted as positive.
False Positive (FP): The number of instances incorrectly predicted as positive.
True Negative (TN): The number of instances correctly predicted as negative.
False Negative (FN): The number of instances incorrectly predicted as negative.

If you still confuse just like the name of the table :) it's ok. We can go through with example

Consider a dataset for predicting whether a patient has a disease (Positive) or not (Negative). Here are the actual and predicted outcomes for 10 patients:

| Patient | Actual Outcome | Predicted Outcome |
|---------|----------------|-------------------|
| 1       | Positive       | Positive          |
| 2       | Negative       | Negative          |
| 3       | Positive       | Negative          |
| 4       | Negative       | Negative          |
| 5       | Positive       | Positive          |
| 6       | Negative       | Positive          |
| 7       | Positive       | Positive          |
| 8       | Negative       | Negative          |
| 9       | Positive       | Negative          |
| 10      | Negative       | Negative          |

Using the above dataset, we can construct the confusion table:

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | 3 (TP)             | 2 (FN)               |
| **Actual Negative** | 1 (FP)             | 4 (TN)               |

Easy isn't it? As you already understand this concept we can go through other metrics.

## Recall
Recall, also known as sensitivity or true positive rate, is a metric used to evaluate the performance of a classification model. It measures the proportion of **actual positives that are correctly identified by the model**.

\[ \text{Recall} = \frac{TP}{TP + FN} \]

Using the confusion table from the previous example, the recall can be calculated as follows:


\[ \text{Recall} = \frac{3}{3 + 2} = \frac{3}{5} = 0.6 \]

This means that the model correctly identifies 60% of the actual positive cases.

## Precision
Precision is a metric used to evaluate the performance of a classification model, particularly in the context of binary classification. It measures the proportion of **positive predictions that are actually correct**.

\[ \text{Precision} = \frac{TP}{TP + FP} \]

Using the confusion table from the previous example, the precision can be calculated as follows:

\[ \text{Precision} = \frac{3}{3 + 1} = \frac{3}{4} = 0.75 \]

This means that 75% of the instances predicted as positive are actually positive.

Precision is particularly important in scenarios where the cost of false positives is high. For example, in medical testing, a high precision ensures that most of the patients identified as having a condition actually have it, reducing the number of patients who undergo unnecessary further testing or treatment.

## ROC
ROC is a curve used to evaluate the performance of a binary classification model. It plots the True Positive Rate (TPR) or as we know as Recall againts the False Positive Rate (FPR).

Below is the formula of the TPR and FPR

True Positive Rate (TPR): Also known as Recall or Sensitivity, it is calculated as: \[ \text{TPR(Recall)} = \frac{\text{TP}}{\text{TP} + \text{FN}} \]

False Positive Rate (FPR): It is calculated as: \[ \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} \]

The ROC curve is created by plotting the TPR against the FPR at different threshold values. Each point on the ROC curve represents a different threshold, showing the trade-off between sensitivity (recall) and FPR at various threshold levels.

## AUC
AUC or area under the curve of ROC is a value that summarize the performance of the classifier. The AUC range from 0 to 1, with higher value indicating better performance.

## Cross Validation
Cross validation is used to evaluate performance of a machine learning model by partitioning the data into subsets. Some subsets use to train your modek, and validating it on the remaining subsets. One of the famous type of cross-validation is K-Fold Cross-Validation. 
The data at K-Fold will divided into `k` equally sized folds. The model is trained on `k-1` folds and validated on the remaining fold. This process is repeated `k` times, with each fold used exactly once as the validation data. The final performance metric is the average of the metrics from each fold.

## Summary

With many metric that can use to evaluating binary classification, it depends on the specific context and goals of your application. Different metric provide different insights. Using multiple metrics can gove a more comprehensive evaluation. 
For example, Accuracy used when the data are balanced. Precision used when the cost of false positives is high (spam, medical diagnosis). If the prediction is false, this will lead to unnecessary treatments. Recall in the other hands utilize when the cost of negative is high, for example when missing a positive case of disesase will be fatal to the patient.