# Credit Risk Analysis with Supervised Machine Learning

## Overview
This analysis uses supervised learning on a dataset from LendingClub to predict credit risk. The dataset is unbalanced with 99% of loan applications marked as low risk and only 1% classified as high risk. To balance, this analysis oversamples the data using `RandomOverSampler` and `SMOTE`, undersamples with `ClusterCentroids`, and combinatorial samples with `SMOTEENN`. `BalancedRandomForestClassifier` and `EasyEnsebleClassifier` are also used to further reduce bias.

## Resoucres
- **Data Source:** LoanStats_2019Q1.csv
- **Software:** Python 3.7.13, Jupyter Notebook 6.4.8, and other packages in `mlenv` virtual environment

## Results

#### RandomOverSampler
With the `RandomOverSampler`, the balanced accuracy score is 65%. The precision and sensitivity for high risk loans is 1% and 62%, respectively. While the low risk loans' precision is 100% and sensitivity is 68%.

| Balanced Accuracy Score | Imbalanced Classification Report |
|-------------------------|----------------------------------|
|![Screen Shot 2022-10-01 at 1 53 32 PM](https://user-images.githubusercontent.com/106405775/193424184-2e8a77cb-0056-455c-bb98-9bd5f910c434.png)| ![Screen Shot 2022-10-01 at 1 53 44 PM](https://user-images.githubusercontent.com/106405775/193424190-c6584168-e5ca-41b8-9f72-54f57633baae.png)|

#### SMOTE
Similar to the previous oversampling model, the `SMOTE` model's balanced accuracy score is 64%. The high risk loans' precision and sensitivity are 1% and 63%. While the low risk loans' precision and sensitivity are 100% and 66%. 

| Balanced Accuracy Score | Imbalanced Classification Report |
|-------------------------|----------------------------------|
|![Screen Shot 2022-10-01 at 1 58 45 PM](https://user-images.githubusercontent.com/106405775/193424251-e4a48c2e-1dc9-4624-9ce8-e9549519f1b4.png)| ![Screen Shot 2022-10-01 at 1 58 51 PM](https://user-images.githubusercontent.com/106405775/193424257-84edef84-4634-4cb5-be14-b94617c3c3c3.png) |

#### ClusterCentroids
The previous two models oversampled the data to create a balanced dataset. Conversely, the `ClusterCentroids` undersamples the dataset to create a balanced dataset. This model's balanced accuracy score is lower than both oversampling models' at 52%. The high risk loans' precision is still 1% and the sensitivity is also similar at 60%. The low risk loans' precision is 100% and the sensitivity is 43%.

| Balanced Accuracy Score | Imbalanced Classification Report |
|-------------------------|----------------------------------|
|![Screen Shot 2022-10-01 at 2 18 50 PM](https://user-images.githubusercontent.com/106405775/193424965-b06a7bce-4977-47e4-8a49-63749674e7ac.png)| ![Screen Shot 2022-10-01 at 2 18 58 PM](https://user-images.githubusercontent.com/106405775/193424977-cb5bfb1b-f3ea-495f-a485-0affc5d06669.png) | 

#### SMOTEENN
The `SMOTEENN` model over and undersamples to create a balanced dataset. The balanced accuracy score is 64%. The high risk loans' precision is still 1% however the sensitivity is higher than the previous three models at 71%. The low risk loans' precision is 100% and sensitivity rate is also higher at 56%.

| Balanced Accuracy Score | Imbalanced Classification Report |
|-------------------------|----------------------------------|
|![Screen Shot 2022-10-01 at 2 20 00 PM](https://user-images.githubusercontent.com/106405775/193425010-ab6fa9d0-eaaf-454e-b3fa-e9ada1dd9f14.png)| ![Screen Shot 2022-10-01 at 2 20 07 PM](https://user-images.githubusercontent.com/106405775/193425017-35ba51df-1e6b-4af9-9510-40c7689eb338.png)| 

#### BalancedRandomForestClassifier
The `BalancedRandomForestClassifier` model randomly undersamples each bootstrap to balanace the dataset. The balanced accuracy score is higher than the previous four models at 79%. The high risk loans' precision is low at 4% with 67% sensitivity. The low risk loans' precision is 100% with 91% sensitivity.

| Balanced Accuracy Score | Imbalanced Classification Report |
|-------------------------|----------------------------------|
|![Screen Shot 2022-10-01 at 2 37 53 PM](https://user-images.githubusercontent.com/106405775/193425560-a0bb4b3c-91be-4d4e-b7de-ebf40907e180.png)| ![Screen Shot 2022-10-01 at 2 38 07 PM](https://user-images.githubusercontent.com/106405775/193425573-9447edd3-9b99-45cb-a835-d0c9e296408f.png) |

#### EasyEnsebleClassifier
The `EasyEnsembleClassifier` model's balanced accuracy score is the highest of the six models at 93%. The high risk loans' precision is 7% with 91% sensitivity. The low risk loans' precision is 100% with 94% sensitivity.

| Balanced Accuracy Score | Imbalanced Classification Report |
|-------------------------|----------------------------------|
|![Screen Shot 2022-10-01 at 2 38 20 PM](https://user-images.githubusercontent.com/106405775/193425593-acb632ba-1f51-413e-8643-9026ab69af7d.png)| ![Screen Shot 2022-10-01 at 2 38 25 PM](https://user-images.githubusercontent.com/106405775/193425601-0ad66694-6042-40c3-834a-a1a3be9f1e06.png)|

## Summary
Of the six models, the ensemble models better predicted credit risk as demonstrated with their higher balanced accuracy scores and sensitivities for high risk loans. Specifically, the `EasyEnsembleClassifier` model has a 91% sensitivity, meaning it detects almost all high risk credit. Unfortunately, all six models show weak precision to determine when the credit risk is high. This means low risk loans are falsely detected as high risk which could disrupt potential revenue for the bank. As such, I would not recommend any of these models to predict credit risk. 
