## 🏠 Housing Price: Pipelines with Custom Transformer
### Summary
The purpose of this project is to predict the Sale Price. Firstly, I perform exploratory data analysis which indicate me how to prepare the data. To have more automation, I decided to make pipelines with custom transformers. After bundled all the pipelines, I use hyperparameter tuning and finally compute the score.
## Project Description

### Technologies
* Python
* Scikit-Learn
* Pandas
* Seaborn
* Matplotlib
* Numpy
* Jupyter
* SciPy
* category_encoders
* xgboost

## Get Data
Dataset is from Kaggle competition about housing prices in Ames.
https://www.kaggle.com/competitions/home-data-for-ml-course/data

## Exploratory Data Analysis
I looked at the data by dividing them into univariate and bivariate analysis.

#### Univariate Analysis - Skew

High skewness of variables will be normalized in feature engineering.

![image](https://user-images.githubusercontent.com/61654792/175182143-28a8ca71-f2bf-4d51-9087-8555efa4b8be.png)

#### Bivariate Analysis - Outliers and Multicollinearity

A few outliers need to be removed.

![image](https://user-images.githubusercontent.com/61654792/175182639-30c6d78b-3a84-4bb2-8f8e-c7bc658ce0ce.png)

Highly correlated variables will be dropped in feature engineering to avoid multicollinearity.

![image](https://user-images.githubusercontent.com/61654792/175182354-587a5a6b-aa05-43a9-a518-a0c4b9989635.png)


## Missing Data

Columns 'PoolQC', 'MiscFeature' and 'Alley' has too many missing values. They will be dropped in feature engineering.

![image](https://user-images.githubusercontent.com/61654792/175182266-cdc48d98-cd44-48ff-86f8-3c142211b66d.png)

## Prepare the Data

In this section, I remove outliers and columns which have too many missing values and can lead to multicollinearity.

## Feature Engineering
 
I divided this section by numerical columns and categorical columns.
All functions I created can be input in steps of pipeline.
#### Numerical Columns
* CustomImputer
* AddAttributes
* DropCorrFeatures
* SkewedFeatures

![image](https://user-images.githubusercontent.com/61654792/175185213-4c52f000-4542-4771-a871-4a71d3ac375a.png)

![image](https://user-images.githubusercontent.com/61654792/175185315-a8cb1cb9-c60d-4a00-8362-fdda9368f2bf.png)


#### Categorical Columns
* CustomImputer
* OrdinalEncoder

![image](https://user-images.githubusercontent.com/61654792/175185411-c9f11d37-5bb8-41f7-941b-7bf0e1c5e338.png)
![image](https://user-images.githubusercontent.com/61654792/175185419-61fbc4c2-2846-44f7-a49c-ce8a0a763d9a.png)

## Pipelines
The most important and satisfying section. Pipelines with which we can combine all data preprocessing!

![image](https://user-images.githubusercontent.com/61654792/175185528-466ea001-6c54-47ed-8bb7-18176811c7af.png)
![image](https://user-images.githubusercontent.com/61654792/175185547-c54b6284-e954-40bc-99ae-90f3d3fb041a.png)

## Modeling
I use XGBRegressor and hyperparameter tuning to evaluate score.

MAE: 14522.152

