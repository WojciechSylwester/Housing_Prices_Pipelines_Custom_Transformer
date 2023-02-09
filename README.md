## 🏠 Housing Price: Pipelines with Custom Transformer
### Summary

![Housing_Custom_Pipeline](https://user-images.githubusercontent.com/61654792/217819437-ee6cb555-e4f9-4709-b2c6-5e343778757f.png)

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
Dataset derives from Kaggle competition about housing prices in Ames.

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
```python
class CustomImputer(BaseEstimator, TransformerMixin):

    def __init__(self, imputer, strategy, fill_value=0):
        
        self.imputer = imputer
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        
        self.imputer = self.imputer(strategy=self.strategy, fill_value = self.fill_value)
        self.imputer.fit(X, y)
        return self

    def transform(self, X):
        
        X_imp_tran = self.imputer.transform(X)
        X_imputer = pd.DataFrame(X_imp_tran, index=X.index, columns=X.columns)
        return X_imputer
```

```python
class SkewedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=0.8):
        
        self.skew_threshold = skew_threshold
    
    def fit(self, X, y=None):

        skew_features = X.select_dtypes(exclude='object').apply(lambda x: skew(x))
        self.skew_features_high = skew_features[abs(skew_features) > self.skew_threshold].index
        return self
    
    def transform(self, X):
        
        X[self.skew_features_high] = np.log1p(X[self.skew_features_high])
        return X
````


#### Categorical Columns
* CustomImputer
* OrdinalEncoder

```python
CentralAir_map = {'Y': 1, 'N': 0}
Street_map = {'Pave': 1, 'Grvl': 0}

binary_mapping = [{'col': col, 'mapping': globals()[col + '_map']}
                     for col in cat_bin]
```

## Pipelines
The most important and satisfying section. Pipelines with which we can combine all data preprocessing!

```python
# Preprocessing for numerical data
num_transformer = Pipeline(steps=[
    ('num_imputer', CustomImputer(SimpleImputer, strategy='median')),
    ('adder', AddAttributes()),
    ('drop_corr', DropCorrFeatures()),
    ('skew_func', SkewedFeatures()),
    ('std_scaler', StandardScaler())
])

# Preprocessing for categorial data
cat_transformer_ordinal = Pipeline(steps=[
    ('cat_ordinal_imputer', CustomImputer(SimpleImputer, strategy='constant', fill_value='NA')),
    ('ordinal_encoder', ce.OrdinalEncoder(mapping = ordinal_mapping))
])

```

```python
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_col_transform),
        ('cat_ordinal', cat_transformer_ordinal, cat_ordinal),
        ('cat_ordinal_num', cat_transformer_ordinal_num, cat_ordinal_num),
        ('cat_bin', cat_transformer_bin, cat_bin),
        ('cat_nominal', cat_transformer_nominal, cat_nominal)
    ], remainder='passthrough')
```

## Modeling
I use XGBRegressor and hyperparameter tuning to evaluate score.

There is a field to improve the score by adjusting more hyperparameters.
