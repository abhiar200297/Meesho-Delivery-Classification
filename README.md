# Meesho-Delivery-Classification
This model made for a Meesho seller to check their product will deliver or return.
# Problem 
A businessman selling his products on Meesho. Many times products get returned by the customer and most of the time products are mismatched from the original or some kind of fraud happens with the seller. In this case the seller lost his many products. If return is genuine then also he loses the packaging cost and courier cost.  Seller wants a model that predicts that order will deliver to the customer or return back. This model will work as an alert alarm for him. He also wants to know that from which state he is getting more orders and which gender is giving more orders. 

# Introduction
In this model we try to predict which of the product will deliver and which one will return back. In this notebook we have done EDA for seller so that he get all information from the past order that how many product delivered, from which state he is getting more order, is data is biased to one gender or not,which product has molre returning and created different model to check the accuracy.

# Libraries required:

Pandas,Numpy,Sklearn,Matplotlib,Seaborn,Plotly

# Methadalogy

data Cleaning:- Many data are duplicated that should be remove to create good model. Show i remove the duplicated data.
Drop the all irrelevant columns so we can't have to work on irrelevant data.
there was total 1627 quantity sold by the seller in from September 2021 to January 2022.
There was no null values but some outliers was present but we will take it as part of data because ouliers are usefull in this kind of data.


Data Visualisation:-
In total data Delivered - 1488 Return - 104, Dataset is imbalance so we need to work to get best accuracy.
![download](https://user-images.githubusercontent.com/99074712/158385420-6ec7f025-caf3-42d1-a305-bc91ce557ffa.png)

Customer is balanced with Gender. It is not biased for one gender. Male = 833 and Female 759![download](https://user-images.githubusercontent.com/99074712/158385847-f6aff3c8-b587-45cb-bb1e-a9e095478a52.png)

Same product is categorised with different name so need to categorise in same category. There was almost 45 category of product but i reduced it to 15 possible category.
Returning of product on the basis of gender:-
         Product Name  Gender  counts
0               Baby Mosquito Net  Female       1
1               Baby Mosquito Net    Male       1
2                      Beanie cap    Male       1
3                         Blanket  Female       3
4                         Blanket    Male       4
5                       Butti set  Female       1
6   DIXCY SCOTT VICTOR DURBY VEST    Male       2
7           Designer Mosquito Net  Female       4
8           Designer Mosquito Net    Male       4
9                          Diaper    Male       1
10           JOCKEY PRINT  BRIEFS    Male       1
11                    JOCKEY VEST    Male       2
12         MACROMAN PRINTED TRUNK    Male       1
13                    Mangalsutra    Male       1
14                   Mosquito net  Female      29
15                   Mosquito net    Male      45
16                          Panty    Male       1
17                      Water rod  Female       1
18                      Water rod    Male       1

# handling categorical values:-
We use Frequency coding to encode Reseller state.
With one hot encoding we deal gender.
Category for Size is not high so i prefer one hot encoding for it.
Still products categories are high so we can't go with one hot encoding it will create many features.
For Product name i choosed mean ordinal encoding.

# Data Selection:
Divided the data in independent variable and dependent variable.
y is target set and x is independent features.
Divided the data in training set and testing set.

Correlation between features dataset.
![download](https://user-images.githubusercontent.com/99074712/158428979-1e93547c-a420-4855-b8c0-53be99b4a58d.png)

# Model prepration
Data set was imbalanced, Not standarised and some outliers were present so Random Forest Classifier perform better in this condition.
Going with hyperparameter tunning taking parameter = {'n_estimators':[10,20,30,40,50,80,90,100,120,140,150,160],'criterion':['gini','entropy']}
Training score was 92.60% and best parameters are 'criterion': 'gini', 'n_estimators': 160
Testing accuracy is also 92.47% which shows model is neither underfitted nor overfitted.
Confusion matrix
array([[294,   3],
       [ 21,   1]]
False positive is high that need to be decrease. False positives are acutually negative but model is predicting is positive that is wrong prediction.

# Conclusion:
Model perform well in with Random Forest Classifier but we need to work on precision so False positive should reduce.
