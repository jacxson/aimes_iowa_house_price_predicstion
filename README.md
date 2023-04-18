# Project 2 - Ames House Price Prediction - Linear Regression
---
## Overview:
For this project I have been hired by a mock real estate investment advisory group to help develop a better house price prediction model for the Ames Iowa region. The setting is around the end of 2010, in a housing market that is still reeling from the impacts of the 2008 crash that has left investors uncertain about how to accurately price houses, given the recent volatility. The firm currently uses a relatively simple model that served them well in the past and provided for easy interpretability, but given the circumstances, a more robust model is needed to safely navigate these waters. 

The success of this project will be measured by:
1) Delivering a model with a significant (30-40%) increase in performance over the current model in use.
2) Conveying the strengths and limitations of the model in such a way that it can be useful to incread the profitability and reduce the risk of real estate investment ventures moving forward. 

## Benchmark Model:
The Benchmark model currently in use is an OLS Linear Regression model, trained on 4 features: Square Footage, Lot Size, Overall Condition of the Home, and the Year the home was built. While the model is extremely simple, it boasts an **r2 score of around 0.69** and a **Mean Absolute Error of 30986.** While these numbers are not extraordinary, they do translate to a 50% performance increase over the null model in terms of Mean Absolute Error. This model also has the advantage of being extremely interpretable, given the relatively small number of features. 

Aside from the somewhat low r2 score, one of the primary weaknesses of the benchmark model is the violation of assumptions required in order to be confident about making inferences based on the model. In the image below, you will see that the residuals follow a clear pattern that is curvilinear. Overcoming this heteroskedasticity will be one of my primary objectives as I consider how to approach modeling this data. 

![](/images/benchmark_resids.png)

## My Approach:
With the aforementioned weaknesses of the benchmark model in mind, I knew that I wanted to epxlore log transformations of the target variable as well as of certain features in order to get a more random distribution of residuals across the spectrum of prices. Looking at the distribution of the sale prices below, it is clear that the data is skewed. 

![](/images/sale_price_skew.png)

After a log transformation of sale price, however, the distribution approximates something close to a normal distribution. 

![](/images/sale_price_transform.png)

The same transformation is even more striking when applied to some of the features, as with the lot area shown below. Since not all of the feature values were skewed, I ran tests to evaluate whether a log transformation improved the skew of each column and the column's correlation with sale price, and then only appllied the transformation to those features that would benefit from such a trasnformation.

![](/images/lot_are_skew.png)
![](/images/lot_are_transform.png)

## Modeling:
After applying these transformations and making sure our data was squeaky clean, it was time to test out different models. I tested four different iterations of a Linear Regression model using Scikit-Learn:

1) OLS
2) OLS with a custom function to iteratively drop the feature with the highest Variance Inflation Factor in order to find a best fit
3) RidgeCV
4) LassoCV

While the custom model showed signs of promise, it also gave the occasional extreme outlier that significantly hindered its performance. With more work, it may be a contender for a future deployment. The clear winner in this group, however, was the lasso model. 
## Production Model:
The production model had an **alpha value of 0.003593813663804626** and yielded the following scores:
* **r2: 0.89**
* **MAE: 14931**

With an improvement of more than 50% over the benchmark model, I would consider this model to be a success. It certainly has room for improvment, especially at the extreme ends of the price spectrum, but it drastically improvement the prediction performance, and it improved the residual distribution as well. I am fairly confident that this model will generalize well to unseen data. 


Below are the residual plots for the lasso model, showing a marked improvement in all areas over the benchmark model. The first is in a scale of Log Sale Price, and the second is scaled to be in sales price.


![](/images/lasso_resids.png)

![](/images/lasso_resids_exp.png)
