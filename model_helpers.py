import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Doctrings were created with assistance from Chat-GPT

def namestr(obj, namespace=globals(), select_index=True, index=0):
    """
    Return a list of strings of the variable name(s) assigned to a given object.

    Args:
        obj: object
            The object whose variable name(s) you want to retrieve.
        namespace: dict, optional
            The namespace in which to look for the variable name(s).
            Defaults to the current module's global namespace.

    Returns:
        A list of strings of the variable name(s) to which the object has been assigned.
    """
    
    return [name for name in namespace if namespace[name] is obj]


def drop_max_vif(feature_dataframe, threshold=None):
    """
    Reduces multicollinearity in a feature dataframe by dropping the column with the highest VIF until all columns fall below a given threshold.

    Args:
        feature_dataframe: pandas.DataFrame
            The feature dataframe to be checked for multicollinearity.
        threshold: float or None, optional
            The threshold value for VIF. If a feature's VIF is higher than this threshold, that feature will be removed
            from the dataframe. Default value is None, which sets the threshold to 5 if not specified.

    Returns:
        pandas.DataFrame
            The feature dataframe with multicollinearity removed.
        list
            List of columns that were dropped to facilitate applying transformations to test data.
    """
   
    if threshold == None:
        threshold = 5
        
    df_copy = feature_dataframe.copy()
    dropped_cols = []
    
    
    for i in range(len(df_copy.columns)):
        vifs = {variance_inflation_factor(df_copy.values, i): i for i in range(len(df_copy.columns))}
        max_vif = max(vifs)
        feature = df_copy.columns[vifs[max_vif]]
        
        if max(vifs) > threshold:
            print(f'Dropping {feature} with a VIF of {max_vif}')
            df_copy.drop(columns=feature, inplace=True)
            dropped_cols.append(feature)

        else:
            print(f'Max VIF is {feature} with a VIF of {max_vif}')
            
            return df_copy, dropped_cols


def create_rank_dict(dataframe, feature, target, auto_order=True):
    """
    Creates a dictionary mapping unique values in a column of a dataframe to ordinal values.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the column to create the rank dictionary for.
        column (str): The name of the column in the dataframe to create the rank dictionary for.

    Returns:
        dict: A dictionary mapping unique values in the column to ordinal values using inputs from prompts printed to the console.

    """
    items = []
    order = {}
    
    if auto_order:
        index = dataframe.groupby(feature)[target].median().sort_values().index
        order = {x: i for i, x in enumerate(index)}
    else:
        items = list(dataframe[feature].unique())
        count = 0
        while len(items) > 0:
            if len(items) != 1:
                nxt_item = input(f'Which item from this list should be assigned to ordinal value {count}? {items}')
                if nxt_item not in items:
                    nxt_item = input(f'Invalid input. Please try again. Which item from this list should be assigned to ordinal value {count}? {items}')
                items.remove(nxt_item)
                order.update({nxt_item: count})
                count += 1
            else:
                order.update({items[0]: count})
                items.remove(items[0])
    return order


def ordinalize(dataframe, column, target, rank_dict=None, return_dict=False, suppress_print=False):
    """
    Encodes the categorical variable in a dataframe column as an ordinal variable.

    Args:
        dataframe (pandas.DataFrame): The dataframe to encode.
        column (str): The name of the column to encode.
        rank_dict (dict, optional): A dictionary specifying the ranking order of the categories. If not provided, the create_rank_dict function will be called within the function.
        return_dict (bool, optional): Whether to return a tuple containing the ordinalized column and the rank_dict, instead of only the ordinalized column. Defaults to False.

    Returns:
        pandas.Series or tuple: A copy of the column of the dataframe with the categorical variable encoded as an ordinal variable. If return_dict is True, the function returns a tuple of the ordinalized column and the rank_dict.
    """
    if suppress_print == False:
        print(f'ValueCounts Before Ordinalizing: \n{dataframe[column].value_counts()}\n')
    
    if rank_dict == None:
        rank_dict = create_rank_dict(dataframe, column, target)
    df_copy = dataframe.loc[:, column].copy()
    df_copy = df_copy.map(rank_dict)

    if suppress_print == False:    
        print(f'/nValueCounts After Ordinalizing: \n{df_copy.value_counts()}')
    
    if return_dict == True:
        return df_copy, rank_dict
    
    return df_copy


def box_plot_columns(dataframe, features, target, cols=None):

    
    if cols == None:
        cols = 3
        
    rows = int(np.ceil(len(features)/cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    count = 0
    while count < len(features):
        for row in range(rows):
            for col in range(cols):
                if count < len(features):
                    sns.boxplot(x=dataframe[features[count]], 
                                y=dataframe[target], 
                                ax=axes[row,col], 
                                order=dataframe.groupby(features[count])[target].median().sort_values().index
                               )
                    axes[row,col].set_title(features[count], size=15)
                    count+= 1
                    
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
    
    return None