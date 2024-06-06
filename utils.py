import pandas as pd
import numpy as np

 
def check_ifreal(y: pd.Series) -> bool:
    try :
        if (y.dtype.name == 'category') :
            return True
    except ValueError :
        pass
    if isinstance(y, pd.DataFrame):
        # If y is a DataFrame, check each column separately
        for column in y.columns:
            try:
                pd.to_numeric(y[column], errors='raise')
            except ValueError:
                return False
    else:
        try:
            pd.to_numeric(y, errors='raise')
        except ValueError:
            return False
    return True


def impurity(Y: pd.Series,criterion) -> float:
    if criterion == "entropy":
        value_counts = Y.value_counts(normalize=True)
        entropy_value = -np.sum(value_counts * np.log2(value_counts + 1e-6))
        return entropy_value
    
    elif criterion == "gini_index":
        gini_ind = 1
        labels = Y.unique()
        for label in labels :
            p = len(Y[Y == label])/len(Y)
            gini_ind -= p*p
        return gini_ind
    
    else:
        mse = np.var(Y)
        return mse


def information_gain(Y: pd.Series, attr: pd.Series, criterion) -> tuple:
    dict1 = {'attr' : attr, 'Y' : Y}
    df = pd.concat(dict1, axis=1)
    if (not check_ifreal(attr)) : #I/p discrete O/P discrete
        unique_attr = attr.unique()
        impurity_1 = impurity(Y,criterion)
        impurity_2 = 0
        for attr in unique_attr:
            df1 = df[df['attr'] == attr]
        
            impurity_2 += impurity(df1["Y"],criterion)*len(df1["Y"])
            
        return unique_attr, impurity_1 - impurity_2/len(Y)
    
    else: # I/P Real O/P discrete
        
            sort_df = df.sort_values("attr").reset_index(drop = True)
            impurity_1 = impurity(Y,criterion)
            max_val = -1
            for ind in range(0,len(sort_df) - 1) :
                if Y[ind] != Y[ind+1] :
                
                    poss_df1 = sort_df[:ind+1]
                    poss_df2 = sort_df[ind+1:]
                
                    impurity_2 = (impurity(poss_df1["Y"],criterion)*len(poss_df1)) + (impurity(poss_df2["Y"],criterion)*len(poss_df2))
                    x = impurity_1 - impurity_2/len(Y)
                    
                    if(x >= max_val) :
                        max_val = x
                        ind1 = ind
                    
            return([ind1], max_val)


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
   
    inf_gains = []
    splits = []
    for feature in features:
        split, in_gain = information_gain(y, X[feature],criterion)
        inf_gains.append(in_gain)
        splits.append(split)
    i = np.argmax(inf_gains)
    opt_split_attribute = features[i]
    split = splits[i]
    return split, opt_split_attribute


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    df = X.join(y)
    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if not check_ifreal(df[attribute]) : # Discrete input
        l = [] # List of dataframes
        for v in value :
            l.append(df[df[attribute] == v])
        return l
    
    # Real input
    sort_df = df.sort_values(by=attribute).reset_index(drop=True)
    return [sort_df[:value[0]+1], sort_df[value[0]+1:].reset_index(drop=True)]