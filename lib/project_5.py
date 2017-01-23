import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


def general_process(model, data_dict, train_score=False, test_score=False):
    """ 
        Track data dictionary and model that has been completed step by step.
        
        Parameters:    models: models
                    data_dict: dictionary containing data
        
        Returns:               dictionary of data plus the model that was completed
        
    """
    
    if 'models' in data_dict.keys():
        data_dict['models'].append(model)
    else:
        data_dict['models'] = [model]
       
    if train_score:
        data_dict['train_score'].append(train_score)
        data_dict['test_score'].append(test_score)
    
    return data_dict


def validate(data_dict, random_state=None):
    """ 
        Validates the necessary data (X_train, X_test, y_train, y_test) needed to continue with processing. If any of them are missing it will create train and test sets with train_test_split using X and y.
        
        Parameters:    data_dict: dictionary containing data
                    random_state: int or RandomState
                                  Pseudo-random number generator state used for random sampling
        
        Returns:                  dictionary with necessary data 
        
    """
    
    if 'X_test' or 'y_test' in data_dict.keys():
        pass
    else:
        data_dict = train_test_split(X,y,random_state=random_state)
        
    return data_dict


def load_data_from_database(user='dsi_student', password='correct horse battery staple', url='joshuacook.me', port='5432', database='dsi', table='madelon'):
    """ 
        Connects to PostgreSQL database and loads the data locally using pandas.
        
        Parameters: user: dsi_student
                password: correct horse battery staple
                     url: joshuacook.me
                    port: 5432
                database: dsi
                   table: madelon
        
        Returns:          pandas DataFrame
        
    """
    
    engine = create_engine("postgresql://{}:{}@{}:{}/{}".format(user, password, url, port, database))
    df = pd.read_sql("SELECT * FROM {}".format(table), con=engine)

    return df


def make_data_dict(df,random_state=None):
    """ 
        Takes a pandas DataFrame, identifies target and features, creates train and test sets with train_test_split.
        
        Parameters:           df: pandas DataFrame
                    random_state: int or RandomState
                                  Pseudo-random number generator state used for random sampling
        
        Returns:                  dictionary containing X_train, X_test, y_train, y_test, X, y
        
    """
    
    X = df[[x for x in df.columns if 'feat' in x]]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=random_state)
    data_dict = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'X': X, 'y': y, 'train_score': [], 'test_score': []}

    return data_dict


def general_transformer(transformer, data_dict, random_state=None):
    """ 
        Fits training data then transforms the feature matrix.
        
        Parameters:  transformer: takes any transformer function
                       data_dict: dictionary containing data
                    random_state: int or RandomState
                                  Pseudo-random number generator state used for random sampling
        
        Returns:                  data dictionary with transformed X_train and X_test
        
    """
    data_dict = validate(data_dict, random_state=random_state)
    transformer.fit(data_dict['X_train'], data_dict['y_train'])

    data_dict['X_train'] = transformer.transform(data_dict['X_train'])
    data_dict['X_test'] = transformer.transform(data_dict['X_test'])
    
    data_dict = general_process(transformer, data_dict)

    return data_dict



def general_model(model, data_dict, random_state=None):
    """ 
        Accepts regressors functions to fit and score.
        
        Parameters:        model: any regressor
                       data_dict: dictionary containing data
                    random_state: int or RandomState
                                  Pseudo-random number generator state used for random sampling
        
        Returns:                  data dictionary with fitted model and train score and test score
        
    """
    
    data_dict = validate(data_dict, random_state=random_state)
    model.fit(data_dict['X_train'], data_dict['y_train'])
    
    train_score = model.score(data_dict['X_train'], data_dict['y_train'])
    test_score = model.score(data_dict['X_test'], data_dict['y_test'])
    
    data_dict = general_process(model, data_dict, train_score=train_score, test_score=test_score)

    
    return data_dict 


def features_selected(data_dict, labels, coef=False):
    """ 
        Takes the labels zips them together to return a pretty labeled DataFrame.
        
        Parameters:   data_dict: dictionary containing data
                         labels: labels 
                    
        Returns:                 pretty labeled DataFrame
        
    """
        
    if len(labels) < 2:
        df = {'{}'.format(labels[0]):[], 'coef':[]}
        for m,n in zip(data_dict[labels[0]], coef):
            if n:
                df[labels[0]].append(m)
                df['coef'].append(n)
                
    else:
        df = {'{}'.format(labels[0]):[], '{}'.format(labels[1]):[]}
        if labels[0] == 'models':  
            for m,n in zip(data_dict[labels[0]][1:], data_dict[labels[1]]):
                if n:
                    df[labels[0]].append(m)
                    df[labels[1]].append(n)
    
        else: 
            for m,n in zip(data_dict[labels[0]], data_dict[labels[1]]):
                if n:
                    df[labels[0]].append(m)
                    df[labels[1]].append(n)

    return pd.DataFrame(df)


def roc_curve_plt(data_dict, best_model):
    """ 
        Builds a ROC curve
        
        Parameters:   data_dict: dictionary containing data
                     best_model: best_model
                    
        Returns:                 pretty ROC curve
        
    """
    y = data_dict['y_test']
    predicted = best_model.predict(data_dict['X_test'])
    fpr, tpr, thresholds = roc_curve(y,predicted)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('(1-Specificity)')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    return plt.show()




