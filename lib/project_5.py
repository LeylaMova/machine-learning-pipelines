import pandas
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split


def general_process(processes, data_dict):
    """ 
        Track data dictionary and processes that has been completed step by step.
        
        Parameters: processes: functions
                    data_dict: dictionary containing data
        
        Returns:               dictionary of data plus the process that was completed
        
    """
    
    if 'process' in data_dict.keys():
        data_dict['process'].append(processes)
    else:
        data_dict['process'] = [processes]
    
    return data_dict


def validate(data_dict, random_state=None):
    """ 
        Validates the necessary data (X_train, X_test, y_train, y_test) needed to continue with processing. If any of them are missing it will create train and test sets with train_test_split using X and y.
        
        Parameters:    data_dict: dictionary containing data
                    random_state: int or RandomState
                                  Pseudo-random number generator state used for random sampling
        
        Returns:                  dictionary with necessary data 
        
    """
    
    if 'X_test' in data_dict.keys():
        pass
    else:
        data_dict = train_test_split(X,y,random_state=random_state)
    return data_dict


def load_data_from_database(user, password, url, port, database, table):
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
    df = pandas.read_sql("SELECT * FROM {}".format(table), con=engine)

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
    data_dict = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'X': X, 'y': y}

    return data_dict


def general_transformer(transformer, data_dict, random_state=None):
    """ 
        Fits training data then transforms the feature matrix.
        
        Parameters:  transformer: takes any transformer function
                       data_dict: dictionary containing data
                    random_state: int or RandomState
                                  Pseudo-random number generator state used for random sampling
        
        Returns:                  dictionary containing transformed feature matrix with target vectors
        
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
        
        Returns:                  dictionary containing the model name, train score and test score
        
    """
    
    data_dict = validate(data_dict, random_state=random_state)
    model.fit(data_dict['X_train'], data_dict['y_train'])
    
    train_score = model.score(data_dict['X_train'], data_dict['y_train'])
    test_score = model.score(data_dict['X_test'], data_dict['y_test'])
    
    data_dict = general_process(model, data_dict)

    
    return {'model': model, 'train_score': train_score, 'test_score': test_score}