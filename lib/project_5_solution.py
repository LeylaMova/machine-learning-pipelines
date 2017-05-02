import psycopg2
from numpy import array
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split

def connect_to_postgres():

    connection = psycopg2.connect(dbname='dsi',
                                  user='dsi_student', 
                                  host='joshuacook.me', 
                                  port=5432, 
                                  password='correct horse battery staple')

    return connection, connection.cursor()

def load_data_from_database(local=False):
    '''
    Loads madelon data set as a pandas DataFrame.
    If local=True, loads from a local csv.
    '''
    if local:
        madelon_df = read_csv('../../data/madelon_full.csv')
    else:
        connection, cursor = connect_to_postgres()

        cursor.execute('SELECT * FROM madelon;')
        results = cursor.fetchall()
        results = array(results)
        
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'madelon';")
        columns = cursor.fetchall()
        columns = [column[0] for column in columns]
        
        madelon_df = DataFrame(results, columns=columns)
        
        connection.close()
        
    madelon_df.index = madelon_df['index']
    madelon_df.drop('index', inplace=True, axis=1)
    
    return madelon_df

def add_to_or_create_process_list(process, data_dictionary):
    if 'processes' in data_dictionary.keys():
        data_dictionary['processes'].append(process)
    else:
        data_dictionary['processes'] = [process]
        
    return data_dictionary

def make_data_dict(dataframe, random_state=None):
    
    y = dataframe['label']
    X = dataframe.drop('label', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        random_state=random_state)
    data_dictionary = {
        'X' : X,
        'y' : y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    return data_dictionary

def general_transformer(transformer, data_dictionary):
    
    transformer.fit(data_dictionary['X_train'], data_dictionary['y_train'])
    
    data_dictionary['X_train'] = transformer.transform(data_dictionary['X_train'])
    data_dictionary['X_test'] = transformer.transform(data_dictionary['X_test'])
    
    add_to_or_create_process_list(transformer, data_dictionary)
    
    return data_dictionary

def general_model(model, data_dictionary):
    
    model.fit(data_dictionary['X_train'], data_dictionary['y_train'])
    
    data_dictionary['train_score'] = model.score(data_dictionary['X_train'], 
                                                 data_dictionary['y_train'])
    data_dictionary['test_score'] = model.score(data_dictionary['X_test'], 
                                                data_dictionary['y_test'])
    
    add_to_or_create_process_list(model, data_dictionary)
    
    return data_dictionary