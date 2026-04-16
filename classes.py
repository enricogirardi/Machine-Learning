from costanti import *
import os as os

# -----------------------------------------------------
## VERSION CLASSES 6.3
# -----------------------------------------------------

import warnings
warnings.simplefilter('ignore')
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("pastel")
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler  # DA USARE SE CI SONO SOLO 2 VALORI UNICI
from sklearn.preprocessing import OneHotEncoder  # DA USARE SE CI SONO MULTIPLE CATEGORIE TESTUALI
from sklearn.preprocessing import StandardScaler  # SCALA I NUMERI IN UN RANGE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# COLUMNS=  df.columns.values

# -----------------------------------------------------
## SPLIT CSV
# -----------------------------------------------------

def initial_split(DROPID,TARGET_DROPID):
    """
     Esegue lo split del CSV
    :param DROPID: if 1, drop an id target feature
    :param TARGET_DROPID: name of target feature
    """
    file = os.path.join(DIRECTORY_DATASET, DATASET_FILE)
    df = pd.read_csv(file)
    simple_split(df,DROPID,TARGET_DROPID)



# -----------------------------------------------------
## ANALISI PRELIMINARE
# -----------------------------------------------------

def col_numeric_categoric(df):
    """
     Divide fra feature numeriche e feature categoriche
    :param df: dataframe
    :return: numeric_columns, categoric_columns
    """
    #########################
    # VALORI NUMERICI E CATEGORICI
    #########################
    columns = list(df.columns)

    categoric_columns = []
    numeric_columns = []

    for i in columns:
        if is_numeric_dtype(df[i]):
            numeric_columns.append(i)
        if df[i].dtypes == 'object':
            categoric_columns.append(i)

    print(f'\n==== Numerical fetures (Tot: {len(numeric_columns)}: ', numeric_columns)
    print(f'\n==== Categorical fetures (Tot: {len(categoric_columns)}: ', categoric_columns)

    return (numeric_columns, categoric_columns)




def initial_dataset_analysis(df):
    """
     Stampa le informazioni iniziali sul dataframe
    :param df: dataframe
    :return:True
    """
    print('\n############################# INITIAL ANALYSIS #############################')

    # print("\n...head().T: ")
    # print(df.head().T)

    print("\n...head(): ")
    print(df.head(3))

    print("\n...shape: ")
    print(df.shape)

    print("\n...describe(): ")
    print(df.describe())

    print("\n...info(): ")
    print(df.info())

    numeric_columns, categoric_columns = col_numeric_categoric(df)


    # VALORI TESTUALI NELLE CATEGORICAL FEATURES
    #########################
    print(f'\n==== Valori contenuti nelle Categorical fetures:')
    for col in categoric_columns:
        print(f'{col} has {df[col].unique()} values')


    # VALORI UNICI
    #########################
    print("\n...Examining Unique Values - Conteggio dei valori unici...")
    unique_value = []
    for i in df.columns:
        x = df[i].value_counts().count()
        unique_value.append(x)
    dupl = pd.DataFrame(unique_value, index=df.columns, columns=["Total Unique Data"])
    print(f'{dupl}')




# -----------------------------------------------------
## PREPARAZIONE DATI
# -----------------------------------------------------

def cambia_Nan_con_moda(df, feature):
    """
    Sostituizione dei valori NaN con la MODA
    :param df: dataframe
    :param feature: feature to change
    """
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)


def cambia_Nan_con_media(df, feature):
    """
    Sostituizione dei valori NaN con la MEDIA
    :param df: dataframe
    :param feature: feature to change
    """
    df[feature] = df[feature].fillna(df[feature].mean())




def cambia_nan_con_valori_random(df, feature):
    """
    Sostituizione dei valori NaN con VALORI RANDON
    :param df: dataframe
    :param feature: feature to change
    """
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample




def manipulation_data(df):
    """
     Conteggiio di valori NaN e Rimozione  di valori duplicati
    :param df: dataframe
    :return:   dataframe after processing
    """

    # CONTEGGIO DEI  VALORI  NaN
    #########################
    print("\n...Check for Nan - Controlla se ci sono valori Nan... ")
    conteggio_nan = df.isnull().sum(axis=0).sum()  # (axis = 0) NaN values in every column. || FOR NaN values in every row (axis = 1)
    if conteggio_nan > 0:
        print(f'\tThere are  {conteggio_nan} NaN values  - Ci sono {conteggio_nan} NaN values.')
    #     # sostituisce con i valori di moda della colonna
    #     for column in df.columns.values:
    #         # cambia_Nan_con_moda(df, column) #todo METODO ALTERNATIVO (tolto perchè abbassa lo score)
    #         cambia_nan_con_valori_random(df, column)
    #
    #     print("\tRemoving NaN values - Rimuovo i  NaN values.")
    #     conteggio_nan = df.isnull().sum(axis=0).sum()
    #     print(f'\tThere are now {conteggio_nan} NaN values -  Adesso ci sono {conteggio_nan} NaN values.')
    else:
        print(f'\tThere are not NaN values  - Non ci sono NaN values.')


    # RIMOZIONE DUPLICATI
    #########################
    print("\n...Check for duplicates - Controlla se ci sono duplicati... ")
    conteggio = df.duplicated().sum()

    if conteggio > 0:
        print(f'\tThere are  {conteggio} duplicates  - Ci sono {conteggio} duplicati.')
        df = df.drop_duplicates()

        print("\tRemoving duplicates - Rimuovo i  duplicati.")
        conteggio = df.duplicated().sum()
        print(f'\tThere are now {conteggio} duplicates -  Adesso ci sono {conteggio} duplicati.')
    else:
        print(f'\tThere are not duplicates  - Non ci sono  duplicati.')

    return df


# -----------------------------------------------------
##  SPLIT DEL DATASET E SCALING
# -----------------------------------------------------

def simple_split(df,DROPID,TARGET_DROPID):
    """
    Esegue uno split del CSV
    :param df: dataframe
    :param DROPID: if 1, drop an id target feature
    :param TARGET_DROPID: name of target feature
    """
    # split the data into train and test_csv set

    if DROPID == 1:
        # ID DROP
        df = df.drop(TARGET_DROPID, axis=1)  # IN CASO DI COLONNA CON ID

    test, train = train_test_split(df, test_size=TEST_SIZE, random_state=0)
    # save the data
    train.to_csv(training_file, index=False)
    test.to_csv(test_file, index=False)


def split_dataframe_scaler(X, y):
    """
     Split del dataframe con scaler.fit trandform
    :param X: input X Matrix
    :param y: input y array
    :return: Splitted dataframes: X_train, X_test, y_train, y_test
    """
    print("\n ... split_dataframe_scaler ... ")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=33)
    scaler = StandardScaler(with_mean=False) # ValueError: Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test



def scaling_X(X):
    """

    :param X: matrix X
    :return: X fit + transformed
    """
    scaler = StandardScaler(with_mean=False)
    # scaler = MinMaxScaler()
    X_fit_trans = scaler.fit_transform(X)
    return X_fit_trans


def scaler_tranform_X(X):
    """

    :param X: matrix X
    :return: X transformed
    """
    scaler = StandardScaler(with_mean=False)
    X_trans = scaler.transform(X)
    return X_trans







# -----------------------------------------------------
## MANIPOLAZIONE DEI DATI
# -----------------------------------------------------

def normalize(df):
    """

    :param df: input dataframe
    :return:  normalized dataframe
    """
    nom = Normalizer(norm='l2')
    data_norm = nom.fit_transform(df)
    return data_norm


def scaler(X):
    """
    :param X: input X Matrix
    :return:  X_scaled
    """
    print("\n ... Scaler ... ")
    scaler = StandardScaler(with_mean=False)  # ValueError: Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def ColTransform(X, trasformatori):
    """
    :param X: input Matrix
    :param trasformatori: list of transformations
    :return: X matrix modified
    """
    print("\n ... ColumnTransformer ... ")
    ct = ColumnTransformer(trasformatori, remainder='passthrough') # remainder : {'drop', 'passthrough'}
    ct.fit(X)
    X = ct.transform(X)
    return X






def txt_perform_report(text, path, title):
    """
     Scrive Report TXT
    :param text:
    :param path:
    :param title:
    """
    if not os.path.exists(DIRECTORY_REPORT):
        os.mkdir(DIRECTORY_REPORT)

    with open(path, 'w+') as f:
        f.write(title)
        row = ""
        for value in text:
            row += str(value)
            row += "\n"
        f.write(row + "\n")


def chosen_hparameters_report(chosen_hparameters, path):
    """
     Scrive Report TXTneipe
    :param chosen_hparameters:  iperparametri scelti dalla cross validation
    :param path: path di  destinazio
    """
    if not os.path.exists(DIRECTORY_REPORT):
        os.mkdir(DIRECTORY_REPORT)

    with open(path, 'w') as f:
        f.write("IPERPARAMETRI MIGLIORI\n")
        row = ""
        for key, values in chosen_hparameters.items():
            row += "MODELLO: " + key + "\n"
            for value in values:
                row += str(value) + "\n"
            row += "\n"
        f.write(row + "\n")