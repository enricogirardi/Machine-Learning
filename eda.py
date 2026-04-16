from classes import *
from grafici import *

"""
DATAFRAME - SIGNIFICATO DELLE COLONNE DEL DATAFRAME
https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

age		-	age	
bp		-	blood pressure
sg		-	specific gravity
al		-   albumin
su		-	sugar
rbc		-	red blood cells
pc		-	pus cell
pcc		-	pus cell clumps
ba		-	bacteria
bgr		-	blood glucose random
bu		-	blood urea
sc		-	serum creatinine
sod		-	sodium
pot		-	potassium
hemo	-	hemoglobin
pcv		-	packed cell volume
wc		-	white blood cell count
rc		-	red blood cell count
htn		-	hypertension
dm		-	diabetes mellitus
cad		-	coronary artery disease
appet	-	appetite
pe		-	pedal edema
ane		-	anemia
class	-	class	


age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane,class


This dataset can be used to predict the chronic kidney disease and it can be collected
from the hospital nearly 2 months of period.
"""


# target feature
target = 'class'



def eda(df, graf):
    """
    Exploratory Data Analysis
    :param df: daframe
    :param  graf: if 1 = esegue il plot dei grafici
    """
    print('\n############################# EDA Exploratory Data Analysis  #############################')

    # FORZATURA DEL DTYPE MUMERICO
    # Alcuni valori annoverati fra le Categorical fetures sono numeri ma non codificati tali,li convertiamo in numero
    cat_to_num = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',]
    for el in cat_to_num:
        df[el] = pd.to_numeric(df[el], errors='coerce') # coerce  invalid parsing will be set as NaN


    # FEATURE NUMERICHE E FEATURE CATEGORICHE
    numeric_columns, categoric_columns = col_numeric_categoric(df)

    # RIMOZIONE ERRORI
    df['dm'] = df['dm'].replace(to_replace={'\tyes': 'yes', ' yes': 'yes', '\tno': 'no'})
    df['cad'] = df['cad'].replace(to_replace={'\tno': 'no'})
    df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd'})

    # GRAFICI INIZIALI
    if graf == 1:
        grafici_1(df, target)
        grafici_3(df)


    # CONVERTE I VALORI CATEGORICI IN NUMERI
    # TOLTO PERCHE' FUNZIONA MEGLIO LA MAPPATURA MANUALE (AUMENTA LA PRECISION ED ACCURACY )
    # cat_to_label =  ['class','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
    # lab_enc = LabelEncoder()
    # for col in cat_to_label:
    #     df[col] = lab_enc.fit_transform(df[col])


    # MAPPATURA DEI VALORI  (METODO EQUIVALENTE CON VALORI CUSTOM)
    # df['class'] = df['class'].map({'ckd': 0, 'notckd': 1})
    # df['rbc'] = df['rbc'].map({'abnormal': 0, 'normal': 1, '?': 2})
    # df['pc'] = df['pc'].map({'abnormal': 0, 'normal': 1, '?': 2})
    # df['pcc'] = df['pcc'].map({'notpresent': 0, 'present': 1, '?': 2})
    # df['ba'] = df['ba'].map({'notpresent': 0, 'present': 1, '?': 2})
    # df['htn'] = df['htn'].map({'no': 0, 'yes': 1, '?': 2})
    # df['dm'] = df['dm'].map({'no': 0, 'yes': 1, '?': 2})
    # df['cad'] = df['cad'].map({'no': 0, 'yes': 1, '?': 2})
    # df['appet'] = df['appet'].map({'poor': 0, 'good': 1, '?': 2})
    # df['pe'] = df['pe'].map({'no': 0, 'yes': 1, '?': 2})
    # df['ane'] = df['ane'].map({'no': 0, 'yes': 1, '?': 2})

    # MAPPATURA IN CUI SOSTITUISCO ? CON NaN
    df['class'] = df['class'].map({'ckd': 0, 'notckd': 1})
    df['rbc'] = df['rbc'].map({'abnormal': 0, 'normal': 1, '?': pd.NA})
    df['pc'] = df['pc'].map({'abnormal': 0, 'normal': 1, '?': pd.NA})
    df['pcc'] = df['pcc'].map({'notpresent': 0, 'present': 1, '?': pd.NA})
    df['ba'] = df['ba'].map({'notpresent': 0, 'present': 1, '?': pd.NA})
    df['htn'] = df['htn'].map({'no': 0, 'yes': 1, '?': pd.NA})
    df['dm'] = df['dm'].map({'no': 0, 'yes': 1, '?': pd.NA})
    df['cad'] = df['cad'].map({'no': 0, 'yes': 1, '?': pd.NA})
    df['appet'] = df['appet'].map({'poor': 0, 'good': 1, '?': pd.NA})
    df['pe'] = df['pe'].map({'no': 0, 'yes': 1, '?': pd.NA})
    df['ane'] = df['ane'].map({'no': 0, 'yes': 1, '?': pd.NA})


    # MISSING VALUES
    print(f'Valori mancanti PRIMA: {df.isna().sum().sum()} ')
    # ELENCO DEI VALORI MANCANTI
    print(f'{df.isna().sum().sort_values(ascending = False)}')

    # SOSTITUZIONE DEI VALORI NUMERICI
    for column in numeric_columns:
        #cambia_nan_con_valori_random(df, column)  # TOLTO PERCHE' TROPPO IMPEVEDIBILE
        cambia_Nan_con_moda(df, column)
        # cambia_Nan_con_media(df, column)

    # SOSTITUZIONE DEI VALORI CATEGORICI
    for column in categoric_columns:
        cambia_Nan_con_moda(df, column)

    print('\nRE-CHECK NUMERICAL AND CATEGORIGAL COLUMNS AFTER TRANSFORMATION')
    col_numeric_categoric(df)


    # MANIPULATION DATA
    df = manipulation_data(df)

    # HEATMAP
    if graf == 1:
        grafici_2(df)

    ##  FEATURE SELECTION
    # definition of X,y
    # RIMUOVO LE DUE COLONNE CON PIU' VALORI MANCANTI
    X = df.drop(['rbc', 'rc','wc','pot','sod', 'class'], axis=1)
    # X = df_X(df, ['', '', '', ''])  # se devo  tenere solo specifiche colonne
    # X = df.drop(target, axis=1)  # tutte le colonne eccetto la target
    y = df[target]


    # TEST
    # X.to_csv('EDA_TEST_X.csv', index=False)
    # y.to_csv('EDA_TEST_y.csv', index=False)
    return (X, y)

