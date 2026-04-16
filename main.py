from open import *
from eda import *
from modelli import *

if __name__ == '__main__':
    # DATASET .arff to .csv + fix format problems
    converti_csv()
    # SPLIT CSV to 2 CSV
    initial_split(DROPID, TARGET_DROPID)

    print('\n#############################\nTRAINING DATA\n#############################')
    # READ CSV -  TRAINING
    df_train = pd.read_csv(training_file)

    # INITIAL ANALYSIS
    initial_dataset_analysis(df_train)

    # EDA - CSV_TRAIN
    X_train, y_train = eda(df_train, 1)

    # Salvataggio CSV per test_csv di X_train
    pd.DataFrame(X_train).to_csv('test_csv/X_train.csv', index=False)

    # SCALER FIT TRANSFORM X_TRAIN
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # STACKING ENSEMBLE
    final_model = stacking_ensemble(X_train, y_train)

    print('\n#############################\nTEST DATA\n#############################')
    # READ CSV - TEST
    df_test = pd.read_csv(test_file)

    # EDA - CSV_TEST
    X_test, y_test = eda(df_test, 0)

    # Salvataggio CSV per test_csv di X_test
    pd.DataFrame(X_test).to_csv('test_csv/X_test.csv', index=False)

    # SCALER TRANSFORM X_TEST
    X_test = scaler.transform(X_test)

    # STACKING ENSEMBLE RISULTATO FINALE
    cross_validate_modello(final_model, X_train, y_train, 'Stacking_Ensemble')
    risultato_modello_test_finale(final_model, X_train, y_train, X_test, y_test, 'Stacking_Ensemble')



    print('\n############################# MODELLI SELEZIONATI SINGOLARMENTE  #############################')
    print(confronto_fra_modelli(X_train, y_train))


    # MODELLO  1 - LogisticRegression
    print(f'\n #########   QUARTO MODELLO SELEZIONATO: LogisticRegression(C=0.0005, class_weight=\'balanced\', solver=\'saga\')')
    final_model = LogisticRegression(C=0.0005, class_weight='balanced', solver='saga')
    cross_validate_modello(final_model, X_train, y_train, 'LogisticRegression')
    risultato_modello_test_finale(final_model, X_train, y_train, X_test, y_test, 'LogisticRegression')

    # MODELLO  2 - KNN
    print(f'\n #########   SECONDO MODELLO SELEZIONATO: KNeighborsClassifier(n_neighbors=1)')
    final_model = KNeighborsClassifier(n_neighbors=1)
    cross_validate_modello(final_model, X_train, y_train, 'KNN')
    risultato_modello_test_finale(final_model, X_train, y_train, X_test, y_test, 'KNN')

    # MODELLO  3 - SVC
    print(f'\n #########   SECONDO MODELLO SELEZIONATO: SVC(C=100.0, class_weight=\'balanced\', gamma=0.001)')
    final_model = SVC(C=100.0, class_weight='balanced', gamma=0.001, kernel='rbf')
    cross_validate_modello(final_model, X_train, y_train, 'SVC')
    risultato_modello_test_finale(final_model, X_train, y_train, X_test, y_test, 'SVC')

    # MODELLO 4 - DecisionTreeClassifier
    print(f'\n #########  PRIMO MODELLO SELEZIONATO: DecisionTreeClassifier(class_weight=\'balanced\', criterion=\'entropy\',max_features=\'sqrt\')')
    final_model = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_features='sqrt')
    cross_validate_modello(final_model, X_train, y_train, 'DecisionTree')
    risultato_modello_test_finale(final_model, X_train, y_train, X_test, y_test, 'DecisionTree')


    # MODELLO  5 - MLP Classifier
    print(f'\n #########   TERZO MODELLO SELEZIONATO: MLPClassifier(activation=\'tanh\', solver=\'lbfgs\')')
    final_model = MLPClassifier(activation='tanh', solver='lbfgs')
    cross_validate_modello(final_model, X_train, y_train, 'MLPClassifier')
    risultato_modello_test_finale(final_model, X_train, y_train, X_test, y_test, 'MLPClassifier')


