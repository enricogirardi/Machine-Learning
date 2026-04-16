from classes import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_blobs
# from xgboost import XGBClassifier  # pip install xgboost
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import RidgeClassifier
# from mlxtend.classifier import StackingClassifier  # pip install mlxtend
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.neural_network import MLPClassifier


# -----------------------------------------------------
# CROSS VALIDATION K-FOLD
# -----------------------------------------------------

def cross_validate_modello(model, X_train, y_train, title):
    """
    Cross Validate
    Una volta scelto un modello con i suoi iperparametri,
    divide il dataset in diversi FOLD (CV_FOLD) e fa un test_csv sul modello finale

    :param model:
    :param X_train: X_train matrix
    :param y_train: y_train array
    :param title: titolo
    :return: df

    Esempio:
    model = DecisionTreeClassifier(class_weight='balanced')
    cross_validate_modello(model, X_train, y_train, 'DT')
    """

    print(f'\n############# Cross Validate -  {title} #############\n')

    # cross validation
    cv = cross_validate(model, X_train, y_train, cv=CV_FOLD_NUMBER, scoring=['accuracy', 'precision', 'recall', 'f1'],
                        return_train_score=True)
    df = pd.DataFrame(cv)
    print(df)

    # REPORT TO CSV
    report_file = os.path.join(DIRECTORY_REPORT, title + '_cross_validate.csv')
    pd.DataFrame(df).to_csv(str(report_file))

    # plot bar delle metriche
    metriche = ['train_accuracy', 'train_recall', 'train_precision', 'train_f1', 'test_accuracy', 'test_recall',
                'test_precision', 'test_f1']
    df[metriche].plot.bar()

    # salva .png in cartella /img
    path = os.path.join(DIRECTORY_IMG, title + '_cross_validate.png')
    plt.savefig(path, facecolor='y', bbox_inches="tight", pad_inches=0.3, transparent=True)

    plt.show()

    return df


# -----------------------------------------------------
# GET PARAMS DEL MODELLO
# -----------------------------------------------------

def parametri_modello(model):
    """
    Stampa i  parametri di un modello
    :param model: modello da studiare, di default il DT
    """
    gp = model.get_params()
    print(f'\n Parametri per il modello: {model.__class__.__name__}')
    print(gp)


# -----------------------------------------------------
# GRID SEARCH
# -----------------------------------------------------


def grid_search_modello(model, iperparametri, X_train, y_train):
    """
    Grid Search per 1 modello
    Cerca il modello migliore per un modello fra tutte le varianti degli iperparametri forniti
    :param model: mdello  - classe
    :param iperparametri: iperparametri del modello scelto
    :param X_train: X matrix
    :param y_train: y array
    :return:

    Esempio:
    modello = DecisionTreeClassifier()
    iperparametri = {'criterion': ['gini', 'entropy']}
    grid_search_modello(modello, iperparametri, X_train, y_train)
    """
    grid = GridSearchCV(estimator=model, param_grid=iperparametri,
                        scoring='accuracy', cv=CV_FOLD_NUMBER)
    grid.fit(X_train, y_train)
    # print(f'Best score: {grid.best_score_}')
    # print(f'Best params: {grid.best_params_}')
    # print(f'Best estimator: {grid.best_estimator_}')
    return (grid)
    # return(grid.best_estimator_)


# -----------------------------------------------------
# STACKING ENSEMBLE
# -----------------------------------------------------


def stacking_ensemble(X_train, y_train):
    """
    Stacking Ensemble

    :param X_train: X matrix for TRAIN
    :param y_train: y array for TRAIN
    :return: Stacking Classifier
    """
    print(
        '\n############################# GRID SEARCH Ricerca degli iperaparametri - Stacking Ensemble  #############################')

    modelli = [
        LogisticRegression(),  # LR
        KNeighborsClassifier(),  # KNN
        SVC(),  # SVC
        DecisionTreeClassifier(),  # DT
        MLPClassifier()     # MLP
    ]

    nomi_modelli = ['LogisticRegression', 'KNeighbors', 'SVC', 'DecisionTree', 'Multilayer Perceptron classifier']

    iperparametri_modelli = [
        # LR
        {'penalty': ['l1', 'l2', 'elasticnet'],
         'C': [1e-5, 5e-5, 1e-4, 5e-4, 1],
         'solver': ['saga'],
         'class_weight': ['balanced']
         },
        # KNN
        {'weights': ['uniform', 'distance'],
         'n_neighbors': list(range(1, 12, 2)),
         'algorithm': ['auto', 'ball_tree', 'brute', 'kd_tree', 'auto']
         },
        # SVC
        {'C': [1e-2, 1, 1e1, 1e2], 'gamma': [0.001, 0.0001],
         # 'kernel': ['linear', 'rbf', 'sigmoid'],
         'class_weight': ['dict', 'balanced']
         },
        # DT
        {'criterion': ['gini', 'entropy', 'log_loss'],
         'max_features': ['sqrt', 'log2'],
         'class_weight': ['balanced']
         },
        # Multi-layer Perceptron classifier
        {'solver': ['lbfgs', 'sgd', 'adam'],
         'activation': ['tanh', 'relu']
         }
    ]

    iperparametri_selezionati = []
    estimators = []
    punteggi = []
    iperparametri_migliori = {}

    for modello, nome_modello, iperparametri_modello in zip(modelli, nomi_modelli, iperparametri_modelli):
        print(f'\n###### {nome_modello}')

        grid = grid_search_modello(modello, iperparametri_modello, X_train, y_train)

        grid.fit(X_train, y_train)
        iperparametri_selezionati.append(grid.best_params_)
        iperparametri_migliori[nome_modello] = (('Accuracy: ' + str(grid.best_score_)), grid.best_params_)
        estimators.append((nome_modello, grid))
        punteggi.append(grid.best_score_)  # append nella lista Punteggi

        # Accuracy
        print('Accuracy:  ', grid.best_score_)

    # CONFRONTO SCORE MODELLI
    models = pd.DataFrame({'Model': nomi_modelli, 'Score': punteggi})

    # STACKING CLASSIFIER
    stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    # IPERPARAMETRI
    print(f'\nIperparametri migliori:\n {iperparametri_migliori}')

    # IPERPARAMETRI TO TXT
    perform_report_path = os.path.join(DIRECTORY_REPORT, STACKING_IPERPARAMETRI)
    chosen_hparameters_report(iperparametri_migliori, perform_report_path)

    # grafici
    models.sort_values(by='Score', ascending=False)
    #cp = sns.countplot(data=models, x='Model', hue='Score', palette=['r', 'g', 'orange', 'b', 'g'])
    cp = sns.countplot(data=models, x='Model', hue='Score', palette='pastel')
    # rotate x-axis labels
    cp.set_xticklabels(cp.get_xticklabels(), rotation=25)

    # salva .png in cartella /img
    path = os.path.join(DIRECTORY_IMG, 'stacking_ensemble' + '_models.png')
    plt.savefig(path, facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)


    plt.show()

    return stacking_classifier


# -----------------------------------------------------
# CONFRONTO SEMPLICE FRA MODELLI
# -----------------------------------------------------


def confronto_fra_modelli(X_train, y_train):
    """
    Mette a confronto i modelli ma stampa solo i risultati
    :param X_train: X matrix for TRAIN
    :param y_train: y array for TRAIN
    :return: Print dei risultati
    """
    print(
        '\n############################# GRID SEARCH Ricerca degli iperaparametri - Metodo Manuale  #############################')

    modelli = [
        LogisticRegression(),  # LR
        KNeighborsClassifier(),  # KNN
        SVC(),  # SVC
        DecisionTreeClassifier(),  # DT
        MLPClassifier()  # MLP
    ]

    nomi_modelli = ['LogisticRegression', 'KNeighbors', 'SVC', 'DecisionTree', 'Multilayer Perceptron classifier']

    iperparametri_modelli = [
        # LR
        {'penalty': ['l1', 'l2', 'elasticnet'],
         'C': [1e-5, 5e-5, 1e-4, 5e-4, 1],
         'solver': ['saga'],
         'class_weight': ['balanced']
         },
        # KNN
        {'weights': ['uniform', 'distance'],
         'n_neighbors': list(range(1, 12, 2)),
         'algorithm': ['auto', 'ball_tree', 'brute', 'kd_tree', 'auto']
         },
        # SVC
        {'C': [1e-2, 1, 1e1, 1e2], 'gamma': [0.001, 0.0001],
         # 'kernel': ['linear', 'rbf', 'sigmoid'],
         'class_weight': ['dict', 'balanced']
         },
        # DT
        {'criterion': ['gini', 'entropy', 'log_loss'],
         'max_features': ['sqrt', 'log2'],
         'class_weight': ['balanced']
         },
        # Multi-layer Perceptron classifier
        {'solver': ['lbfgs', 'sgd', 'adam'],
         'activation': ['tanh', 'relu']
         }
    ]

    iperparametri_selezionati = []
    estimators = []
    punteggi = []
    iperparametri_migliori = {}

    for modello, nome_modello, iperparametri_modello in zip(modelli, nomi_modelli, iperparametri_modelli):
        print(f'\n###### {nome_modello}')

        grid = grid_search_modello(modello, iperparametri_modello, X_train, y_train)

        grid.fit(X_train, y_train)
        iperparametri_selezionati.append(grid.best_params_)
        iperparametri_migliori[nome_modello] = (('Accuracy: ' + str(grid.best_score_)), grid.best_params_)
        estimators.append((nome_modello, grid))
        punteggi.append(grid.best_score_)  #  append nella lista Punteggi

        # Accuracy
        print('Accuracy:  ', grid.best_score_)

    # CONFRONTO SCORE MODELLI
    models = pd.DataFrame({'Model': nomi_modelli, 'Score': punteggi})

    # IPERPARAMETRI
    print(f'\nIperparametri migliori:\n {iperparametri_migliori}')

    #IPERPARAMETRI TO TXT
    perform_report_path = os.path.join(DIRECTORY_REPORT, FINAL_MODEL_IPERPARAMETRI)
    chosen_hparameters_report(iperparametri_migliori, perform_report_path)

    # grafici
    models.sort_values(by='Score', ascending=False)
    # cp = sns.countplot(data=models, x='Model', hue='Score', palette=['r', 'g', 'orange', 'b', '#9FE2BF'])
    cp = sns.countplot(data=models, x='Model', hue='Score', palette='pastel')
    # rotate x-axis labels
    cp.set_xticklabels(cp.get_xticklabels(), rotation=25)

    # salva .png in cartella /img
    path = os.path.join(DIRECTORY_IMG, 'modelli_singoli' + '_models.png')
    plt.savefig(path, facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)


    plt.show()


def risultato_modello_test_finale(final_model, X_train, y_train, X_test, y_test, title):
    """

    :param final_model: final model results
    :param X_test: X matrix for TEST
    :param y_test: y array for TEST

    """
    print('\n ...  Punteggi ... \n')

    #FIT DEL TRAIN E PREDIZIONI SUL TEST
    final_model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='macro')
    recall = recall_score(y_test, y_test_pred, average='macro')
    f1 = f1_score(y_test, y_test_pred, average='macro')
    class_report = classification_report(y_test, y_test_pred)

    # Print the evaluation metrics
    print(f"Training Accuracy  : {train_accuracy:.5f}\n")
    print(f"Test Accuracy : {accuracy:.5f}")
    print(f"Test Precision  : {precision:.5f}")
    print(f"Test Recall  : {recall:.5f}")
    print(f"Test F1 Score  : {f1:.5f}\n")
    print(f"Classification Report:\n{class_report}")

    # REPORT TO CSV
    cr = classification_report(y_test, y_test_pred, output_dict=True)
    final_report_file = os.path.join(DIRECTORY_REPORT, title + '_classification_report.csv')
    pd.DataFrame(cr).to_csv(str(final_report_file))

    #Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print('\nConfusion_matrix\n' + str(conf_matrix))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()

    # salva .png in cartella /img
    path = os.path.join(DIRECTORY_IMG, title + '_confusion_matrix.png')
    plt.savefig(path, facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)

    plt.show()
