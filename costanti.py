import os as os

ARF_CSV_FILE = 'chronic_kidney_disease_full.csv'
DATASET_FILE = 'chronic_kidney_disease_full_header.csv'
headers = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
           'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']

DROPID = 0  # se deve cancellare la colonna degli id
TARGET_DROPID = 'id'  # nome della colonna id

TEST_SIZE = 25 / 100
DIRECTORY_DATASET = 'csv'
DIRECTORY_REPORT = 'report'
DIRECTORY_IMG = 'img'
DATASET_TRAINING = 'training.csv'
DATASET_TEST = 'test_csv.csv'
CV_FOLD_NUMBER = 5

STACKING_IPERPARAMETRI = 'Stacking_Ensemble_Iperparametri.txt'
FINAL_MODEL_IPERPARAMETRI = 'Modelli_Finali_iperparametri.txt'

training_file = os.path.join(DIRECTORY_DATASET, DATASET_TRAINING)
test_file = os.path.join(DIRECTORY_DATASET, DATASET_TEST)
