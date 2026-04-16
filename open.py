from costanti import *
import os as os
import pandas as pd

# elenca tuti i files arff  presenti nella cartella
files = [arff for arff in os.listdir(DIRECTORY_DATASET) if arff.endswith(".arff")]

def toCsv(content):
    """
    Prepara il contenuto per il formato CSV
    :param content: arff
    :return: contenuto preparato per il  csv
    """
    data = False
    header = ""
    newContent = []
    for line in content:
        line = line.replace('\t', '')  # NB: RIMUOVE I DATI TABULARI CHE FANNO SBALLARE ALCUNE RIGHE
        if not data:
            if "@attribute" in line:
                attri = line.split()
                columnName = attri[attri.index("@attribute")+1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            newContent.append(line)
    return newContent


def converti_csv():
    """
    Converte il contenuto in CSV
    """


    # Main loop for reading and writing files
    for el,file_t in enumerate(files):
        f_ele = os.path.join(DIRECTORY_DATASET, file_t)
        print(f_ele)
        # with open(DIRECTORY_DATASET+file , "r") as inFile:
        with open(f_ele, "r") as inFile:
        # with open('csv/chronic_kidney_disease_full.arff', "r") as inFile:
            content = inFile.readlines()
            name,ext = os.path.splitext(inFile.name)
            new = toCsv(content)
            with open(name+".csv", "w") as outFile:
                outFile.writelines(new)

    # CAMBIA HEADER
    arf_csv = os.path.join(DIRECTORY_DATASET, ARF_CSV_FILE)
    # print(arf_csv)
    # legge il CSV appena scritto
    df_csv = pd.read_csv(str(arf_csv), on_bad_lines='skip')


    # cambia gli headers del CSV e lo salva in un nuovo csv
    arf_csv_header = os.path.join(DIRECTORY_DATASET, DATASET_FILE)
    df_csv.to_csv(str(arf_csv_header), header=headers , index= False ) # index= False: omette la colonna degli ID quando salva

