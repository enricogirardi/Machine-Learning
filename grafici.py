from classes import *


# -----------------------------------------------------
## GRAFICI SPECIFICI PER IL PROGETTO
# -----------------------------------------------------

def grafici_1(df, target):
    """
    Grafici inserito nel punto 1
    :param df: dataframe
    :param target: taget column
    """
    numeric_columns, categoric_columns = col_numeric_categoric(df)

    # GRAFICI
    labels= ['Paziente affetto', 'Paziente non affetto']
    explode = (0.1, 0)
    title = 'Malattia Renale Cronica '
    pie(df, target, labels, explode, title)  # Target Feature

    countplot(df, categoric_columns, 'Feature categoriche')  # Feature categoriche
    distplot(df, numeric_columns, 'Feature numeriche')  # Feature numeriche



def grafici_2(df):
    """
    Grafici inserito nel punto 2
    :param df: dataframe
    """
    heatmap(df)


def grafici_3(df):
    """
    Grafici inserito nel punto 3
    :param df: dataframe
    """
    cols = df.columns.values
    missing_values = (df[cols].isna().sum() / df.shape[0] * 100).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    missing_values.plot(kind='bar')
    plt.title('Valori Mancanti')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # salva .png in cartella /img
    path = os.path.join(DIRECTORY_IMG, 'Missing-Values' + '.png')
    plt.savefig(path,  facecolor='lightblue', bbox_inches="tight", pad_inches=0.3, transparent=True)


    plt.show()


    ## NON USATI
    # col_names = [' ', ' ', ' ']
    # crosstables(df, col_names, target)
    # col_names2 = [' ', ' ']
    # boxplot(df, col_names2, target)
    # pairplot(df, ' ', target)
    # histplot(df, target)
    # kdeplot(df, col_names2, target)
    # scatter(df, ' ', ' ', target)


# -----------------------------------------------------
## GRAFICI - DEFINIZIONE CLASSI
# -----------------------------------------------------

def crosstables(df, col_names, target):
    """

    :param df: dataframe
    :param col_names: columns name
    :param target: target feature
    """
    print(f'\n...Crosstable for {col_names}....')
    for col in col_names:
        content_tbl = pd.crosstab(df[col], df[target])
        plt.figure(figsize=(10, 6))
        plt.title(col)
        sns.heatmap(content_tbl, annot=True, cmap='coolwarm', fmt='d')
        plt.show()


def boxplot(df, col_names, target):
    """

    :param df: dataframe
    :param col_names: columns name
    :param target: target feature
    """
    print(f'\n...Boxplot for {col_names} & {target}...')
    for col in col_names:
        plt.figure(figsize=(10, 8))
        plt.title(f'boxplot for {col} & {target}')
        sns.boxplot(data=df, x=col, y=target)
        plt.show()


def kdeplot(df, col_names, target):
    """

    :param df: dataframe
    :param col_names: columns name
    :param target: target feature
    """
    print("\n...Kdeplot...")
    # solo per valori numerici
    for col in col_names:
        fg2 = sns.FacetGrid(df, hue=target)
        fg2.map(sns.kdeplot, col)
        fg2.add_legend()
        plt.show()


def scatter(df, col1, col2, target):
    """

    :param df: dataframe
    :param col1: column one
    :param col2: column two
    :param target: target feature
    """
    print("\n...Scatterplot... ")
    fg = sns.FacetGrid(df, hue=target)
    fg.map(plt.scatter, col1, col2)
    fg.add_legend()
    plt.show()



def pairplot(df,to_drop, target):
    """

    :param df: dataframe
    :param to_drop: columns to drop
    :param target: target feature
    """
    print(f'\n...Pairlot for {target}...')
    sns.pairplot(df.drop(to_drop, axis=1), hue=target, height=3)
    plt.show()


def histplot(df, target):
    """

    :param df: dataframe
    :param target: target feature
    """
    print(f'\n...Pairpplot for {target}...')
    sns.histplot(data=df, y=target)
    plt.show()


def pie(df, target, labels, explode, title):
    """

    :param df: dataframe
    :param target: target feature
    """
    print("\n ...Pie Chart... ")
    df[target].value_counts().plot(kind="pie",  labels=labels, explode=explode, startangle = 90)
    plt.title(title)

    # salva .png in cartella /img
    path = os.path.join(DIRECTORY_IMG, title + '.png')
    plt.savefig(path,  facecolor='lightblue', bbox_inches="tight", pad_inches=0.3, transparent=True)


    plt.show()


def countplot(df, col_names, title):
    """

    :param df: datafrme
    :param col_names:  coloumns name
    """
    plt.figure(figsize=(20, 15))
    plotnumber = 1

    for column in col_names:
        if plotnumber <= 11:
            ax = plt.subplot(3, 4, plotnumber)
            sns.countplot(df[column], palette='rocket')
            plt.xlabel(column)

        plotnumber += 1

    plt.tight_layout()

    # salva .png in cartella /img
    path = os.path.join(DIRECTORY_IMG, title + '.png')
    plt.savefig(path,  facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)


    plt.show()


def distplot(df, col_names, title):
    """

    :param df: datafrme
    :param col_names:  coloumns name
    """
    # features distribution
    plt.figure(figsize=(25, 20))
    plotnumber = 1

    for column in col_names:
        if plotnumber <= 15:
            ax = plt.subplot(3, 5, plotnumber)
            sns.distplot(df[column])
            plt.xlabel(column)

        plotnumber += 1

    plt.tight_layout()

    # salva .png in cartella /img
    path = os.path.join(DIRECTORY_IMG, title + '.png')
    plt.savefig(path,  facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)

    plt.show()



def heatmap(df):
    """

    :param df: datafrme
    """
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr(), annot=True, linewidths=2)

    # salva .png in cartella /img
    path = os.path.join(DIRECTORY_IMG, 'heatmap' + '.png')
    plt.savefig(path,  facecolor='lightblue', bbox_inches="tight", pad_inches=0.3, transparent=True)


    plt.show()