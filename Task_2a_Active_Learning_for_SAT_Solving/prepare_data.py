"""Prepare SAT-solving data

Script that prepares the dataset for the practical course by:

- downloading databases with meta data and instance features (base, gate) from GBD
- downloading runtime data from SAT-Competition website
- merging databases
- filtering the dataset (2022 Anniversary Track, satisfiablity known, no NAs in instance features)
"""


import pathlib
import urllib.request
import zipfile

import gbd_core.api
import pandas as pd


DATA_DIR = pathlib.Path('data/')
DATABASE_NAMES = ['base', 'gate', 'meta']  # in GBD data repo
FEATURE_DATA_URL = 'https://git.scc.kit.edu/fv2117/gbd-data/-/raw/master/gbdnew/'
RUNTIME_DATA_CATEGORICAL_COLUMNS = ['benchmark', 'claimed-result', 'verified-result']
RUNTIME_DATA_INFILE = 'anni-seq.csv'  # in *.zip file downloaded from SAT-Competition website
RUNTIME_DATA_OUTFILE = 'runtimes.csv'  # how we will store the runtimes data
RUNTIME_DATA_TEMPFILE = 'runtimes_temp.zip'  # will be deleted after running the script
RUNTIME_DATA_URL = 'https://satcompetition.github.io/2022/downloads/sc2022-detailed-results.zip'

if __name__ == '__main__':
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download and save instance-feature data from GBD repo with GBD API:
    for db_name in DATABASE_NAMES:
        urllib.request.urlretrieve(url=f'{FEATURE_DATA_URL}{db_name}.db',
                                   filename=DATA_DIR / f'{db_name}.db')
        with gbd_core.api.GBD(dbs=[str(DATA_DIR / f'{db_name}.db')]) as api:
            features = api.get_features()
            features.remove('hash')  # will be added to result anyway, so avoid duplicates
            database = pd.DataFrame(api.query(resolve=features), columns=['hash'] + features)
            database.to_csv(DATA_DIR / f'{db_name}.csv', index=False)

    # Download and save runtime data from SAT-Competition website:
    urllib.request.urlretrieve(url=RUNTIME_DATA_URL, filename=RUNTIME_DATA_TEMPFILE)
    with zipfile.ZipFile(file=RUNTIME_DATA_TEMPFILE, mode='r') as zip_file:
        zip_file.extract(member=RUNTIME_DATA_INFILE, path=DATA_DIR)  # zip also contains other files
    pathlib.Path(RUNTIME_DATA_TEMPFILE).unlink()  # delete
    pathlib.Path(DATA_DIR / RUNTIME_DATA_INFILE).rename(DATA_DIR / RUNTIME_DATA_OUTFILE)
    DATABASE_NAMES.append(RUNTIME_DATA_OUTFILE.replace('.csv', ''))

    # Merge database files:
    dataset = pd.read_csv(DATA_DIR / 'meta.csv')
    dataset.rename(columns=lambda x: f'meta.{x}' if x not in ('hash', 'result') else x, inplace=True)
    numeric_cols = []
    for db_name in DATABASE_NAMES:
        if db_name != 'meta':
            database = pd.read_csv(DATA_DIR / (db_name + '.csv'))
            database.drop(columns=RUNTIME_DATA_CATEGORICAL_COLUMNS, inplace=True, errors='ignore')
            database.rename(columns=lambda x: f'{db_name}.{x}' if x != 'hash' else x, inplace=True)
            numeric_cols.extend([x for x in database.columns if x != 'hash'])
            dataset = dataset.merge(database, on='hash', how='left', copy=False)

    # Filter instances:
    dataset = dataset[dataset['meta.track'].fillna('').str.contains('anni_2022')]
    dataset = dataset[dataset['result'] != 'unknown']
    dataset[numeric_cols] = dataset[numeric_cols].transform(pd.to_numeric, errors='coerce')
    dataset = dataset[dataset[numeric_cols].notna().all(axis='columns')]  # no missing values
    dataset.drop(columns=[x for x in dataset.columns if 'meta.' in x], inplace=True)
    assert dataset['hash'].nunique() == len(dataset)
    dataset.to_csv(DATA_DIR / 'dataset.csv', index=False)
