from sklearn.datasets import fetch_kddcup99
import pandas as pd
import numpy as np

class Loader:

    def load_data(self):
        """ Carrega e retorna um dataset.
            Dados escolhidos: KDD-Cup99 (Detecção de Intrusão)
        """

        # Load data as a Pandas Dataframe
        dataset = fetch_kddcup99(shuffle=True, as_frame=True, download_if_missing=True)
        data = dataset.data.copy()
        # Remove duplicated data
        data.drop_duplicates(subset=None, keep='first', inplace=True)

        # Type conversion (needed step - data are defined as objects)
        data = data.convert_dtypes()
        data.dtypes
        # naming - from bytes to string
        str_df = data.select_dtypes([np.object])
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            data[col] = str_df[col]

        # Reducing atack classes, normal class remains (it already converts from byte to string)
        attack_types = {
            b'normal': 'normal',
            b'back': 'dos',
            b'buffer_overflow': 'u2r',
            b'ftp_write': 'r2l',
            b'guess_passwd': 'r2l',
            b'imap': 'r2l',
            b'ipsweep': 'probe',
            b'land': 'dos',
            b'loadmodule': 'u2r',
            b'multihop': 'r2l',
            b'neptune': 'dos',
            b'nmap': 'probe',
            b'perl': 'u2r',
            b'phf': 'r2l',
            b'pod': 'dos',
            b'portsweep': 'probe',
            b'rootkit': 'u2r',
            b'satan': 'probe',
            b'smurf': 'dos',
            b'spy': 'r2l',
            b'teardrop': 'dos',
            b'warezclient': 'r2l',
            b'warezmaster': 'r2l',
        }

        # Adding the class_type column to dataset
        context = dataset.target.apply(lambda sample:attack_types[sample[:-1]])
        data['class_type'] = context
        return data
