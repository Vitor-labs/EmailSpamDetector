import io
import pickle
import pathlib
import zipfile
import requests
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

MAX_NUM_WORDS = 280
MAX_SEQ_LENGTH = 300

BASE_DIR = pathlib.Path().resolve().parent
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True, parents=True)
SPAM_DATASET_PATH = EXPORT_DIR / "spam-dataset.csv"

METADATA_EXPORT_PATH = EXPORT_DIR / 'spam-metadata.pkl'
TOKENIZER_EXPORT_PATH = EXPORT_DIR / 'spam-tokenizer.json'


ytb_spam_df_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip'
sms_spam_df_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'

def get_df_data_by_url(url: str) -> pd.DataFrame:
    """Extracts data from csv files in .zip by url

    Args:
        url (str): url from .zip repository

    Raises:
        ValueError: No csv file on unziped repository

    Returns:
        pd.DataFrame: Data Frame Loaded
    """
    response = requests.get(url, timeout=5)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    csv_files = [file_name for file_name in zip_file.namelist() if file_name.endswith(".csv")]

    if len(csv_files) == 1:
        dataframe = pd.read_csv(zip_file.open(csv_files[0]))

    elif len(csv_files) == 0:
        data_file = [file_name for file_name in zip_file.namelist() if '.' not in file_name]

        if len(data_file) > 1:
            dataframe = pd.read_csv(zip_file.open(data_file[0]), sep='\t', header=None)

        else:
            raise ValueError("No CSV file found and no data file without extension found.")

    else:
        # Multiple CSV files, concatenate them into a single DataFrame
        csv_data = [
            zip_file.open(file_name).read().decode("latin-1")
            for file_name in csv_files
        ]

        dataframe = pd.concat([pd.read_csv(io.StringIO(data)) for data in csv_data])
        del csv_data

    return dataframe

ytb_df = get_df_data_by_url(ytb_spam_df_url)
sms_df = get_df_data_by_url(sms_spam_df_url)

mapping = {'spam': 1, 'ham': 0}

sms_df.columns = ['class', 'body']
sms_df['source'] = 'sms'
sms_df['class'] = sms_df['class'].replace(mapping)

ytb_df = ytb_df.drop(columns=['COMMENT_ID', 'AUTHOR', 'DATE', 'Unnamed: 0'], axis=1)
ytb_df = ytb_df.rename(columns={'CLASS':'class', 'CONTENT': 'body'})
ytb_df['source'] = 'youtube'

spam_df = pd.concat([sms_df, ytb_df])
spam_df.head()

classes = spam_df['class'].tolist()
texts = spam_df['body'].tolist()

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

X = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)
y = to_categorical(np.asarray(classes))

X_train, X_test, y_train, y_t2 = train_test_split(X, y, test_size=0.5, random_state=42)
X_test, X_valid, y_train, y_test = train_test_split(X_test, y_t2, test_size=0.33, random_state=42)

spam_df.to_csv(SPAM_DATASET_PATH)

training_data = {
    "X_train": X_train, 
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "max_words": MAX_NUM_WORDS,
    "max_seq_length": MAX_SEQ_LENGTH,
    "label_legend": mapping,
}

tokenizer_json = tokenizer.to_json()
TOKENIZER_EXPORT_PATH.write_text(tokenizer_json)

with open(METADATA_EXPORT_PATH, 'wb') as f:
    pickle.dump(training_data, f)
