########## 1. Import required libraries ##########

import os
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.naive_bayes import MultinomialNB

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

########## 2. Define text preprocessing methods ##########

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', str(text))

def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', str(text))

NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    return " ".join(
        [word for word in str(text).split() if word.lower() not in final_stop_words_list]
    )

def clean_str(string):
    string = str(string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

########## 3. Read data ##########

# choice: 'caffe', 'pytorch', 'tensorflow', 'incubator-mxnet', 'keras'
project = 'caffe'
path = f'datasets/{project}.csv'

if not os.path.exists(path):
    raise FileNotFoundError(f"Dataset not found: {path}")

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999).reset_index(drop=True)

pd_all['Title'] = pd_all['Title'].fillna('')
pd_all['Body'] = pd_all['Body'].fillna('')

pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if row['Body'].strip() else row['Title'],
    axis=1
)

data = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})[["id", "Number", "sentiment", "text"]].copy()

########## 4. Configure parameters & train ##########

REPEAT = 10
out_csv_name = f'{project}_NB.csv'
text_col = 'text'

data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

# MultinomialNB uses alpha
params = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
}

accuracies = []
precisions = []
recalls = []
f1_scores = []
auc_values = []

for repeated_time in range(REPEAT):
    train_text, test_text, y_train, y_test = train_test_split(
        data[text_col],
        data['sentiment'],
        test_size=0.2,
        random_state=repeated_time,
        stratify=data['sentiment']
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000
    )

    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)

    clf = MultinomialNB()
    grid = GridSearchCV(clf, params, cv=5, scoring='f1', error_score='raise')

    grid.fit(X_train, y_train)

    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(X_test)

    # For binary AUC, use probability of the positive class
    y_prob = best_clf.predict_proba(X_test)[:, 1]

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
    recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
    auc_values.append(roc_auc_score(y_test, y_prob))

final_accuracy = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall = np.mean(recalls)
final_f1 = np.mean(f1_scores)
final_auc = np.mean(auc_values)

print("=== Naive Bayes + TF-IDF Results ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

header_needed = not os.path.exists(out_csv_name)

df_log = pd.DataFrame({
    'repeated_times': [REPEAT],
    'Accuracy': [final_accuracy],
    'Precision': [final_precision],
    'Recall': [final_recall],
    'F1': [final_f1],
    'AUC': [final_auc],
    'CV_list(AUC)': [str(auc_values)]
})

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have been saved to: {out_csv_name}")