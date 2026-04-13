########## 1. Import required libraries ##########

import os
import re
import pandas as pd
import numpy as np

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Classifier
from sklearn.svm import LinearSVC

# Text cleaning & stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

########## 2. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', str(text))

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text))

# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join(
        [word for word in str(text).split() if word.lower() not in final_stop_words_list]
    )

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
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

########## 3. Download & read data ##########

# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'tensorflow'
path = f'datasets/{project}.csv'

if not os.path.exists(path):
    raise FileNotFoundError(f"Dataset not found: {path}")

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999).reset_index(drop=True)  # Shuffle

# Fill missing values
pd_all['Title'] = pd_all['Title'].fillna('')
pd_all['Body'] = pd_all['Body'].fillna('')

# Merge Title and Body into a single column
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if row['Body'].strip() else row['Title'],
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})

pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

########## 4. Configure parameters & Start training ##########

# 1) Data file to read
datafile = 'Title+Body.csv'

# 2) Number of repeated experiments
REPEAT = 10

# 3) Output CSV file name
out_csv_name = f'{project}_SVM.csv'

# Read and clean data
data = pd.read_csv(datafile).fillna('')
text_col = 'text'

data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

# Hyperparameter grid for SVM
params = {
    'C': [0.01, 0.1, 1, 10, 100]
}

# Lists to store metrics across repeated runs
accuracies = []
precisions = []
recalls = []
f1_scores = []
auc_values = []

for repeated_time in range(REPEAT):
    # --- 4.1 Split into train/test ---
    train_text, test_text, y_train, y_test = train_test_split(
        data[text_col],
        data['sentiment'],
        test_size=0.2,
        random_state=repeated_time,
        stratify=data['sentiment']
    )

    # --- 4.2 TF-IDF vectorization ---
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000
    )

    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)

    # --- 4.3 SVM model & GridSearch ---
    clf = LinearSVC(max_iter=10000)

    grid = GridSearchCV(clf,
        params,
        cv=5,
        scoring='f1',
        error_score='raise',
    )

    grid.fit(X_train, y_train)

    # Retrieve the best model
    best_clf = grid.best_estimator_

    # --- 4.4 Make predictions & evaluate ---
    y_pred = best_clf.predict(X_test)

    # For ROC AUC with LinearSVC, use decision_function scores
    y_score = best_clf.decision_function(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Precision (macro)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precisions.append(prec)

    # Recall (macro)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recalls.append(rec)

    # F1 Score (macro)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_scores.append(f1)

    # AUC
    auc_val = roc_auc_score(y_test, y_score)
    auc_values.append(auc_val)

    #print(f"Run {repeated_time + 1}/{REPEAT} | Best C: {grid.best_params_['C']} | AUC: {auc_val:.4f}")

# --- 4.5 Aggregate results ---
final_accuracy = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall = np.mean(recalls)
final_f1 = np.mean(f1_scores)
final_auc = np.mean(auc_values)

print("\n=== SVM + TF-IDF Results ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

# Save final results to CSV
header_needed = not os.path.exists(out_csv_name)

df_log = pd.DataFrame(
    {
        'repeated_times': [REPEAT],
        'Accuracy': [final_accuracy],
        'Precision': [final_precision],
        'Recall': [final_recall],
        'F1': [final_f1],
        'AUC': [final_auc],
        'CV_list(AUC)': [str(auc_values)]
    }
)

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have been saved to: {out_csv_name}")