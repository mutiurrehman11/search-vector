import psycopg2
from psycopg2.extras import psycopg2
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
import pickle
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Pipeline.MLReRanker import MLReRanker

# Database configuration
DB_CONFIG = {
    'host': '167.86.115.58',
    'port': 5432,
    'dbname': 'prospects_dev',
    'user': 'devuser',
    'password': 'testdev123'
}

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

def fetch_engagement_data(conn):
    """Fetches player engagement data from the database."""
    query = "SELECT * FROM player_engagement_events;"
    df = pd.read_sql(query, conn)
    return df


def prepare_training_data(df):
    """Prepares the engagement data for training the LTR model."""
    # Define relevance scores for engagement types
    relevance_scores = {
        'impression': 0,
        'profile_view': 1,
        'follow': 2,
        'message': 3,
        'save_to_playlist': 4
    }
    df['relevance'] = df['event_type'].map(relevance_scores)

    # Feature Engineering (example)
    df['query_length'] = df['query_context'].apply(lambda x: len(x.get('search_query', '')) if isinstance(x, dict) and 'search_query' in x else 0)

    df['query_context_str'] = df['query_context'].apply(json.dumps, sort_keys=True)

    # Create a group for each user-query combination
    df['group'] = df.groupby(['user_id', 'query_context_str']).ngroup()

    return df


def main():
    """Main function to train and save the model."""
    conn = get_db_connection()
    engagement_df = fetch_engagement_data(conn)
    print("Successfully fetched engagement data:")
    print(engagement_df.head())

    # Prepare data
    data_df = prepare_training_data(engagement_df)

    # Define features and target
    features = ['query_length'] # Add more features here
    target = 'relevance'
    group_col = 'group'

    # Split data by group
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(data_df, groups=data_df[group_col]))

    train_df = data_df.iloc[train_idx]
    test_df = data_df.iloc[test_idx]

    # Sort by group for LGBMRanker
    train_df = train_df.sort_values(by=group_col)
    test_df = test_df.sort_values(by=group_col)

    X_train = train_df[features].to_numpy()
    y_train = train_df[target].to_numpy()
    groups_train = train_df.groupby(group_col).size().to_numpy()

    X_test = test_df[features].to_numpy()
    y_test = test_df[target].to_numpy()
    groups_test = test_df.groupby(group_col).size().to_numpy()


    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    ranker.fit(
        X_train, y_train, group=groups_train,
        eval_set=[(X_test, y_test)],
        eval_group=[groups_test],
        callbacks=[lgb.early_stopping(10, verbose=True)]
    )

    # Save the model
    reranker = MLReRanker()
    reranker.model = ranker
    reranker.save_model("models/reranker.pkl")

    print("\nModel trained and saved as models/reranker.pkl")

    conn.close()

if __name__ == '__main__':
    main()