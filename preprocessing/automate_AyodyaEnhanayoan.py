import pandas as pd
from pathlib import Path
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

def preprocess_data(data, target_column, save_pipeline_path, save_header_path):
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    column_names = data.columns.drop(target_column)
    df_header = pd.DataFrame(columns=column_names)
    df_header.to_csv(save_header_path, index=False)
    print(f"header saved: {save_header_path}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    dump(preprocessor, save_pipeline_path)
    print(f"pipeline saved: {save_pipeline_path}")

    return X_train_transformed, X_test_transformed, y_train, y_test



if __name__ == "__main__":
    try:
        print("tes")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        raw_data_path = os.path.join(current_dir, "..", "heart_raw.csv")
        
        df = pd.read_csv(raw_data_path)
        
        X_train, X_test, y_train, y_test = preprocess_data(
            df, 
            target_column='HeartDisease', 
            save_pipeline_path=os.path.join(current_dir, 'pipeline.joblib'), 
            save_header_path=os.path.join(current_dir, 'header.csv')
        )

        X_train_df = pd.DataFrame(X_train)
        csv_output_path = os.path.join(current_dir, 'heart_preprocessed.csv')
        X_train_df.to_csv(csv_output_path, index=False)

        print(f"csv created: {csv_output_path}")
        print(f"size: {X_train_df.shape}")
        
    except Exception as e:
        print(f"error: {e}")