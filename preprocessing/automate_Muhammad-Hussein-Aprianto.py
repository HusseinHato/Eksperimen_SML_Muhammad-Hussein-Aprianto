import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_csv('../croprecommendation_raw.csv')

processed_df = processed_df.dropna()

processed_df = df.drop_duplicates()

numerical_cols = ['K', 'N', 'P', 'temperature', 'humidity', 'ph', 'rainfall']

for col in numerical_cols:
    Q1 = processed_df[col].quantile(0.25)
    Q3 = processed_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Menghitung outliers yang akan dihapus
    outliers_to_remove = len(processed_df[(processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)])
    
    # Hapus outliers
    processed_df = processed_df[
        (processed_df[col] >= lower_bound) & 
        (processed_df[col] <= upper_bound)
    ]
    
    print(f"{col}: {outliers_to_remove} outliers dihapus")

scaler = MinMaxScaler()
processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])

label_encoder = LabelEncoder()
processed_df['label'] = label_encoder.fit_transform(processed_df['label'])

processed_df.to_csv('./preprocessing/croprecommendation_preprocessing.csv', index=False)
