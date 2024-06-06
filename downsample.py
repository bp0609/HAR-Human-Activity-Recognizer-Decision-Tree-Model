import pandas as pd

# Load your CSV file into a DataFrame
df1 = pd.read_csv('/Collected Data/laying_5.csv')


df1["time"] = pd.to_datetime(df1["time"], unit='s')  # Assuming 'time' is in seconds

# Set the 'time' column as the index
df1.set_index("time", inplace=True)

# Resample to 50Hz and interpolate to fill missing values
df1_resampled = df1.resample('20ms').mean()  # 50Hz
df1_resampled = df1_resampled.interpolate(method='linear')

# If needed, reset the index to have 'time' as a regular column
df1_resampled.reset_index(inplace=True)


df1_resampled.to_csv('/Collected Data/laying_5.csv')