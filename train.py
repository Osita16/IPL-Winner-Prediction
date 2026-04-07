import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# 1. Load your data
df = pd.read_csv('your_dataset.csv') 

# 2. Define your features and target
# (This is just an example; your columns might be named differently)
X = df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']]
y = df['result'] 

# 3. Create a pipeline (The "Pipe")
step1 = ColumnTransformer(transformers=[
    ('trf', OneHotEncoder(sparse=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

step2 = LogisticRegression(solver='liblinear')

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# 4. Train the model
pipe.fit(X, y)

# 5. SAVE THE FILE (This creates the pipe.pkl you are missing!)
pickle.dump(pipe, open('pipe.pkl', 'wb'))

print("Success! pipe.pkl has been created.")