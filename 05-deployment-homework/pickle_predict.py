import pickle

# from sklearn.feature_extraction import DictVectorizer

input_file = f'pipeline.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

lead = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

X = dv.transform([lead])
y_pred = model.predict_proba(X)[0, 1]

print('input: ', lead)
print('output: ', y_pred)
print("ok")