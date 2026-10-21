import pickle
from datetime import datetime

C = 1.0

input_file = f'model_C={C}.bin'

script_load_start_time = datetime.now()
with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
script_load_end_time = datetime.now()
print('script_load_start_time: ', script_load_start_time.strftime('%Y-%m-%d %H:%M:%S'))
print('script_load_end_time: ', script_load_end_time.strftime('%Y-%m-%d %H:%M:%S'))
print('script_load_duration: ', (script_load_end_time - script_load_start_time).total_seconds(), 'seconds')

# Let's pretend this is a new customer
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

start_time = datetime.now()
print('start_time: ', start_time.strftime('%Y-%m-%d %H:%M:%S'))
X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]
end_time = datetime.now()
print('end_time: ', end_time.strftime('%Y-%m-%d %H:%M:%S'))
print('duration: ', (end_time - start_time).total_seconds(), 'seconds')

print('input: ', customer)
print('output: ', y_pred)