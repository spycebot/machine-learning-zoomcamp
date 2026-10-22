import requests

url = 'http://localhost:9696/predict'

customer_id = "Kate Shannon"

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

response = requests.post(url, json=customer).json()

print(response)

if response['churn'] == True:
    print('>> Sending promo email to %s at %s' % (customer_id, response['timestamp']))
else:
    print('>> No email to be sent to %s at %s' % (customer_id, response['timestamp']))

