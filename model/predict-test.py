import requests
import json

url = 'http://localhost:9696/predict'

user_id = 'xyz-123'

user_input = {"status": "A11",
             "duration": 6,
             "credit_history": "A34",
             "purpose": "A43",
             "credit_amount": 1169,
             "savings": "A65",
             "employment_since": "A75",
             "installment_rate": 4,
             "personal_status_and_sex": "A93",
             "others_debtors_guarantors": "A101",
             "residence_since": 4,
             "property": "A121",
             "age": 27,
             "installment_plans": "A143",
             "housing": "A152",
             "existing_credits": 2,
             "job": "A173",
             "people_maitenance": 1,
             "telephone": "A192",
             "foreign_worker": "A201"}


response = requests.post(url, json=user_input).json()

print(response)

if response['risk'] == True:
    print('Bad credit risk %s' % user_id)
else:
    print('Good credit risk %s' % user_id)