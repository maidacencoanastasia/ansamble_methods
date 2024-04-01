import requests

url = 'http://localhost:5000/api/data'
data_to_send = {
    # Your data here
}

response = requests.post(url, json=data_to_send)

if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print(f'Error: {response.status_code}')
