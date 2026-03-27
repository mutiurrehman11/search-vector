import requests
import json

url = "http://167.86.115.58:5000/api/v1/search"
data = {"position": "midfielder"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.json())