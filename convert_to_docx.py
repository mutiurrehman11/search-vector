import requests
import os

# Read the Markdown file
with open('api_documentation.md', 'r') as f:
    markdown_content = f.read()

# Set up the API request
url = 'https://api.cloudconvert.com/v2/convert'
headers = {
    'Authorization': 'Bearer YOUR_API_KEY',  # Replace with your CloudConvert API key
    'Content-Type': 'application/json'
}
data = {
    'input': 'markdown',
    'output': 'docx',
    'file': markdown_content
}

# Make the API request
response = requests.post(url, headers=headers, json=data)

# Save the DOCX file
with open('api_documentation.docx', 'wb') as f:
    f.write(response.content)