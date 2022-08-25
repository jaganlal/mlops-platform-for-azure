import requests

url = "http://7e85f703-478b-4890-8ea3-a21ab2e6c157.centralus.azurecontainer.io/score"

payload="{\"SepalLengthCm\": 6.6, \"SepalWidthCm\": 3, \"PetalLengthCm\": 4.4, \"PetalWidthCm\": 1.4}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
#print(response.content)
print(response.text)

