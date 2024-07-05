import requests

r = requests.get("https://go.dev/")
#print(r.content)
print("Response from Website",r.status_code)
