import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'title':"Kasganj violence", 'text':"Violence happened at Kaasganj on 26 January"})

print(r.json())