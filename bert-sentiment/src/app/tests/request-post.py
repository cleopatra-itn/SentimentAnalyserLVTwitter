import requests

url = 'http://localhost:5000/predict'
r = requests.post(url, json={
                  'sentence': '27 . aprīlī aicinām ģimenes uz jums veltītu pasākumu bossikā'})

print(r.json())
