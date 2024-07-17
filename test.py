import requests
import base64

data = {
    "text": "hello my name is john",
    "language": "en",
    "silence_length": [0.4, 0.5]
}
res = requests.post(
        url="http://localhost:8000/tts",
        json=data,
)

with open("result.wav", "wb") as fp:
    fp.write(res.content)