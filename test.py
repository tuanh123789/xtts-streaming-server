import requests
import base64

data = {
    "text": "Icom describes the IC-V82 as a handheld radio which was exported to the Middle East from 2004 to 2014 and has not been shipped since then. The manufacturing of the batteries has also stopped, it says.",
    "language": "en",
    "silence_length": 0.5,
    "temperature": 65,
    "top_k": 50,
    "top_p": 0.8,
    "speed": 1
}
res = requests.post(
        url="http://localhost:8000/tts",
        json=data,
)

print(res.content)
with open("result.wav", "wb") as fp:
    fp.write(res.content)