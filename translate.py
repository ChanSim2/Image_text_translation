import os
import sys
import urllib.request
import json

def get_translate(text):
    client_id = "s2yDfeed8RDWFeWyfhRF"
    client_secret = "qnEvDhxuZE"
    encText = urllib.parse.quote(text)
    data = "source=en&target=ko&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        res = json.loads(response_body.decode('utf-8'))
        return res['message']['result']['translatedText']
    else:
        print("Error Code:" + rescode)

#trans_text = get_translate("I want to go home")
#print(trans_text)

"""
import urllib.request

def get_translate(text):
    client_id = "s2yDfeed8RDWFeWyfhRF" # <-- client_id 기입
    client_secret = "qnEvDhxuZE" # <-- client_secret 기입

    data = {'text' : text,
            'source' : 'en',
            'target': 'ko'}

    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {"s2yDfeed8RDWFeWyfhRF":client_id,
              "qnEvDhxuZE":client_secret}

    response = urllib.request.post(url, headers=header, data=data)
    rescode = response.status_code

    if(rescode==200):
        send_data = response.json()
        trans_data = (send_data['message']['result']['translatedText'])
        return trans_data
    else:
        print("Error Code:" , rescode)
        
trans = get_translate("translate test")
print(trans)
"""
