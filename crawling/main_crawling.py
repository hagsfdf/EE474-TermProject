from bs4 import BeautifulSoup
import requests
import urllib.request
from time import sleep
from random import randint
import os
import numpy as np

# 남성화 페이지 갯수는 이렇습니다..
MAX = 1325 // 20

pages = [str(i) for i in range(1,MAX)]


basicURL = "https://www.shoemarker.co.kr/ASP/Product/ProductList.asp?SCode1=01&SearchType=C"

if not os.path.exists('image/'):
    os.mkdir('image/')
else:
    print("you may delete the contents in the image file")

def main():
    result = []

    for page in pages:
        URL = basicURL + "&Page=" + page

        # 이거 해야지만 html 형식을 쉽게 다룰 수 있음
        _html = ""
        resp = requests.get(URL)
        # 반드시 필요합니다.
        sleep(randint(5,15))
        if resp.status_code == 200:
            _html = resp.text
        #  HTML 형식을 쉽게 다루게 해줌
        soup = BeautifulSoup(_html, 'html.parser')
        # 태그가 div이고 class이름이 ly-img인것을 모두 가져와요
        img = soup.find_all("div", class_= "ly-img")

        webPrice = soup.find_all("span", class_= "ns-type-bl-eb18x")

        for num in range(len(webPrice)):
            imgLink = img[num].img['src']
            fullLink = 'https://www.shoemarker.co.kr' + imgLink
            textPrice = webPrice[num].get_text('')
            # print(fullLink, textPrice)
            price = int(''.join(textPrice[:-1].split(',')))
            dirIm = "image/"+ "{:06d}".format(num+1+int(page)*20)+ ".jpg"
            urllib.request.urlretrieve(fullLink, dirIm)
            result.append((dirIm, price))
    dtype = [('dir','U16'),('price',int)]
    result = np.array(result, dtype=dtype)
    np.save("train_data.npy", result)
    print('done!')

if __name__ == '__main__':
    main()
    
