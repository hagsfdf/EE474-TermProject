from bs4 import BeautifulSoup
import requests
import urllib.request
from time import sleep
from random import randint
import os
import numpy as np
from refine_image import refine_image
# 남성화 페이지 갯수는 이렇습니다..

#pages = [str(1)]





def men():
    if not os.path.exists('imageGucciMen/'):
        os.mkdir('imageGucciMen/')
    else:
        print("you may delete the contents in the image file")
    pages = [str(i) for i in range(1,4)]
    basicURL = "https://www.gucci.com/kr/ko/ca/men/mens-shoes-c-men-shoes/"
    result = []
    univNum = 0
    for page in pages:
        URL = basicURL + page

        # 이거 해야지만 html 형식을 쉽게 다룰 수 있음
        _html = ""
        resp = requests.get(URL)
        # 반드시 필요합니다.
        sleep(randint(5,12))
        if resp.status_code == 200:
            _html = resp.text
        #  HTML 형식을 쉽게 다루게 해줌
        soup = BeautifulSoup(_html, 'html.parser')
        # 태그가 div이고 class이름이 ly-img인것을 모두 가져와요
        img = soup.find_all("div", class_= "product-tiles-grid-item-image")

        webPrice = soup.find_all("p", class_= "price")
        for num in range(len(webPrice)):
            univNum += 1
            imgLink = img[num].img['src']
            fullLink = 'https:' + imgLink
            textPrice = webPrice[num].get_text('')
            # print(fullLink, textPrice)
            textPrice = textPrice.strip()
            price = int(''.join(textPrice[2:].split(',')))
            dirIm = "imageGucciMen/"+ "{:06d}".format(univNum)+ ".jpg"
            refine_image(fullLink, dirIm)
            result.append((dirIm, price))
    dtype = [('dir','U28'),('price',int)]
    result = np.array(result, dtype=dtype)
    np.save("train_dataGucciMen.npy", result)
    print('done!')

def women():
    if not os.path.exists('imageGucciWomen/'):
        os.mkdir('imageGucciWomen/')
    else:
        print("you may delete the contents in the image file")
    pages = [str(i) for i in range(1,5)]
    basicURL = 'https://www.gucci.com/kr/ko/ca/women/womens-shoes-c-women-shoes/'
    result = []
    univNum = 0
    for page in pages:
        URL = basicURL + page

        # 이거 해야지만 html 형식을 쉽게 다룰 수 있음
        _html = ""
        resp = requests.get(URL)
        # 반드시 필요합니다.
        sleep(randint(5,12))
        if resp.status_code == 200:
            _html = resp.text
        #  HTML 형식을 쉽게 다루게 해줌
        soup = BeautifulSoup(_html, 'html.parser')
        # 태그가 div이고 class이름이 ly-img인것을 모두 가져와요
        img = soup.find_all("div", class_= "product-tiles-grid-item-image")

        webPrice = soup.find_all("p", class_= "price")
        for num in range(len(webPrice)):
            univNum += 1
            imgLink = img[num].img['src']
            fullLink = 'https:' + imgLink
            textPrice = webPrice[num].get_text('')
            # print(fullLink, textPrice)
            textPrice = textPrice.strip()
            price = int(''.join(textPrice[2:].split(',')))
            dirIm = "imageGucciWomen/"+ "{:06d}".format(univNum)+ ".jpg"
            urllib.request.urlretrieve(fullLink, dirIm)
            result.append((dirIm, price))
    dtype = [('dir','U28'),('price',int)]
    result = np.array(result, dtype=dtype)
    np.save("train_dataGucciWomen.npy", result)
    print('done!')

if __name__ == '__main__':
    men()
    women()
    
