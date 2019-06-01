from bs4 import BeautifulSoup
import requests
import urllib.request
from time import sleep
from random import randint
import os
import numpy as np
from refine_image import refine_image


def nike(univNum):
    result = []
    pages = [str(i) for i in range(1,7)]

    basicURL = "https://www.shoemarker.co.kr/ASP/Product/Brand.asp?SearchType=C&SBrandCode=NK&SCode1=01&SCode2=&SCode3=&SSort=1&Page="

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
        img = soup.find_all("div", class_= "ly-img")

        webPrice = soup.find_all("span", class_= "ns-type-bl-eb18x")
        for num in range(len(webPrice)):
            imgLink = img[num].img['src']
            fullLink = 'https://www.shoemarker.co.kr' + imgLink
            textPrice = webPrice[num].get_text('')
            # print(fullLink, textPrice)
            price = int(''.join(textPrice[:-1].split(',')))
            dirIm = "imageBrand/"+ "{:06d}".format(univNum)+ ".jpg"
            univNum += 1
            urllib.request.urlretrieve(fullLink, dirIm)
            result.append((dirIm, price, 'nike'))
    dtype = [('dir','U30'),('price',int),('brand', 'U12')]
    result = np.array(result, dtype=dtype)
    print('nike done!')
    return univNum, result


def adidas(univNum):
    result = []
    pages = [str(i) for i in range(1,12)]

    basicURL1 = "https://www.shoemarker.co.kr/ASP/Product/Brand.asp?SearchType=S&SBrandCode=AD&SSort=1&Page="
    basicURL2 = "&SPrice=0&EPrice=30"

    for page in pages:
        URL = basicURL1 + page + basicURL2

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
        img = soup.find_all("div", class_= "ly-img")

        webPrice = soup.find_all("span", class_= "ns-type-bl-eb18x")
        webBrand = soup.find_all("div", class_ = "ns-type-bl-eb13x")
        for num in range(len(webPrice)):
            imgLink = img[num].img['src']
            fullLink = 'https://www.shoemarker.co.kr' + imgLink
            textPrice = webPrice[num].get_text('')
            textBrand = webBrand[num].get_text('')
            brand = textBrand.lower()
            # print(fullLink, textPrice)
            price = int(''.join(textPrice[:-1].split(',')))
            dirIm = "imageBrand/"+ "{:06d}".format(univNum)+ ".jpg"
            univNum += 1
            urllib.request.urlretrieve(fullLink, dirIm)
            result.append((dirIm, price, 'adidas'))
    dtype = [('dir','U30'),('price',int),('brand', 'U12')]
    result = np.array(result, dtype=dtype)
    print('adidas done!')
    return univNum, result

def puma(univNum):
    result = []
    pages = [str(i) for i in range(1,6)]

    basicURL1 = "https://www.shoemarker.co.kr/ASP/Product/Brand.asp?SearchType=S&SBrandCode=PU&SSort=1&Page="
    basicURL2 = "&SPrice=0&EPrice=30"


    for page in pages:
        URL = basicURL1 + page + basicURL2

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
        img = soup.find_all("div", class_= "ly-img")

        webPrice = soup.find_all("span", class_= "ns-type-bl-eb18x")
        for num in range(len(webPrice)):
            imgLink = img[num].img['src']
            fullLink = 'https://www.shoemarker.co.kr' + imgLink
            textPrice = webPrice[num].get_text('')
            # print(fullLink, textPrice)
            price = int(''.join(textPrice[:-1].split(',')))
            dirIm = "imageBrand/"+ "{:06d}".format(univNum)+ ".jpg"
            univNum += 1
            urllib.request.urlretrieve(fullLink, dirIm)
            result.append((dirIm, price, 'puma'))
    dtype = [('dir','U30'),('price',int),('brand', 'U12')]
    result = np.array(result, dtype=dtype)
    print('puma done!')
    return univNum, result


def fila(univNum):
    result = []
    pages = [str(i) for i in range(1,9)]

    basicURL = "https://www.shoemarker.co.kr/ASP/Product/Brand.asp?SearchType=C&SBrandCode=FILA&SCode1=01&SCode2=&SCode3=&SSort=1&Page="


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
        img = soup.find_all("div", class_= "ly-img")

        webPrice = soup.find_all("span", class_= "ns-type-bl-eb18x")
        for num in range(len(webPrice)):
            imgLink = img[num].img['src']
            fullLink = 'https://www.shoemarker.co.kr' + imgLink
            textPrice = webPrice[num].get_text('')
            # print(fullLink, textPrice)
            price = int(''.join(textPrice[:-1].split(',')))
            dirIm = "imageBrand/"+ "{:06d}".format(univNum)+ ".jpg"
            univNum += 1
            urllib.request.urlretrieve(fullLink, dirIm)
            result.append((dirIm, price, 'fila'))
    dtype = [('dir','U30'),('price',int),('brand', 'U12')]
    result = np.array(result, dtype=dtype)
    print('fila done!')
    return univNum, result

def descente(univNum):
    pages = [str(i) for i in range(1,6)]
    basicURL = """https://shop.descentekorea.co.kr/product/list.do?cate=1140000\
&cateList=1140000&brandList=&meterialList=&playerList=&clothesSizeList=&suppliesSizeList=&\
shoesSizeList=&priceList=100000%7C300000&etcList=&brandIdList=&meterialIdList=&playerIdList=&\
clothesSizeIdList=&suppliesSizeIdList=&shoesSizeIdList=&priceIdList=checkbox_price_1&etcIdList=&schColorList=&\
listsize=20&sort=new&dcRate=&page="""
    result = []
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
        img = soup.find_all("div", class_= "goods-list__items-img")

        webPrice = soup.find_all("div", class_= "goods-list__items-price")
        for num in range(len(webPrice)):
            univNum += 1
            imgLink = img[num].img['src']
            fullLink = 'https:' + imgLink
            textPrice = webPrice[num].get_text('')
            # print(fullLink, textPrice)
            textPrice = textPrice.strip()
            price = int(''.join(textPrice[:-1].split(',')))
            dirIm = "imageBrand/"+ "{:06d}".format(univNum)+ ".jpg"
            refine_image(fullLink, dirIm)
            result.append((dirIm, price, 'descente'))
    dtype = [('dir','U30'),('price',int), ('brand', 'U12')]
    result = np.array(result, dtype=dtype)
    print('descente done!')
    return univNum, result


if __name__ == '__main__':
    if not os.path.exists('imageBrand/'):
        os.mkdir('imageBrand/')
    else:
        print("you may delete the contents in the image file")

    univNum = 0
    univNum, result_nike = nike(univNum)
    univNum, result_adidas = adidas(univNum)
    univNum, result_puma = puma(univNum)
    univNum, result_fila = fila(univNum)
    univNum, result_descente = descente(univNum)
    result = np.concatenate((result_nike, result_adidas,result_puma, result_fila, result_descente))
    np.save('train.npy', result)
    print(univNum)

