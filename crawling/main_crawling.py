from bs4 import BeautifulSoup
import requests
import urllib.request
from time import sleep
from random import randint

# 남성화 페이지 갯수는 이렇습니다..
MAX = 1325 // 20

pages = [str(i) for i in range(1,20)]

basicURL = "https://www.shoemarker.co.kr/ASP/Product/ProductList.asp?SCode1=01&SearchType=C"



def main():
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

        price = soup.find_all("span", class_= "ns-type-bl-eb18x")

        for link in img:
            imgLink = 'https://www.shoemarker.co.kr'+link.img['src']
            # urllib.request.urlretrieve(imgLink, link.img['alt']+ ".jpg")


        for num in range(len(price)):
            imgLink = img[num].img['src']
            fullLink = 'https://www.shoemarker.co.kr' + imgLink
            textPrice = price[num].get_text()
            print(fullLink, textPrice)
            # 가격을 제목으로 하는 사진을 저장하는 코드입니다.
            # urllib.request.urlretrieve(fullLink, "image/"+''.join(textPrice[:-1].split(',')) + '.png')

if __name__ == '__main__':
    main()