from bs4 import BeautifulSoup
import requests

# URL
URL = "https://www.shoemarker.co.kr/ASP/Product/ProductList.asp?SCode1=01&SearchType=C" 

# 이거 해야지만 html 형식을 쉽게 다룰 수 있음
_html = ""
resp = requests.get(URL)
if resp.status_code == 200:
    _html = resp.text
    
#  HTML 형식을 쉽게 다루게 해줌
soup = BeautifulSoup(_html, 'html.parser')

# 예시
# https://shopping-phinf.pstatic.net/main_8196178/81961785945.jpg?type=f168
# 39800

# 태그가 div이고 class이름이 ly-img인것을 모두 가져와요
img = soup.find_all("div", class_= "ly-img")

price = soup.find_all("span", class_= "num _price_reload")

img

#  가져온 것중에 text만 가져와요 
for link in price:
     print(link.get_text(''))