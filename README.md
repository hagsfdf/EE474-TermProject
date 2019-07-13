이것은 EE474 term project repo입니다.

# Project Goal

해당 모델은 한정된 브랜드의 신발 사진을 가지고, 그 가격을 유추하는 VGG 모델입니다. 이를 위해 신발 이미지에서 브랜드를 유추하는 모델, 신발 이미지와 브랜드를 함께 사용하여 그 가격을 유추하는 모델 두 개를 개발하였습니다.

# 읽어봄직한 링크들

막 추가해주세요!

## 데이터베이스 만들기

- [importing](https://cyc1am3n.github.io/2018/09/13/how-to-use-dataset-in-tensorflow.html)
- [creating](https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image)

## 크롤링 관하여

- [basic crawling using bs4](https://twpower.github.io/84-how-to-use-beautiful-soup)
- [Save url to image](https://stackoverflow.com/questions/8286352/how-to-save-an-image-locally-using-python-whose-url-address-i-already-know)
- [multiplePageCrawling](https://l0o02.github.io/2018/06/14/python-crawling-pagination-1/)

### 1차적으로 이미지 뽑아 올 수 있는 신발 사이트들
- [슈마커](https://www.shoemarker.co.kr/)
- [데상트](https://shop.descentekorea.co.kr/product/list.do?redirectBrndCd=Q&cate=2204000)
- [구찌](https://www.gucci.com/kr/ko/ca/men/mens-shoes/mens-moccasins-loafers-c-men-shoes-moccasins-loafers)
- [크록스](https://www.crocs.co.kr/c/men)
- 그 외 많ㄷ...


# 수많은 문제점

## 모델 
1. MNIST CNN 
2. VGG-16 net (train accuracy 80\~90% / test accuracy **40\~50%**)
3. ResNet 

### Solution
1. Brand + Image를 피쳐로 쓰는거
    - ![MULTIPLEFEATURE](https://medium.com/all-of-us-are-belong-to-machines/gentlest-intro-to-tensorflow-part-3-matrices-multi-feature-linear-regression-30a81ebaaa6c)
2. BN, dropout, 

## 전처리 
1. RGB normalization
2. 윤곽선 검출 후에 ,
3. 신발코를 detect Orientation으로 돌려놓는 것
