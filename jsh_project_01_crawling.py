from selenium import webdriver
from bs4 import BeautifulSoup
import time
import requests
import datetime as dt
import os
from selenium.webdriver.common.by import By 
'''
# 1. 브라우저 제어를 위한 객체 생성
driver = webdriver.Chrome()
driver.implicitly_wait(10)

# 크롤링 키워드 리스트
keywords = ['inuit+traditional+cloth']

for keyword in keywords:
    # 구글 크롤링
    url = "https://www.google.com/search"
    param = "?sca_esv=4e2a714bc4107f31&sca_upv=1&q={keyword}&udm=2&fbs=AEQNm0DmKhoYsBCHazhZSCWuALW8l8eUs1i3TeMYPF4tXSfZ9zKNKSjpwusJM2dYWg4btGKvTs8msUkFt41RLL2EsYFXj1HJ-6Tz3zY-OaA8p5OIwAWoskWKBUk7Wrmnfn4KU7j2wY01Bvk2SJxfFAN8F6MR6ZyWB_5kKhL1r7wLM0C2lAj0l-qOc3_ZqvOIBxls8ucop53t&sa=X&ved=2ahUKEwj-jMbH7OGGAxUbsVYBHbISCpIQtKgLegQIChAB&biw=1920&bih=945&dpr=1"
    api_url = url + param.format(keyword=keyword)

    driver.get(api_url)
    time.sleep(3)
    
    # 스크롤 2번
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
    time.sleep(3)
    #driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
    #time.sleep(3)
    
    # 더보기 버튼
    # 버튼 태그 얻기
    btn = driver.find_element(By.CSS_SELECTOR, 'div.GNJvt.ipz2Oe')
    # 버튼 태그 클릭
    btn.click()
    time.sleep(3)

    # 스크롤 20번
    for i in range(20):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
        time.sleep(3)    
    
    # 2. 이미지 수집
    # BeautifulSoup 객체생성
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    # img 태그 추출
    img = soup.select('img[src]')
    # print(len(img))
    # print('-' * 20)

    # 이미지 url 추출
    img_list = []
    for item in img:
        if not 'data:' in item.attrs['src']:
            img_list.append(item.attrs['src'])
    #print(len(img_list))
    #print('-' * 20)

    # url 중복 데이터 제거
    img_list = list(set(img_list))
    #print(len(img_list))
    #print('-' * 20)

    # 3. 이미지 저장
    # 1) user-agent 준비
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0'
    headers = {'User-Agent': user_agent, 'Referer': None}

    # 2) 이미지 저장 폴더 만들기
    # 폴더명
    datetime_str = dt.datetime.now().strftime("%y%m%d_%H%M%S")
    dirname = keyword + '_' + datetime_str
    # 폴더 생성
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    # 3) 이미지 저장하기
    for i, url in enumerate(img_list):
        print('>', end='')
        path = '%s/%03d.jpg' % (dirname, i + 1)

        try:
            r = requests.get(url, headers=headers, stream=True)
            if r.status_code != 200:
                raise Exception()  

            with open(path, 'wb') as f:
                f.write(r.raw.read())
                print('-----> 저장 성공')

        except Exception as e:
            print('-----> 저장 실패')
            continue

# 브라우저 종료
driver.quit()

# 4, 훈련, 테스트셋 폴더 생성
setdir = ['train', 'test']

# 하위 폴더
subdir = ['기모노_일본','델_몽골','레드코트_영국','만띠야_스페인','사리_인도','슈카_케냐', 
          '아트쿠크_이누이트','운쿠_페루','치파오_중국','킬트_스코틀랜드','토브_사우디아라비아',
          '판초_멕시코','한복_한국']

# 폴더 만들기
for i in setdir:
    if not os.path.exists(i):
        os.mkdir(i)
    for j in subdir:
        sub_dir = os.path.join(i, j)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
'''