from bs4 import BeautifulSoup
from selenium import webdriver
import selenium
import urllib.request
import os

def find_images(name):
    if not os.path.exists(name):
        os.makedirs(name)
    driver = webdriver.Chrome("chromedriver")
    driver.get("https://www.google.com/search?q="+name+"&source=lnms&tbm=isch")
    for i in range(500):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            driver.find_element_by_xpath('//*[@id="smb"]').click()
        except:
            print("Can t find the button")
    bs = BeautifulSoup(driver.page_source, "html.parser")
    html_image = bs.findAll("img", {"class":"rg_ic rg_i"})
    for idx, element in enumerate(html_image):
        print(element.get("src"))
        try:
            urllib.request.urlretrieve(element.get("src"), name + "/" + name + "_" + str(idx) + ".png")
        except:
            print("Error")
    driver.close()

annimal = ["cat", "dog"]

for el in annimal:
    find_images(el)