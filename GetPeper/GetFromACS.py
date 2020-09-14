#!/usr/bin/env python3
# Dragon's Code
# encoding = utf-8
# date = 20200907
# 1.修改框架使其用于图形界面
# 1.1 传入keyword，运行get_paper_max_num()返回文献数量
# 1.2 传入paper_num，运行“main”函数生成列表信息
# 1.2 运行“run_it”带有交互作为程序使用


import urllib.request as ur
from http import cookiejar
from math import ceil
from re import sub
from bs4 import BeautifulSoup


class GetFromACS:
    def __init__(self):
        self.keyword = ''
        self.paper_number = 0
        # 创建标题、作者、期刊、日期、摘要、网址、doi列表
        self.title_list = []
        self.author_list = []
        self.publication_list = []
        self.date_list = []
        self.abstract_list = []
        self.web_list = []
        self.doi_list = []
        # 缓存网页信息
        self.html = ''

        self.opener = opener_creat()
        self.base_url = 'https://pubs.acs.org/action/doSearch?AllField='
        self.header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                     'Chrome/80.0.3987.132 Safari/537.36'}

    def open_html(self, url, timeout=15):
        req = ur.Request(url, headers=self.header)
        response = self.opener.open(req, timeout=timeout)
        html = response.read()
        return html

    def get_paper_max_num(self, page=0, size=100):
        keyword = sub(r'\W', '+', self.keyword)
        url = self.base_url + keyword + '&startPage=' + str(page) + '&pageSize=' + str(size)
        self.html = self.open_html(url, timeout=15)
        bs = BeautifulSoup(self.html, 'lxml')
        paper_max_num = bs.find(attrs={'class': 'result__count'}).string
        return paper_max_num

    def get_content(self):
        bs = BeautifulSoup(self.html, 'lxml')
        for content in bs.find_all(attrs={'class': 'issue-item clearfix'}):
            self.title_list.append(content.h5.text)
            self.author_list.append(content.ul.text)
            self.date_list.append(content.find(attrs={'class': 'pub-date-value'}).text)
            self.doi_list.append(content.h5.a['href'][5:])
            self.web_list.append('https://pubs.acs.org' + content.h5.a['href'])
            # 部分文献无Abstract
            abstract = content.find(attrs={'class': 'hlFld-Abstract'})
            if abstract:
                self.abstract_list.append(abstract.text)
            else:
                self.abstract_list.append(abstract)
            # chapter 与 article 格式不同
            if content.find(attrs={'class': 'infoType'}).string == 'Chapter':
                self.publication_list.append(
                    sub(r'\s+\W\s+', ' ', content.find(attrs={'class': 'issue-item_chapter'}).text))
            else:
                self.publication_list.append(content.find(attrs={'class': 'issue-item_jour-name'}).text)

    def main(self, num):
        self.paper_number = int(num)
        self.get_content()
        page = ceil(self.paper_number / 100)
        if page - 1:
            for i in range(1, page):
                self.get_paper_max_num(page=i)
                self.get_content()
        self.title_list = self.title_list[:self.paper_number]
        self.title_list = self.title_list[:self.paper_number]
        self.author_list = self.author_list[:self.paper_number]
        self.date_list = self.date_list[:self.paper_number]
        self.doi_list = self.doi_list[:self.paper_number]
        self.web_list = self.web_list[:self.paper_number]
        self.abstract_list = self.abstract_list[:self.paper_number]

    def run_it(self):
        self.keyword = input('请输入带查找的关键词：')
        print(self.get_paper_max_num())
        paper_num = input('请输入需要获取的数量：')
        self.main(paper_num)
        print(len(self.title_list))


def opener_creat():
    # 创建cookie
    cookie = cookiejar.CookieJar()
    cookie_handler = ur.HTTPCookieProcessor(cookie)
    http_handler = ur.HTTPHandler()
    https_handler = ur.HTTPSHandler()
    opener = ur.build_opener(cookie_handler, http_handler, https_handler)
    return opener


if __name__ == '__main__':
    get = GetFromACS()
    get.run_it()
