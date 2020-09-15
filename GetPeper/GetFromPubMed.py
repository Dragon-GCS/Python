# Dragon's Code
# encoding = utf-8
# vision = Module
# date = 20200823
# 1.修改框架使其用于图形界面
# 1.1 传入keyword，运行get_page_and_PMID()返回文献数量
# 1.2 传入paper_number，运行“main”函数生成列表信息
# 1.2 运行“run_it”带有交互作为程序使用
# date = 20200909
# 1.修改部分内容，与其他GetFrom模块统一


import urllib.request as ur
from math import ceil
from re import sub
from http import cookiejar
from bs4 import BeautifulSoup


class GetFromPubmed:

    def __init__(self):
        self.keyword = ''
        self.paper_number = 0
        # 创建PMID、标题、作者、期刊、日期、摘要、网址、doi列表
        self.pmid_addr = []
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
        self.base_url = 'https://pubmed.ncbi.nlm.nih.gov/'
        self.header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                     'Chrome/80.0.3987.132 Safari/537.36'}
        # 文献搜索提示数量
        self.count = 0

    def open_html(self, url, timeout=15):
        req = ur.Request(url, headers=self.header)
        response = self.opener.open(req, timeout=timeout)
        html = response.read()
        return html

    def get_paper_max_num(self, page=1, size=100):
        keyword = sub(r'\W', '+', self.keyword)
        # 合成网址
        url = self.base_url + '?term=' + keyword + '&size=' + str(size) + '&page=' + str(page)
        self.html = self.open_html(url, 20)
        bs = BeautifulSoup(self.html, 'lxml')
        # 获取找到的文献总数
        paper_max_num = bs.find(attrs={'class': 'results-amount'}).span.string
        return paper_max_num

    def get_pmid(self):
        bs = BeautifulSoup(self.html, 'lxml')
        for pmid in bs.find_all(class_='docsum-pmid'):
            self.pmid_addr.append(self.base_url + pmid.string)
        if len(self.pmid_addr) > self.paper_number:
            self.pmid_addr = self.pmid_addr[:self.paper_number]

    def get_content(self, pmid):
        try:
            bs = BeautifulSoup(self.open_html(pmid), 'lxml')
            self.title_list.append(sub(r'\W\s+', '', bs.find(class_='heading-title').text))
            self.author_list.append(bs.find(class_='authors-list').a.string)
            self.publication_list.append(sub(r'\W\s+', '', bs.find(class_='journal-actions-trigger trigger').string))
            self.date_list.append(bs.find(class_='cit').string)
            try:
                self.abstract_list.append(sub(r'\n|\W+\s+', '', bs.find(class_='abstract-content selected').text))
            except Exception:
                self.abstract_list.append('No Abstract')
            try:
                self.web_list.append(bs.find(class_='full-text-links-list').a['href'])
            except Exception:
                self.web_list.append(pmid)
            try:
                self.doi_list.append(sub(r'\W\s+', '', bs.find(attrs={'data-ga-action': 'DOI'}).string))
            except Exception:
                self.doi_list.append('No DOI')
        except Exception as e:
            print('错误原因：' + str(e))
            print('错误文献：%s,第%d篇' % (self.pmid_addr[self.count - 1], self.count))
            self.title_list.append('Unconnected')
            self.author_list.append('Unconnected')
            self.publication_list.append('Unconnected')
            self.date_list.append('Unconnected')
            self.abstract_list.append('Unconnected')
            self.web_list.append(pmid)
            self.doi_list.append('Unconnected')

    def main(self, num):
        self.paper_number = int(num)
        # 确认PMID列表
        self.get_pmid()
        page = ceil(self.paper_number / 100)
        if page - 1:
            for i in range(2, page + 1):
                self.get_paper_max_num(page=i)
                self.get_pmid()
        for pmid in self.pmid_addr:
            self.count += 1
            print('正在获取第%i篇,共计%s篇' % (self.count, len(self.pmid_addr)))
            self.get_content(pmid)
        print('获取完成')

    def run_it(self):
        self.keyword = input('请输入需要查询的关键词：')
        print('开始在PubMed上查找关键词为“%s”的文献……' % self.keyword)
        if self.get_paper_max_num():
            print('共查找到%s篇文献' % self.get_paper_max_num())
            num = input('请输入需要获取信息的文献数量：')
            print('开始获取')
            self.main(num)

        else:
            print('未找到相关文献')


def opener_creat():
    # 创建cookie
    cookie = cookiejar.CookieJar()
    cookie_handler = ur.HTTPCookieProcessor(cookie)
    http_handler = ur.HTTPHandler()
    https_handler = ur.HTTPSHandler()
    opener = ur.build_opener(cookie_handler, http_handler, https_handler)
    return opener


if __name__ == '__main__':
    getpaper = GetFromPubmed()
    getpaper.run_it()
