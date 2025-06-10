"""
步骤分解：
1.发送请求：模拟浏览器发送请求，避免被拦截
2.解析HTML：用BeautifulSoup将网页内容转为可操作的树形结构
3.提取数据：通过标签和类名定位目标信息
4.保存数据：将结果写入文本文件
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from urllib.parse import urljoin


class CassSpider:
    def __init__(self):
        """初始化爬虫参数"""
        self.base_url = "https://www.cass.cn"  # 社科院官网
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        self.data = []  # 存储爬取的数据
        self.keywords = ["社会治理", "社区建设", "基层治理"]  # 搜索关键词
        # 明确禁用代理
        self.proxies = {
            'http': None,
            'https': None,
        }

    def search_articles(self, keyword, page=1):
        """搜索包含关键词的文章"""
        search_url = f"{self.base_url}/search/index.htm?q={keyword}&p={page}"
        try:
            # 禁用SSL验证并明确禁用代理
            response = requests.get(search_url, headers=self.headers, timeout=10, verify=False, proxies=self.proxies)
            response.raise_for_status()  # 检查请求是否成功
            soup = BeautifulSoup(response.text, 'html.parser')

            # 提取文章列表
            articles = soup.select(".search-result .result-item")
            for article in articles:
                title_elem = article.select_one(".result-title a")
                if not title_elem:
                    continue

                title = title_elem.text.strip()
                link = urljoin(self.base_url, title_elem['href'])
                date = article.select_one(".result-date").text.strip()
                source = article.select_one(".result-source").text.strip()

                # 获取文章详情
                content = self.get_article_content(link)

                self.data.append({
                    'title': title,
                    'date': date,
                    'source': source,
                    'link': link,
                    'content': content,
                    'keyword': keyword
                })
                print(f"已爬取: {title}")

                # 随机延时，避免过快请求
                time.sleep(random.uniform(1, 3))

            # 检查是否有下一页
            next_page = soup.select_one(".next")
            if next_page and 'disabled' not in next_page.get('class', []):
                return self.search_articles(keyword, page + 1)
            return self.data

        except Exception as e:
            print(f"搜索页面请求失败: {e}")
            return []

    def get_article_content(self, url):
        """获取文章详情内容"""
        try:
            # 禁用SSL验证并明确禁用代理
            response = requests.get(url, headers=self.headers, timeout=15, verify=False, proxies=self.proxies)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # 根据社科院文章页面结构提取内容
            content_div = soup.select_one(".article-content")
            if content_div:
                # 提取所有段落文本
                paragraphs = [p.text.strip() for p in content_div.select("p") if p.text.strip()]
                return "\n".join(paragraphs)
            return "内容无法提取"

        except Exception as e:
            print(f"文章详情请求失败: {e}")
            return "内容请求失败"

    def save_to_csv(self, filename="cass_social_governance.csv"):
        """保存数据到CSV文件"""
        if not self.data:
            print("没有数据可保存")
            return

        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已保存到 {filename}")

    def run(self):
        """运行爬虫主程序"""
        for keyword in self.keywords:
            print(f"开始搜索关键词: {keyword}")
            self.search_articles(keyword)
            time.sleep(random.uniform(3, 5))  # 关键词间延时

        self.save_to_csv()


# 运行爬虫
if __name__ == "__main__":
    # 禁用不安全请求的警告
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    spider = CassSpider()
    spider.run()