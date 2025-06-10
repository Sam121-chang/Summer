import requests
from bs4 import BeautifulSoup

# 1. 发送请求（添加请求头模拟浏览器）
url = "https://movie.douban.com/top250"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
response.raise_for_status()  # 检查请求是否成功，失败则抛出异常

# 2. 解析HTML
soup = BeautifulSoup(response.text, "html.parser")

# 3. 提取电影名称和评分
movies = []
for item in soup.find_all("div", class_="item"):  # 定位每个电影条目
    # 提取名称（排除英文别名）
    title = item.find("span", class_="title").text
    if "/" in title:
        main_title = title.split("/")[0].strip()  # 取中文主标题
    else:
        main_title = title

    # 提取评分
    rating = item.find("span", class_="rating_num").text
    movies.append(f"名称：{main_title}，评分：{rating}")

# 4. 保存到文件
with open("douban_top250.txt", "w", encoding="utf-8") as f:
    for movie in movies:
        f.write(movie + "\n")

print("数据爬取完成，已保存到douban_top250.txt")