import requests
import json
import pandas as pd
import time
import random
from datetime import datetime
import os


class StatsGovSpider:
    def __init__(self):
        # 国家统计局数据查询URL
        self.base_url = "https://data.stats.gov.cn/easyquery.htm"
        # 浏览器请求头（建议替换为自己浏览器的User-Agent）
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://data.stats.gov.cn/easyquery.htm?cn=HGND",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Host": "data.stats.gov.cn"
        }
        # 代理配置（如需使用代理，请取消注释并替换为有效代理）
        # self.proxies = {
        #     'http': 'http://proxy.example.com:8080',
        #     'https': 'http://proxy.example.com:8080'
        # }
        self.proxies = None
        # 数据存储目录
        self.data_dir = "stats_data"
        os.makedirs(self.data_dir, exist_ok=True)

    def _random_sleep(self, min_sec=2, max_sec=5):
        """随机延时，避免请求过于频繁"""
        sleep_time = random.uniform(min_sec, max_sec)
        print(f"等待 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)

    def _retry_request(self, url, data, max_retries=3):
        """带重试机制的请求函数"""
        for attempt in range(max_retries):
            try:
                self._random_sleep()
                response = requests.post(
                    url,
                    headers=self.headers,
                    data=data,
                    proxies=self.proxies,
                    timeout=20
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"请求失败 ({attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self._random_sleep(5, 10)  # 重试前等待更长时间
                else:
                    print("已达到最大重试次数，请求失败")
                    return None

    def get_indicator_tree(self):
        """获取指标树形目录"""
        print("正在获取指标目录...")
        data = {
            "m": "getTree",
            "wdcode": "zb",
            "dbcode": "hgnd"  # 年度数据
        }
        return self._retry_request(self.base_url, data)

    def find_indicator_by_name(self, tree_data, target_name, level=0, max_level=3):
        """递归查找目标指标ID"""
        if not tree_data or not tree_data.get("tree"):
            return None

        # 递归查找函数
        def dfs(nodes, current_level=0):
            if current_level > max_level:
                return None

            for node in nodes:
                if node.get("name") == target_name:
                    print(f"找到目标指标: {target_name} (ID: {node.get('code')})")
                    return node.get("code")

                if node.get("nodes"):
                    result = dfs(node.get("nodes"), current_level + 1)
                    if result:
                        return result
            return None

        return dfs(tree_data.get("tree"))

    def get_indicator_data(self, indicator_code, years=20):
        """获取指定指标的历史数据"""
        print(f"正在获取指标 {indicator_code} 的数据...")
        data = {
            "m": "QueryData",
            "dbcode": "hgnd",
            "rowcode": "zb",
            "colcode": "sj",
            "wds": [
                {"wdcode": "zb", "valuecode": indicator_code},
                {"wdcode": "sj", "valuecode": f"LAST{years}"}  # 最近N年数据
            ]
        }
        return self._retry_request(self.base_url, data)

    def parse_data(self, json_data, indicator_name):
        """解析JSON数据为DataFrame"""
        if not json_data or "returndata" not in json_data:
            print("数据格式异常，无法解析")
            return None

        returndata = json_data["returndata"]
        datanodes = returndata.get("datanodes", [])
        wdnodes = returndata.get("wdnodes", [])

        # 获取时间维度（年份）
        time_nodes = next((node for node in wdnodes if node.get("wdcode") == "sj"), None)
        if not time_nodes or not time_nodes.get("nodes"):
            print("未找到时间维度数据")
            return None

        # 构建时间映射
        time_map = {node.get("code"): node.get("cname") for node in time_nodes.get("nodes")}

        # 解析数据节点
        result = []
        for node in datanodes:
            try:
                # 获取年份
                sj_code = next((w.get("valuecode") for w in node.get("wds", []) if w.get("wdcode") == "sj"), None)
                year = time_map.get(sj_code, sj_code)

                # 获取数据值
                data_value = node.get("data", {}).get("data", "无数据")

                result.append({
                    "年份": year,
                    "指标名称": indicator_name,
                    "数据值": data_value
                })
            except Exception as e:
                print(f"解析数据节点时出错: {e}")

        if not result:
            print("未解析到有效数据")
            return None

        return pd.DataFrame(result)

    def save_data(self, df, indicator_name):
        """保存数据到Excel和CSV"""
        if df is None or len(df) == 0:
            print("无数据可保存")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{indicator_name}_{timestamp}"

        # 保存为Excel
        excel_path = os.path.join(self.data_dir, f"{filename}.xlsx")
        try:
            df.to_excel(excel_path, index=False, engine="openpyxl")
            print(f"数据已保存至: {excel_path}")
        except Exception as e:
            print(f"保存Excel失败: {e}")

        # 保存为CSV（备选方案）
        csv_path = os.path.join(self.data_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"数据已保存至: {csv_path}")

    def run(self, target_indicator="人口总数", years=20):
        """运行爬虫主流程"""
        print(f"=== 开始爬取国家统计局数据 - {target_indicator} ===")

        # 1. 获取指标目录
        tree_data = self.get_indicator_tree()
        if not tree_data:
            print("获取指标目录失败，程序终止")
            return

        # 2. 查找目标指标ID
        indicator_code = self.find_indicator_by_name(tree_data, target_indicator)
        if not indicator_code:
            print(f"未找到指标 '{target_indicator}'，程序终止")
            return

        # 3. 获取指标数据
        data_json = self.get_indicator_data(indicator_code, years)
        if not data_json:
            print("获取指标数据失败，程序终止")
            return

        # 4. 解析数据
        df = self.parse_data(data_json, target_indicator)
        if df is None:
            print("数据解析失败，程序终止")
            return

        # 5. 保存数据
        self.save_data(df, target_indicator)
        print(f"=== 爬取完成 - 共获取 {len(df)} 条数据 ===")


if __name__ == "__main__":
    # 创建爬虫实例
    spider = StatsGovSpider()

    # 可修改要爬取的指标名称（例如："国内生产总值"、"居民消费价格指数"等）
    target_indicator = "人口总数"

    # 运行爬虫
    spider.run(target_indicator, years=20)  # 获取最近20年数据