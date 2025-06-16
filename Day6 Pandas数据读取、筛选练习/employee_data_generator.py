import os
import random
import pandas as pd
from pathlib import Path

# 确保中文显示正常
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)


class EmployeeDataGenerator:
    """员工数据生成器类，用于生成模拟的员工数据并保存到文件"""

    def __init__(self):
        # 数据配置
        self.departments = ["技术部", "市场部", "销售部", "人力资源部", "财务部", "行政部"]
        self.min_age = 22
        self.max_age = 60
        self.min_salary = 3000
        self.max_salary = 30000
        self.genders = ["男", "女", "其他"]

    def generate_data(self, num_entries=100):
        """生成员工数据

        Args:
            num_entries: 生成的数据条目数
        Returns:
            包含员工数据的DataFrame
        """
        data = {
            "员工ID": [f"EMP{str(i + 1).zfill(4)}" for i in range(num_entries)],
            "部门": [random.choice(self.departments) for _ in range(num_entries)],
            "年龄": [random.randint(self.min_age, self.max_age) for _ in range(num_entries)],
            "工资": [round(random.uniform(self.min_salary, self.max_salary), 2) for _ in range(num_entries)],
            "性别": [random.choice(self.genders) for _ in range(num_entries)]
        }
        return pd.DataFrame(data)

    def save_to_csv(self, df, directory, filename="employee_data.csv"):
        """将数据保存为CSV文件

        Args:
            df: 包含数据的DataFrame
            directory: 保存目录
            filename: 文件名
        """
        path = os.path.join(directory, filename)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"数据已保存到 {path}")

    def save_to_excel(self, df, directory, filename="employee_data.xlsx"):
        """将数据保存为Excel文件

        Args:
            df: 包含数据的DataFrame
            directory: 保存目录
            filename: 文件名
        """
        path = os.path.join(directory, filename)
        df.to_excel(path, index=False)
        print(f"数据已保存到 {path}")

    def save_to_json(self, df, directory, filename="employee_data.json"):
        """将数据保存为JSON文件

        Args:
            df: 包含数据的DataFrame
            directory: 保存目录
            filename: 文件名
        """
        path = os.path.join(directory, filename)
        df.to_json(path, orient="records", force_ascii=False, indent=2)
        print(f"数据已保存到 {path}")


def main():
    """主函数，程序入口点"""
    try:
        # 创建数据生成器实例
        generator = EmployeeDataGenerator()

        # 生成数据（默认为100条记录）
        print("正在生成员工数据...")
        df = generator.generate_data(100)

        # 显示数据基本信息和前几行
        print("\n数据基本信息：")
        df.info()

        print("\n数据前几行预览：")
        print(df.head().to_string())

        # 创建保存目录（如果不存在）
        save_dir = "employee_data"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 保存数据到不同格式
        print("\n正在保存数据...")
        generator.save_to_csv(df, save_dir)
        generator.save_to_excel(df, save_dir)
        generator.save_to_json(df, save_dir)

        print("\n数据生成与保存完成！")

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()