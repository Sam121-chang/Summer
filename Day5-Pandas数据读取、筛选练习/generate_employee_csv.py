import csv


def generate_employee_csv():
    # 定义CSV文件的列名
    fieldnames = ['姓名', '年龄', '部门', '工资']

    # 定义员工数据
    employees = [
        {'姓名': '张三', '年龄': 28, '部门': '研发', '工资': 18000},
        {'姓名': '李四', '年龄': 35, '部门': '销售', '工资': 12000},
        {'姓名': '王五', '年龄': 22, '部门': '行政', '工资': 8000},
        {'姓名': '赵六', '年龄': 32, '部门': '研发', '工资': 25000}
    ]

    try:
        # 打开CSV文件并写入数据
        with open('employee.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 写入列名
            writer.writeheader()

            # 写入员工数据
            writer.writerows(employees)

        print("CSV文件生成成功！")
    except Exception as e:
        print(f"生成CSV文件时出错: {e}")


if __name__ == "__main__":
    generate_employee_csv()