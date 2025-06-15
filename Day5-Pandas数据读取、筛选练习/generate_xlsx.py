import xlsxwriter

def generate_sample_xlsx():
    # 创建一个工作簿并添加一个工作表
    workbook = xlsxwriter.Workbook('employee.xlsx')
    worksheet = workbook.add_worksheet()

    # 定义表头
    headers = ['姓名', '年龄', '职业', '收入']

    # 示例数据
    data = [
        ['张三', 28, '工程师', 12000],
        ['李四', 34, '医生', 18000],
        ['王五', 42, '教师', 9500],
        ['赵六', 36, '律师', 25000],
        ['钱七', 29, '设计师', 11000]
    ]

    # 写入表头
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)

    # 写入数据
    for row, row_data in enumerate(data, 1):
        for col, value in enumerate(row_data):
            worksheet.write(row, col, value)

    # 添加一个简单的图表
    chart = workbook.add_chart({'type': 'column'})
    chart.add_series({
        'categories': '=Sheet1!$A$2:$A$6',
        'values': '=Sheet1!$D$2:$D$6',
        'name': '收入对比'
    })
    chart.set_title({'name': '不同职业的收入对比'})
    chart.set_x_axis({'name': '姓名'})
    chart.set_y_axis({'name': '收入 (元)'})
    worksheet.insert_chart('F2', chart)

    # 关闭工作簿
    workbook.close()
    print("Excel文件生成成功！")

if __name__ == "__main__":
    generate_sample_xlsx()