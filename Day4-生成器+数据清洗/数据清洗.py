"""
数据清洗脚本：从杂乱到规整的自动化流水线
数据清洗的常见场景
问题类型	    示例	                        解决方案
缺失值	    年龄字段存在 NaN	            删除、填充（均值 / 中位数）
重复数据	    同一用户多条相同记录	        去重（drop_duplicates）
格式错误	    日期写成 "2023-13-01"	    正则表达式校验
异常值	    身高字段出现 300cm	        基于统计阈值过滤
不一致编码	中文乱码（如 "浣犲ソ"）	        指定正确编码（utf-8）
"""

#实战：CSV文件清洗脚本
#需求：清洗包含用户信息的CSV文件，处理缺失值
import pandas as pd
import re


def clean_data(input_file, output_file):
    # 1. 读取数据（设置低内存模式避免大文件溢出）
    df = pd.read_csv(input_file, low_memory=False)

    # 2. 处理缺失值
    missing_cols = ['age', 'income']  # 需要处理的列
    for col in missing_cols:
        if df[col].isna().sum() > 0:
            # 数值型用中位数填充
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            # 文本型用"Unknown"填充
            else:
                df[col].fillna('Unknown', inplace=True)

    # 3. 去重
    df.drop_duplicates(subset=['user_id'], keep='first', inplace=True)

    # 4. 格式校验（示例：手机号格式）
    phone_pattern = r'^1[3-9]\d{9}$'
    invalid_phones = ~df['phone'].apply(lambda x: bool(re.match(phone_pattern, str(x))))
    if invalid_phones.sum() > 0:
        print(f"发现{invalid_phones.sum()}条无效手机号，已替换为NaN")
        df.loc[invalid_phones, 'phone'] = None

    # 5. 异常值处理（示例：年龄超过100的设为中位数）
    df.loc[df['age'] > 100, 'age'] = df['age'].median()

    # 6. 保存清洗后的数据
    df.to_csv(output_file, index=False)
    print(f"清洗完成，保存至{output_file}")
    return df


# 使用示例
clean_data('raw_data.csv', 'cleaned_data.csv')



#进阶技巧：生成器与数据清洗结合
#场景：清洗超大数据文件（如 100GB），内存无法一次性加载。
def stream_clean_large_file(input_file, output_file, chunk_size=1000):
    with open(output_file, 'w') as outfile:
        header_written = False

        # 逐块读取数据
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            # 清洗当前块
            cleaned_chunk = clean_data_chunk(chunk)

            # 写入头部（仅第一次）
            if not header_written:
                cleaned_chunk.to_csv(outfile, index=False)
                header_written = True
            else:
                cleaned_chunk.to_csv(outfile, index=False, header=False)


def clean_data_chunk(chunk):
    # 与之前的清洗逻辑类似，但针对单个数据块
    chunk = chunk.drop_duplicates()
    chunk['age'].fillna(chunk['age'].median(), inplace=True)
    # 其他清洗操作...
    return chunk


# 处理超大数据文件
stream_clean_large_file('100GB_data.csv', 'cleaned.csv')