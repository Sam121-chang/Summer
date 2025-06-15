"""
核心流程
1.打开文件：open(path,mode,encoding)
2.读写文件：read()/write()/readlines()/writelines()
3.关闭文件：close() 或 with open() as f:

关键模式
r   只读   with open("a.txt','r")
w   写入   with open("a.txt", "w")
a	追加	  with open("a.txt", "a")
rb	二进制只读	with open("img.jpg", "rb")
"""
#实战示例 1
with open("source.txt","r") as src, open("target.txt" , "w") as tar:
    tar.write(src.read())

#实战练习 2 统计文本文件单词数
with open("text.txt","r") as f:
    words = f.read().split()
    print(f"单词数：{len(words)}")
"""
文件编码	
中文文件必须指定encoding="utf-8"，二进制文件用rb/wb模式

with 语句优势	
自动管理文件资源，避免因异常导致文件未关闭

"""