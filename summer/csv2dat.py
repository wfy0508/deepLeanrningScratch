import os
import re
import sys

file_name = 'test-0000'
xml_file_name = file_name + '.csv'
file_csv = open(xml_file_name, 'r')
file_dat = open(file_name + ".dat", 'w')
flag = True
testdata_num = 0
dic_task_des = {}
for line_num, line_content in enumerate(file_csv.readlines()):
    if flag:
        testdata_num = testdata_num + 1
        line_content = re.sub(',\n|,\r\n|\n', '', line_content)
        line_content = re.sub(',', '€€', line_content)
        file_dat.writelines(line_content + '\n')

file_dat.close()

task_des = {file_name: dic_task_des}
file_csv.close()
