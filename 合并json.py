import os
import json
from natsort.natsort import natsorted

result = []

for path, dirname, filename in os.walk('.'):
    json_file_list = natsorted([file for file in filename if file.endswith('数.json')])

    for json_file in json_file_list:
        with open(json_file, 'r', encoding='utf8') as f:
            dic = json.load(f)
            dic = {'title': json_file, **dic}
            result.append(dic)


with open('合计结果.json', 'w', encoding='utf8') as f:
    f.write(json.dumps(result, ensure_ascii=False))
