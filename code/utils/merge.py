import json
import numpy as np
import os

def merge(count,name,city):
    print('Merging...')
    result=list()
    for i in range(count):
        with open(os.path.join('../result', city, 'result_{}.json'.format(i)), 'r') as f:
            temp=json.load(f)
        result.append(temp)
        os.remove(os.path.join('../result', city, 'result_{}.json'.format(i)))

    print('Saving...')
    with open(os.path.join('../result', city, name), 'w') as f:
        json.dump(result,f)