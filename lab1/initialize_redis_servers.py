import redis
import pandas as pd

r1 = redis.Redis(host='localhost', port=6379)
r2 = redis.Redis(host='localhost', port=6380)

def initialize_distributed_system(path_to_csv):
    df = pd.read_csv(path_to_csv, sep='\t', header=0, names=['Datetime (UTC)', 'T (°C)', 'P (hPa)', 'Humidity (%)'])
    values = df.values
    for idx, val in enumerate(values):
        key = val[0]
        # key = key.replace('-', '').replace('T', '').replace(':', '')
        value = ''
        for x in val:
            value += str(x) + ' '
        if(idx%2==0):
            r1.set(key, value)
        else:
            r2.set(key, value)
        # r1.zadd(key, value, 1)
    print(values[1])


initialize_distributed_system('svtl_meteo_20120101-20220101.csv')
