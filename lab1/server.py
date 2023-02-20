import pika
import redis
import os
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


# initialize_distributed_system('svtl_meteo_20200101-20210101.csv')

def callback(ch, method, properties, body):
    body = str(body)
    pointer = body.find('|')
    begin = body[2:pointer]
    end = body[pointer+1:len(body)]
    print(begin, ' ', end)
    pattern = ''
    for idx, x in enumerate(begin):
        if(begin[idx]==end[idx]):
            pattern = pattern + begin[idx]
        else:
            break
    pattern = pattern + '*'
    print(pattern)
    keys1 = r1.keys(pattern)
    keys2 = r2.keys(pattern)

    response1 = r1.mget(keys1)
    response2 = r2.mget(keys2)
    response = ''

    for idx, x in enumerate(response1+response2):
        # print(type(x.decode('utf-8')))
        response = response + x.decode('utf-8') + '\n';



    ch.basic_publish(exchange='', routing_key=properties.reply_to,
                     properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)



# callback('2020-01-01T00:40:00|2020-01-01T05:20:00')

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
queue = 'rpc_queue'


channel.queue_declare(queue=queue)
channel.basic_qos(prefetch_count=1) # one message per receiver at a time
channel.basic_consume(queue=queue, on_message_callback=callback)
channel.start_consuming()