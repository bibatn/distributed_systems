import pika
import redis
import multiprocessing
import time
import errno

r1 = []
r2 = []
r1.append(redis.Redis(host='localhost', port=6379)) # main server1
r2.append(redis.Redis(host='localhost', port=6380)) # main server2

r1.append(redis.Redis(host='localhost', port=6381)) # replica of server1
r1.append(redis.Redis(host='localhost', port=6382))
r2.append(redis.Redis(host='localhost', port=6383)) # replica of server2
r2.append(redis.Redis(host='localhost', port=6384))
def get_keys1(pattern, keys):
    try:
        keys[0] = r1[0].keys(pattern)
    except redis.ConnectionError as e:
        time.sleep(1000)


def get_keys2(pattern, keys):
    try:
        keys[1] = r2[0].keys(pattern)
    except redis.ConnectionError as e:
        time.sleep(1000)


def callback(ch, method, properties, body):
    count = [0, 0]  # number of fails on server [i]
    body = str(body)
    pointer = body.find('|')
    begin = body[2:pointer]
    end = body[pointer + 1:len(body)]
    print(begin, ' ', end)
    pattern = ''
    for idx, x in enumerate(begin):
        if (begin[idx] == end[idx]):
            pattern = pattern + begin[idx]
        else:
            break
    pattern = pattern + '*'
    print(pattern)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    p = multiprocessing.Process(target=get_keys1, args=(pattern, return_dict))
    jobs.append(p)
    p.start()
    p = multiprocessing.Process(target=get_keys2, args=(pattern, return_dict))
    jobs.append(p)
    p.start()
    print('I am here!')
    for idx, proc in enumerate(jobs):
        proc.join(timeout=5)
        print('HERE!!')
        if proc.is_alive():
            proc.terminate()
            count[idx] += 1
            if(idx==0):
                r1[0], r1[count[idx]] = r1[count[idx]], r1[0] # swap server on replica ! add check for end of number fails
                jobs[idx] = multiprocessing.Process(target=get_keys1, args=(pattern, return_dict))
                jobs[idx].start()
                jobs[idx].join() # add what to do
            if(idx==1):
                r2[0], r2[count[idx]] = r2[count[idx]], r2[0]  # swap server on replica ! add check for end of number fails
                jobs[idx] = multiprocessing.Process(target=get_keys2, args=(pattern, return_dict))
                jobs[idx].start()
                jobs[idx].join()  # add what to do

    print('I am here1')

    keys1 = return_dict[0]
    keys1.sort()
    keys2 = return_dict[1]
    keys2.sort()

    response1 = r1[0].mget(keys1)
    response2 = r2[0].mget(keys2)
    response = ''

    for idx, x in enumerate(response1 + response2):
        # print(type(x.decode('utf-8')))
        response = response + x.decode('utf-8') + '\n';


    print ('I am here 3')

    ch.basic_publish(exchange='', routing_key=properties.reply_to,
                     properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)


# callback('2020-01-01T00:40:00|2020-01-01T05:20:00')

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
queue = 'rpc_queue'

channel.queue_declare(queue=queue)
channel.basic_qos(prefetch_count=1)  # one message per receiver at a time
channel.basic_consume(queue=queue, on_message_callback=callback)
channel.start_consuming()
