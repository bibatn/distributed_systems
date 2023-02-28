import pika
import uuid
import sys


class Client(object):
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.queue = result.method.queue

        self.channel.basic_consume(queue=self.queue, on_message_callback=self.callback, auto_ack=True)
    def callback(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, begin, end):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='', routing_key=routing_key,
                                   properties=pika.BasicProperties(reply_to=self.queue, correlation_id=self.corr_id, ),
                                   body=(begin + '|' + end))
        while not self.response:
            self.connection.process_data_events()

        return self.response


if __name__ == '__main__':
    while 1:
        in_ = input ("Enter begin end step: ")
        pointer = in_.find(' ')
        begin = in_[0:pointer]
        pointer1 = in_.find(' ', pointer + 1)
        end = in_[pointer + 1:pointer1]
        step = in_[pointer1+1:len(in_)]

        print(begin, end, step)

        routing_key = 'rpc_queue'

        client_rpc = Client()
        print('begin: ', begin, 'end: ', end, ' f')
        response = client_rpc.call(begin, end)
        print(response.decode('utf-8'))
