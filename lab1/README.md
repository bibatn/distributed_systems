Зависимости:  
python3 -m venv env  
pip install pika  
pip install redis   
pip install pandas   
. env/bin/activate  

Посмотреть запущенные процессы redis:  
ps aux | grep redis  
  
Запуск:  
Запустить сервера redis(желательно скриптом, 2 сервера обязательно должны быть на портах 6379, 6380)    
./run_configuration.sh    
Выполнить инициализацию серверов   
python3 initialize_redis_servers.py   
python3 server.py   
python3 client.py 2020-01-01T00:40:00 2020-01-01T05:20:00   
Очистить сервис redis:  
redis-cli flushall
Замечания:   
- Запускать сервера redis  и реплики скриптом    
- Данные распределить пополам по временим   
- добавить переключение на реплику по истечению таймаута   


2020-01-01 2020-01-02 30