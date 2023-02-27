redis-server --port 6379 &
redis-server --port 6380 &
redis-server --port 6381 &
redis-cli -p 6381 &
replicaof localhost 6379 &
quit &
redis-server --port 6382 &
redis-cli -p 6382 &
replicaof localhost 6379 &
quit &
redis-server --port 6383 &
redis-cli -p 6382 &
replicaof localhost 6380 &
quit &
redis-server --port 6384 &
redis-cli -p 6382 &
replicaof localhost 6380 &
quit &
. env/bin/activate
python3 initialize_redis_servers.py
