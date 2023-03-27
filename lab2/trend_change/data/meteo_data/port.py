import os
import sys

with open(sys.argv[1], 'r') as data:
  plaintext = data.read()

plaintext = plaintext.replace('\t', ',')
with open(sys.argv[1][:-4]+'_.csv', 'w') as f:
    f.write(plaintext)
