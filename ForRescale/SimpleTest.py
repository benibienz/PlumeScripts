""" Simple dummy code """
import random

results = []
for i in range(100000):
    results.append(random.random())

print(results[:10])