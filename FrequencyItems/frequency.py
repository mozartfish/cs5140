import hashlib
from collections import Counter
import random
import math


def process_data(stream_data):
    characters = []
    with open(stream_data, 'r') as data:
        new_data = data.readlines()

    for line in new_data:
        for c in line:
            characters.append(c)

    return characters


# Count Min-Sketch
def count_min_sketch(stream_data, k, t):
    rows, cols = (5, 10)
    hash_matrix = [[0 for i in range(cols)] for j in range(rows)]
    #     print(hash_matrix)
    for c in range(len(stream_data)):
        character = stream_data[c]
        for h in range(5):
            if h == 0:
                hash1 = hashlib.sha3_224(character.encode())
                index = int(hash1.hexdigest(), 16) % 10
                hash_matrix[h][index] += 1
            elif h == 1:
                hash2 = hashlib.sha3_256(character.encode())
                index = int(hash2.hexdigest(), 16) % 10
                hash_matrix[h][index] += 1
            elif h == 2:
                hash3 = hashlib.sha3_384(character.encode())
                index = int(hash3.hexdigest(), 16) % 10
                hash_matrix[h][index] += 1
            elif h == 3:
                hash4 = hashlib.sha3_512(character.encode())
                index = int(hash4.hexdigest(), 16) % 10
                hash_matrix[h][index] += 1
            else:
                hash5 = hashlib.shake_256(character.encode())
                index = int(hash5.hexdigest(1), 16) % 10
                hash_matrix[h][index] += 1

    return hash_matrix


def find_character_count(character, hash_matrix):
    hash_list = []
    for h in range(5):
        if h == 0:
            hash1 = hashlib.sha3_224(character.encode())
            index = int(hash1.hexdigest(), 16) % 10
            count = hash_matrix[h][index]
            hash_list.append(count)
        elif h == 1:
            hash2 = hashlib.sha3_256(character.encode())
            index = int(hash2.hexdigest(), 16) % 10
            count = hash_matrix[h][index]
            hash_list.append(count)
        elif h == 2:
            hash3 = hashlib.sha3_384(character.encode())
            index = int(hash3.hexdigest(), 16) % 10
            count = hash_matrix[h][index]
            hash_list.append(count)
        if h == 3:
            hash4 = hashlib.sha3_512(character.encode())
            index = int(hash4.hexdigest(), 16) % 10
            count = hash_matrix[h][index]
            hash_list.append(count)
        else:
            hash5 = hashlib.shake_256(character.encode())
            index = int(hash5.hexdigest(1), 16) % 10
            count = hash_matrix[h][index]
            hash_list.append(count)

    print(f"The hash count array for {character}")
    print(hash_list)
    print()

    return min(hash_list)


s1 = process_data('S1.txt')
foo = count_min_sketch(s1, 10, 5)
thing = find_character_count('a', foo)
print(thing/len(s1))




