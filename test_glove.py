import numpy as np
import pprint


embeddings_dict = {}
# words = []
with open("/home/chroner/PhD_remote/RL_Event_Schema_Induction/data/external/glove.840B.300d.txt", 'r',
          encoding="utf-8", errors='ignore') as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
        except ValueError:
            vector = np.asarray(values[-300:], "float32")
            embeddings_dict[values[0] + " " + values[1]] = vector
            print(values[0])
            print(values[0] + " " + values[1])


# pprint.pprint(embeddings_dict)
# for word in words:
#     print(word)
