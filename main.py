import numpy as np
from ans import encode_message, decode_message, stream

def information(p):
    return -np.log2(p)

def entropy(dist):
    return np.sum([p*information(p) for p in dist])

M = 2**10
alphabet = {
    "symbols" : ["A", "B", "C", "D"],
    "probabilities" : np.array([0.05, 0.05, 0.8, 0.1]),
}

# Small messages examples

message_len = 10
message = np.random.choice(alphabet["symbols"], message_len, p = alphabet['probabilities'])
encoded = encode_message(message, alphabet, M)
decoded = decode_message(encoded, alphabet, M, message_len)
print(f"Message {message} encoded as {encoded} decoded as {decoded}.")

# Streaming example

message_len = 1000
message = np.random.choice(alphabet["symbols"], message_len, p = alphabet['probabilities'])
result = stream(message, alphabet, M)

print(f"p(symbols) theorical entropy : {entropy(alphabet['probabilities']):.2f} bits")
print(f"p(symbols) estimated entropy : {len(result)/message_len:.2f} bits")
