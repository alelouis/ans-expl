import numpy as np

def cumul_inverse(slot, cdf_freq): 
    for i, v in enumerate(cdf_freq): 
        if slot < v: return i-1
    
def decode_symbol(state, alphabet, M):
    probs, symbols = alphabet["probabilities"], alphabet["symbols"]
    frequencies = (probs * M).astype(int)
    cdf_freq = np.insert(np.cumsum(frequencies),0,0)
    slot = state % M
    symbol_idx = cumul_inverse(slot, cdf_freq)
    symbol = symbols[symbol_idx]
    prev_state = (state//M)*frequencies[symbol_idx] + slot - cdf_freq[symbol_idx]
    return symbol, prev_state

def encode_symbol(symbol, state, alphabet, M):
    probs, symbols = alphabet["probabilities"], alphabet["symbols"]
    frequencies = (probs * M).astype(int)
    cdf_freq = np.insert(np.cumsum(frequencies), 0, 0)
    symbol_frequency = frequencies[symbols.index(symbol)]
    next_state = (state//symbol_frequency) * M + cdf_freq[symbols.index(symbol)] + (state % symbol_frequency) 
    return next_state

def encode_message(message, alphabet, M):
    state = 0
    for symbol in message:
        state = encode_symbol(symbol, state, alphabet, M)
    return state

def decode_message(state, alphabet, M, message_len):
    message = []
    for _ in range(message_len):
        symbol, state = decode_symbol(state, alphabet, M)
        message.append(symbol)
    return message[::-1]

def stream(message, alphabet, M):
    buffer_size = 30
    result = ""
    for i in range(len(message)//buffer_size):
        buffer = message[i*buffer_size:(i+1)*buffer_size]
        encoded = encode_message(buffer, alphabet, M)
        result += bin(encoded)[2:]
    return result
