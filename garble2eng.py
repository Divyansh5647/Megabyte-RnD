with open('garble_in', 'r', encoding='utf-8') as file:
    garbled_output = file.read()
# Convert back to bytes
original_bytes = garbled_output.encode('latin1')
# for i in range(len(garbled_output)):
#     print(f'\n{i} success')
# original_bytes = bytes(ord(char) for char in garbled_output)



def decode_tokens(tokens):
    # return ''.join(list(map(decode_token, tokens)))
    # tokens = bytes(tokens)
    first_valid_index = None
    last_valid_index = None
    
    # Find the first valid UTF-8 start byte
    for i in range(len(tokens)):
        if (tokens[i] & 0xC0) != 0x80:  # Not a continuation byte
            first_valid_index = i
            break
    
    # Find the last valid UTF-8 start byte
    for i in range(len(tokens) - 1, -1, -1):
        if (tokens[i] & 0xC0) != 0x80:  # Not a continuation byte
            last_valid_index = i
            break
    
    # Extract the valid range and decode
    if first_valid_index is not None and last_valid_index is not None and first_valid_index <= last_valid_index:
        valid_bytes = tokens[first_valid_index:last_valid_index]
        return valid_bytes.decode('utf-8', errors='replace')
    else:
        return None  # No valid UTF-8 range found

# Decode to get the Bengali script
decoded_string = decode_tokens(original_bytes)
print(decoded_string)
