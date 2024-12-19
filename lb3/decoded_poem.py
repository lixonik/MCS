import struct
from decimal import Decimal, getcontext

getcontext().prec = 10_000

def load_encoded_file(filepath):
    with open(filepath, 'rb') as f:
        text_length = struct.unpack('>I', f.read(4))[0]
        num_symbols = struct.unpack('>I', f.read(4))[0]

        assert text_length > 0, "Invalid text length!"

        cumulative_probs = {}
        for i in range(num_symbols):
            char_len = struct.unpack('>B', f.read(1))[0]
            char = f.read(char_len).decode('utf-8')
            low = struct.unpack('>d', f.read(8))[0]
            high = struct.unpack('>d', f.read(8))[0]
            cumulative_probs[char] = (low, high)

        encoded_value = struct.unpack('>d', f.read(8))[0]
    return text_length, cumulative_probs, encoded_value



def debug_file(filepath):
    with open(filepath, 'rb') as f:
        content = f.read()
        print("File content (bytes):", content)


def arithmetic_decode(text_length, cumulative_probs, encoded_value):
    decoded_text = []
    encoded_value = Decimal(encoded_value)
    # print("Decoding Process:")
    for step in range(text_length):
        for char, (low, high) in cumulative_probs.items():
            low, high = Decimal(low), Decimal(high)
            if low <= encoded_value < high:
                decoded_text.append(char)
                range_ = high - low
                encoded_value = (encoded_value - low) / range_

                # Debugging output for each decoding step
                # print(f"Step {step + 1}: Decoded '{char}', Low = {low}, High = {high}, Encoded Value = {encoded_value}")
                break
        else:
            print(f"Decoding error at step {step + 1}: Encoded Value = {encoded_value}")
            raise ValueError("Decoding failed: No matching character found.")

    return ''.join(decoded_text)

def debug_probabilities(cumulative_probs, title):
    print(f"{title}:")
    for char, (low, high) in sorted(cumulative_probs.items()):
        print(f"  Char: {repr(char)} Low: {low}, High: {high}")


def main():
    input_file = './poem.var3'
    output_file = './decoded_poem.txt'

    text_length, cumulative_probs, encoded_value = load_encoded_file(input_file)
    debug_probabilities(cumulative_probs, "Cumulative Probabilities (Decoding)")

    decoded_text = arithmetic_decode(text_length, cumulative_probs, encoded_value)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(decoded_text)
        
    # debug_file('./poem.var3')

    print(f"Decoded text saved to {output_file}")

if __name__ == '__main__':
    main()
