from collections import Counter
import math
import struct
from decimal import Decimal, getcontext
getcontext().prec = 10_000

def calculate_probabilities(text):
    total_chars = len(text)
    frequencies = Counter(text)
    probabilities = {char: freq / total_chars for char, freq in frequencies.items()}
    sorted_probs = sorted(probabilities.items(), key=lambda x: -x[1])

    cumulative = 0
    cumulative_probs = {}
    for char, prob in sorted_probs:
        cumulative_probs[char] = (cumulative, cumulative + prob)
        cumulative += prob

    print("Cumulative Probabilities (Encoding):")
    for char, (low, high) in cumulative_probs.items():
        print(f"Char: {char}, Low: {low}, High: {high}")

    return cumulative_probs, probabilities


def shannon_entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities.values())



def arithmetic_encode(text, cumulative_probs):
    low, high = Decimal(0), Decimal(1)
    for char in text:
        range_ = high - low
        char_low, char_high = map(Decimal, cumulative_probs[char])
        high = low + range_ * char_high
        low = low + range_ * char_low
    return (low + high) / 2

def validate_cumulative_probs(cumulative_probs):
    prev_high = Decimal(0)
    for char, (low, high) in sorted(cumulative_probs.items(), key=lambda x: x[1][0]):
        low, high = Decimal(low), Decimal(high)
        assert prev_high <= low, f"Overlap detected for char '{char}'"
        prev_high = high
    assert abs(prev_high - 1) < Decimal("1e-10"), "Probabilities do not sum to 1!"



def save_encoded_file(filepath, cumulative_probs, encoded_value, text_length):
    with open(filepath, 'wb') as f:
        # number of characters in the original text
        f.write(text_length)

        # number of symbols
        f.write(len(cumulative_probs))

        # cumulative probabilities
        for char, (low, high) in cumulative_probs.items():
            char_bytes = char.encode('utf-8')
            f.write(len(char_bytes))  # Length of character bytes
            f.write(char_bytes)  # Actual bytes of the character
            f.write(low)  # Lower bound
            f.write(high))  # Upper bound

        # encoded value
        f.write(struct.pack('>d', encoded_value))

def debug_probabilities(cumulative_probs, title):
    print(f"{title}:")
    for char, (low, high) in sorted(cumulative_probs.items()):
        print(f"  Char: {repr(char)} Low: {low}, High: {high}")


def main():
    input_file = './poem.txt'
    output_file = './poem.var3'

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    cumulative_probs, probabilities = calculate_probabilities(text)

    validate_cumulative_probs(cumulative_probs)

    entropy = shannon_entropy(probabilities)

    encoded_value = arithmetic_encode(text, cumulative_probs)

    save_encoded_file(output_file, cumulative_probs, encoded_value, len(text))

    original_length = len(text) * 8  # Each character is 1 byte = 8 bits
    compressed_length = math.ceil(-math.log2(encoded_value - 0) if encoded_value != 0 else 1)  # Estimate

    # Compression ratio
    compression_ratio = original_length / compressed_length

    debug_probabilities(cumulative_probs, "Cumulative Probabilities (Encoding)")
    print(f"Shannon Entropy: {entropy:.4f}")
    print(f"Original Length: {original_length} bits")
    print(f"Compressed Length: {compressed_length} bits")
    print(f"Compression Ratio: {compression_ratio:.4f}")

if __name__ == '__main__':
    main()
