text = "hello transformer"

chars = sorted(list(set(text)))
print("chars:", chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

print("stoi:", stoi)
print("itos:", itos)


def encode(s):
    ids = []
    for ch in s:
        ids.append(stoi[ch])
    return ids


def decode(ids):
    chars_out = []
    for i in ids:
        chars_out.append(itos[i])
    return "".join(chars_out)


encoded = encode("transformer")
decoded = decode(encoded)

print("encoded:", encoded)
print("decoded:", decoded)
