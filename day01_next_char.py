text = "hello"

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}

x = text[:-1]
y = text[1:]
x_ids = [stoi[ch] for ch in x]
y_ids = [stoi[ch] for ch in y]

print("text:", text)
print("chars:", chars)
print("stoi:", stoi)
print("x:", x)
print("y:", y)
print("x_ids:", x_ids)
print("y_ids:", y_ids)
print()
print("training pairs:")

for current_char, next_char in zip(x, y):
    print(current_char, "->", next_char)

print()
print("numeric training pairs:")

for current_id, next_id in zip(x_ids, y_ids):
    print(current_id, "->", next_id)
