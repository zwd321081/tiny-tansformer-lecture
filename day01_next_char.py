text = "hello"

x = text[:-1]
y = text[1:]

print("text:", text)
print("x:", x)
print("y:", y)
print()
print("training pairs:")

for current_char, next_char in zip(x, y):
    print(current_char, "->", next_char)
