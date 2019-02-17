
data = {}
for x in range(5591):
    data[x] = 0

file = open("data/Pfosten.txt")

for line in file:
    begin = int(line[:4])
    end = int(line[5:])

    for x in range(begin, end):
        data[x] = 1

file = open("data/Nichts.txt")

for line in file:
    begin = int(line[:4])
    end = int(line[5:])

    for x in range(begin, end):
        data[x] = 2




