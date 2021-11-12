import csv
file = open('message.txt', 'r')
count = 0
l = []
for line in file:
    if line[0] == '[':
        line = line.replace('[', '')
        line = line.replace(']', '')
        print(line)
        l.append(line)
        count += 1
with open('quadrant.csv', 'w', newline='') as csvfile:
    wrt = csv.writer(csvfile, delimiter = ' ')