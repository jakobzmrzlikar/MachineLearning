# for i in range(398):
#     line = input().split()
#     line = line[:7]
#     line.append(line.pop(0))
#     for i in range(len(line)):
#         if line[i] == "?":
#             line[i] = "100.0"
#     print (",".join(line))

# for i in range(150):
#     line = input().split(",")
#     if line[-1] == "Iris-virginica":
#         line[-1] = "2"
#     elif line[-1] == "Iris-versicolor":
#         line[-1] = "1"
#     elif line[-1] == "Iris-setosa":
#         line[-1] = "0"
#     print(",".join(line))

for i in range(1000):
    line = input().split()
    print(",".join(line))
