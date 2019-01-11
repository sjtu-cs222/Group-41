import numpy as np

def count_one(a):
    count = 0
    for i in range(len(a)):
        if int(a[i]) == 1:
            count += 1
    return count

def include_plus(a, b):
    count = 0
    for i in range(len(a)):
        if int(a[i]) == 0 and int(b[i]) == 1:
            count += 1
    return count

def difference(a, b):
    return count_one(a) - count_one(b)


def to_str(a):
    stri = ''
    for i in range(len(a)):
        stri += str(a[i])
    return stri

def include(a, b):
    c = ''
    for i in range(len(a)):
        if int(a[i]) == 0 and int(b[i]) == 1:
            c += '1'
        else:
            c += str(a[i])
    return c

if __name__ == "__main__":
    vector_set = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
          0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    )
    order = vector_set.tolist()
    print(order)
    for i in range(len(order)):
        order[i] = [count_one(order[i]), to_str(order[i])]
    order.sort(key=lambda x: x[0])
    '''print(order)
    adjacent = np.zeros((len(order), len(order)))
    adjacent += len(order[0][1])
    print(adjacent)
    print(count_one(order[0][1]))
    print(len(order))'''
    '''for j in range(len(order)):
        for i in range(j+1, len(order)):
            adjacent[j][i] = include_plus(order[i][1], order[j][1]) + difference(order[i][1], order[j][1])'''
    dic = {}
    '''for j in range(0, len(order)):
        for i in range(0, j):
            if adjacent[i][j] == 0:
                try:
                    dic[order[i][1]].append(order[j][1])
                except KeyError as e:
                    dic[order[i][1]] = [order[j][1]]'''
    for i in range(0, len(order)):
        try:
            dic[order[i][1]].extend([order[i][1]])
        except KeyError as e:
            dic[order[i][1]] = [order[i][1]]
    adj = np.zeros((len(order), len(order)))
    adj += np.size(order[0][1])
    while len(dic) > 3:
        mini = 2*len(order[0][1])
        mini_size = 2*len(order[0][1])
        for pair1 in dic:
            for pair2 in dic:
                if pair1 != pair2:
                    if count_one(pair1) >= count_one(pair2):
                        if include_plus(pair1, pair2) + difference(pair1, pair2) < mini:
                            mini = include_plus(pair1, pair2) + difference(pair1, pair2)
                            mini_size = count_one(pair1)
                        elif include_plus(pair1, pair2) + difference(pair1, pair2) == mini:
                            if count_one(pair1) < mini_size:
                                mini_size = count_one(pair1)
                    else:
                        if include_plus(pair2, pair1) + difference(pair2, pair1) < mini:
                            mini = include_plus(pair2, pair1) + difference(pair2, pair1)
                            mini_size = count_one(pair2)
                        elif include_plus(pair2, pair1) + difference(pair2, pair1) == mini:
                            if count_one(pair2) < mini_size:
                                mini_size = count_one(pair2)
        print(mini)
        change = True
        while change:
            if len(dic) <= 3:
                break
            change = False
            flag = False
            for pair1 in dic:
                for pair2 in dic:
                    if pair1 != pair2:
                        if count_one(pair1) >= count_one(pair2):
                            if include_plus(pair1, pair2) + difference(pair1, pair2) == mini and count_one(pair1) == mini_size:
                                if include(pair1, pair2) == pair1:
                                    dic[pair1].extend(dic[pair2])
                                else:
                                    try:
                                        dic[include(pair1, pair2)].extend(dic[pair1] + dic[pair2])
                                    except KeyError as e:
                                        dic[include(pair1, pair2)] = dic[pair1] + dic[pair2]
                                    del dic[pair1]
                                del dic[pair2]
                                flag = True
                                change = True
                                break
                        else:
                            if include_plus(pair2, pair1) + difference(pair2, pair1) == mini and count_one(pair2) == mini_size:
                                if include(pair2, pair1) == pair2:
                                    dic[pair2].extend(dic[pair1])
                                else:
                                    try:
                                        dic[include(pair2, pair1)].extend(dic[pair1] + dic[pair2])
                                    except KeyError as e:
                                        dic[include(pair2, pair1)] = dic[pair1] + dic[pair2]
                                    del dic[pair2]
                                del dic[pair1]
                                flag = True
                                change = True
                                break
                if flag:
                    break
    print(dic)