from matplotlib import pyplot as plt


def Lg(data, testdata):
    predict = 0
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]
    if testdata in data_x:
        # print "testdata is already known"
        return data_y[data_x.index(testdata)]
    for i in range(len(data_x)):
        af = 1
        for j in range(len(data_x)):
            if j != i:
                af *= 1.0 * (testdata - data_x[j]) / (data_x[i] - data_x[j])
        predict += data_y[i] * af
    return predict


def plot(data, nums):
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]
    Area = [min(data_x), max(data_x)]
    X = [Area[0] + 1.0 * i * (Area[1] - Area[0]) / nums for i in range(nums)]
    X[len(X) - 1] = Area[1]
    Y = [Lg(data, x) for x in X]
    plt.plot(X, Y, label='result')
    for i in range(len(data_x)):
        plt.plot(data_x[i], data_y[i], 'ro', label="point")
    plt.savefig('Lg.jpg')
    plt.show()


data = [[0, 0], [1, 2], [2, 3], [3, 8], [4, 2]]
print(Lg(data, 1.5))
plot(data, 100)
