import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

model = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

line = numpy.linspace(0, 6, 100)

print(model(6))

#r2 = r2_score(test_y, model(test_x))
#print(r2)

#plt.scatter(train_x, train_y)
#plt.plot(line, model(line))
#plt.show()
