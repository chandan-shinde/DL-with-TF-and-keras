import numpy as np

def mnistExtract(trainName, testName):
    with open(trainName,'r') as f:
        train = f.read()
        train = train.split('\n')
    with open(testName,'r') as f:
        test = f.read()
        test = test.split('\n')

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for line in train:
        if line != '':
            x_train.append(np.array(line.split(',')[1:], dtype=np.float32))
            y_train.append(np.array(line.split(',')[0]))

    for line in test:
        if line != '':
      
            x_test.append(np.array(line.split(',')[1:], dtype=np.float32))
            y_test.append(np.array(line.split(',')[0]))

    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    del train, test
    return x_train, x_test, y_train, y_test
    
def onehot(x):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse=False, categories='auto')
    return enc.fit_transform(x.reshape(len(x),1))