from train_test import train

if __name__ == "__main__":    
    data_folder = 'ROSMAP'
    testonly = False    # True
    modelpath = '../DPNET-main/'
    train(data_folder, modelpath, testonly)