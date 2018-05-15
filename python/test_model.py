import caffe

if __name__ == "__main__":
    #文件的存放路径
    root = '/home/user/Projects/data/caffe_dataset_cn_sim/'
    caffe.set_mode_cpu
    net = caffe.Net('/home/user/Projects/deepLearning_OCR/lenet_train_test.prototxt',root+'lenet_iter_50000.caffemodel',caffe.TEST)
    conv1_w = net.params['conv11'][0].data
    conv1_b = net.params['conv11'][1].data
    print(conv1_w,conv1_b)
    print(conv1_w.size,conv1_b.size)