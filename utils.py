import math
import random
import matplotlib.pyplot as plt
import numpy as np
import preprocessing
import pickle
import os
def plot_xs_ys(title,xs_title, ys_title , folder_path,xs ,*arg_ys ):
    plt.xlabel(xs_title)
    plt.ylabel(ys_title)
    plt.title(title)
    for ys in arg_ys:
        ys=list(ys)
        plt.plot(xs, ys)
        #folder_path = './graph/' + file_path.split('/')[-1].split('.')[0]
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    plt.savefig(folder_path +'/'+title)
    plt.close()
def draw_graph( log_folder_path ,save_folder , step_list, model_name):

    f=open(log_folder_path + 'test_acc')
    test_acc=pickle.load(f)
    f = open(log_folder_path +  'test_cost')
    test_cost = pickle.load(f)
    f = open(log_folder_path + 'train_acc')
    train_acc = pickle.load(f)
    f = open(log_folder_path + 'train_cost')
    train_cost = pickle.load(f)

    plot_xs_ys(model_name + ' Train Accuracy','Step','Train Accuracy',save_folder,step_list , train_acc)
    plot_xs_ys(model_name + ' Train Loss', 'Step', 'Train Loss', save_folder,step_list, train_cost )
    plot_xs_ys(model_name + ' Validation Accuracy', 'Step', 'Validation Accuracy', save_folder,step_list, test_acc)
    plot_xs_ys(model_name + ' Validation Loss', 'Step', 'Validation Loss', save_folder,step_list, test_cost)
    plot_xs_ys(model_name + ' Train_Validation Accuracy','Step','Train_Validation Accuracy ',save_folder,step_list, train_acc, test_acc)
    plot_xs_ys(model_name + ' Trrain_Validation Loss','Step','Train_Validation Loss ',save_folder,step_list, train_cost, test_cost)





def plot_images(imgs , names=None):
    h=math.ceil(math.sqrt(len(imgs)))
    fig=plt.figure(figsize=(20,20))

    for i in range(len(imgs)):
        ax=fig.add_subplot(h,h,i+1)
        ind=random.randint(0,len(imgs)-1)
        img=imgs[ind]
        plt.imshow(img)
        if not names is None:
            ax.set_xlabel(names[ind])
    plt.savefig('./1.png')
    plt.show()

def key_value_change(mapping_info):

    keys=mapping_info.keys()
    values= mapping_info.values()
    assert len(set(keys))==len(keys)
    assert len(set(values)) == len(values)
    print 'before '
    print 'keys:',keys
    print 'values:',values

    new_dic={}
    for i in range(len(keys)):
        new_dic[values[i]]=keys[i]
    print 'after'
    print 'keys:',new_dic.keys()
    print 'values:',new_dic.values()
    return new_dic


def mapping_onehot2str(labels , mapping_info):
    mappind_str=[]
    assert np.ndim(labels) ==2
    cls=np.argmax(labels , axis=1)
    print cls
    for i in cls:
        mappind_str.append(mapping_info[i])
    return mappind_str


if __name__ == '__main__':
    """
    mapping_info = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, \
                    'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    train_imgs, train_labs, test_imgs, test_labs = preprocessing.get_cifar(type_='image')
    mapping_info=key_value_change(mapping_info)
    mapping_str=mapping_onehot2str(train_labs , mapping_info)
    """

    batch_iteration = 100
    training_epochs = 2000

    n = training_epochs * batch_iteration
    xs = range(0, n, batch_iteration)
    print xs
    #utils.draw_graph(folder_path='./cost_acc' =, show_graph = True, xs)
    draw_graph('./cost_acc/3conv' , './cost_acc/3conv/graph' , step_list=xs )