import math
import random
import matplotlib.pyplot as plt
import numpy as np
import preprocessing
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
    mapping_info = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, \
                    'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    train_imgs, train_labs, test_imgs, test_labs = preprocessing.get_cifar(type_='image')
    mapping_info=key_value_change(mapping_info)
    mapping_str=mapping_onehot2str(train_labs , mapping_info)
