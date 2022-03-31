# import os
# import data_utils as util
# from Splitbase import splitbase
#
#
# if __name__ == '__main__':
#     import sys
#     datapath = os.path.abspath(sys.argv[1])
#
#     trainDir = os.path.join(datapath, 'trainset_reclabelTxt')
#     evaDir = os.path.join(datapath, 'valset_reclabelTxt')
#     train = os.listdir(trainDir)
#     eva = os.listdir(evaDir)
#
#     train = [x.strip('.txt') for x in train]
#     eva = [x.strip('.txt') for x in eva]
#
#     sets = ['training', 'evaluation']
#
#     datalist = {'training': {}, 'evaluation': {}}
#     for set in sets:
#         if set == 'training':
#             dlist = train
#             dDir = trainDir
#         else:
#             dlist = eva
#             dDir = evaDir
#
#         for d in dlist:
#             ipath = os.path.join(datapath, 'trainData/images/{}.tif'.format(d))
#             apath = os.path.join(dDir, 'trainData/data-txt/{}.txt'.format(d))
#
#             datalist[set][d] = {}
#             datalist[set][d]['imagepath'] = ipath
#             datalist[set][d]['objects'] = util.parse_dota_poly(apath)
#
#     # example usage of ImgSplit
#     trainingsplit = splitbase(outdir=os.path.join(datapath, 'training'), ext='.tif')
#     trainingsplit.splitdata(datalist['training'], 1)
#
#     evaluationsplit = splitbase(outdir=os.path.join(datapath, 'evaluation'), ext='.tif')
#     evaluationsplit.splitdata(datalist['evaluation'], 1)
import os
import data_utils as util
import random
from Splitbase import splitbase


if __name__ == '__main__':
    import sys
    datapath = os.path.abspath(sys.argv[1])
    # datapath = os.path.abspath('../data')
    # pfiles = os.listdir(os.path.join(datapath, 'trainData/Images/'))
    # nfiles = os.listdir(os.path.join(datapath, 'trainData/negative image set/'))
    # anno = os.listdir(os.path.join(datapath, 'trainData/data-txt'))
    trainlistfile = os.path.join(datapath, 'plane/train/train_list.txt')
    evalistfile = os.path.join(datapath, 'plane/val/val_list.txt')
    # testlistfile = os.path.join(datapath, 'Main/test.txt')
    with open(trainlistfile, 'r') as f:
        train = f.readlines()
    with open(evalistfile, 'r') as f:
        eva = f.readlines()
    # with open(testlistfile, 'r') as f:
    #     eva = f.readlines()

    train = [x.strip() for x in train]
    eva = [x.strip() for x in eva]



    # ptrain = random.sample(pfiles, len(pfiles) // 3 * 2)
    # peva = ['p' + x.strip('.tif') for x in pfiles if x not in ptrain]
    # ptrain = ['p' + x.strip('.tif') for x in ptrain]

    # ntrain = random.sample(nfiles, len(nfiles) // 3 * 2)
    # neva = ['n' + x.strip('.jpg') for x in nfiles if x not in ntrain]
    # ntrain = ['n' + x.strip('.jpg') for x in ntrain]
    #
    # train = []
    # eva = []
    #
    # train.extend(ptrain)
    # # train.extend(ntrain)
    # eva.extend(peva)
    # # eva.extend(neva)

    sets = ['training', 'evaluation']

    datalist = {'training': {}, 'evaluation': {}}
    for set in sets:
        if set == 'training':
            dlist = train
            imageFile = os.path.join(datapath, 'plane/train/JPEGImages/{}.tif')
        else:
            dlist = eva
            imageFile = os.path.join(datapath, 'plane/val/JPEGImages/{}.tif')

        for d in dlist:
            ipath = imageFile.format(d)
            apath = os.path.join(datapath, 'plane/data-txt/{}.txt'.format(d))

            datalist[set][d] = {}
            datalist[set][d]['imagepath'] = ipath
            datalist[set][d]['objects'] = util.parse_dota_poly(apath)
    # i = 0
    # outfilename = '{:0>3d}'
    #
    # datalist = {'training': {}, 'evaluation': {}}
    # for set in sets:
    #     if set == 'training':
    #         dlist = train
    #     else:
    #         dlist = eva
    #
    #     while len(dlist) != 0:
    #         file = random.sample(dlist, 1)[0]
    #         dlist.remove(file)
    #         if file.startswith('p'):
    #             ipath = os.path.join(datapath, 'trainData/Images/{}.tif'.format(file.strip('p')))
    #             apath = os.path.join(datapath, 'trainData/data-txt/{}.txt'.format(file.strip('p')))
    #         # else:
    #         #     ipath = os.path.join(datapath, 'trainData/negative image set/{}.jpg'.format(file.strip('n')))
    #         #     apath = ''
    #
    #         datalist[set][outfilename.format(i)] = {}
    #         datalist[set][outfilename.format(i)]['imagepath'] = ipath
    #         datalist[set][outfilename.format(i)]['objects'] = util.parse_dota_poly(apath)
    #         i += 1

    # example usage of ImgSplit
    trainingsplit = splitbase(outdir=os.path.join(datapath, 'training'), ext='.tif')
    trainingsplit.splitdata(datalist['training'], 1)

    evaluationsplit = splitbase(outdir=os.path.join(datapath, 'evaluation'), ext='.tif')
    evaluationsplit.splitdata(datalist['evaluation'], 1)