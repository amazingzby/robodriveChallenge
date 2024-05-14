import numpy as np
import pickle

#200,200,16 -> 200,200,16,17 one hot
def one_hot(data):
    y_one_hot = np.zeros((1,200,200,16,17),float)
    for i in range(200):
        for j in range(200):
            for k in range(16):
                y_one_hot[0,i,j,k,data[0,i,j,k]] = 1
    return y_one_hot



def submitFusion(pkl_list,pkl_weights):
    pklData = []
    tokens = []
    for idx in range(len(pkl_list)):
        with open(pkl_list[idx],"rb") as f:
            curData = pickle.load(f)
            pklData.append(curData)
            if idx == 0:
                sampleKeys = list(curData.keys())
                for key in sampleKeys:
                    tokens.append([key,list(curData[key].keys())])
    submitData = dict()
    count = 0
    for token_list in tokens:
        submitData[token_list[0]] = dict()
        for tmpToken in token_list[1]:
            #(1, 200, 200, 16),uint8
            tmpData = np.zeros((1,200,200,16,17),dtype=float)
            for idx in range(len(pklData)):
                elemData = pklData[idx][token_list[0]][tmpToken]
                elemData = one_hot(elemData) * pkl_weights[idx]
                tmpData = tmpData + elemData
            curData = np.argmax(tmpData,axis=4).astype(np.uint8)
            submitData[token_list[0]][tmpToken] = curData
            count += 1
            if count % 10 == 0:
                print("%d samples has been processed!"%count)
    with open("submits/pred.pkl","wb") as f:
        pickle.dump(submitData,f)
    return

#fusion v2
#pkl_list = ["submits/occ3d_v1/pred.pkl",
#                "submits/resnet_v3_epoch6/pred.pkl",
#                "submits/vovnet-occ/pred.pkl",
#                "submits/baseline/pred.pkl",
#                "submits/resnet_v1/pred.pkl"]

#fusion v3
#pkl_list = ["submits/occ3d_v1/pred.pkl",
#                "submits/baseline/pred.pkl",
#                "submits/resnet_v1/pred.pkl"]

#fusion_v4
#pkl_list = ["submits/occ3d_v1/pred.pkl",
#                "submits/vovnet-occ/pred.pkl",
#                "submits/resnet_v1/pred.pkl"]

if __name__ == "__main__":
    pkl_list = ["submits/occ3d_v1/pred.pkl",
                "submits/resnet_v3_epoch6/pred.pkl",
                "submits/vovnet-occ/pred.pkl",
                "submits/baseline/pred.pkl",
                "submits/resnet_v1/pred.pkl"]
    pkl_weights = [1.1,
                   1.05,
                   1.0,
                   1.07,
                   1.06]
    submitFusion(pkl_list,pkl_weights)
