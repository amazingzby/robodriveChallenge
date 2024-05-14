import pickle
import numpy as np

if __name__ == "__main__":
    filePath1 = "/home/zby/data/robodrive-phase2/sample_scenes.pkl"
    #filePath2 = "/home/zby/data/nuscenes/nuscenes_infos_temporal_val.pkl"
    filePath2 = "/home/zby/work/project/robodrive-24/toolkit/track-3/SurroundOcc/data/robodrive_infos_temporal_test.pkl"
    f1 = open(filePath1,"rb")
    f2 = open(filePath2,"rb")
    data1 = pickle.load(f1) #phase2 data
    data2 = pickle.load(f2) #nuscenes val数据

    scene_dict = {}
    for i in range(len(data2['infos'])):
        scene_token = data2['infos'][i]['scene_token']
        if scene_token not in scene_dict:
            scene_dict[scene_token] = []
        sample = data2['infos'][i]
        scene_dict[scene_token].append(sample)
    evalFilePath = "robodrive-phase2.pkl"
    eval_dict = dict()
    eval_dict['metadata'] = data2['metadata']
    eval_dict['infos'] = []
    for key in data1.keys():
        fileName = "pkls/"+"robodrive-phase2"+"-"+key+".pkl"
        scenes_keys = data1[key]
        saved_dict = dict()
        saved_dict['metadata'] = data2['metadata']
        saved_dict['infos'] = []
        for token in scenes_keys:
            samples = scene_dict[token]
            for sample in samples:
                sample = sample.copy()
                for cam_key in sample['cams'].keys():
                    sample['cams'][cam_key]['data_path'] = 'data/robodrive-phase2/'+key+'/samples/'+cam_key+"/"+sample['cams'][cam_key]['data_path'].split("/")[-1]
                saved_dict['infos'].append(sample) #先保存每个场景的样本
                sample['token'] = key+"-"+sample['token']
                if len(sample['prev']) > 0:
                    sample['prev'] = key+"-"+sample['prev']
                if len(sample['next']) > 0:
                    sample['next'] = key + "-" + sample['next']
                eval_dict['infos'].append(sample)#保存为验证集样本
        print(len(saved_dict['infos']))
        with open(fileName, 'wb') as f:
            pickle.dump(saved_dict,f)
    print("Eval dataset number:%d"%len(eval_dict['infos']))
    with open(evalFilePath,"wb") as f:
        pickle.dump(eval_dict,f)

