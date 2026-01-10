import tqdm
import cv2
import os
import json
labels_path = '/media/jianbo/ioe/motdata/VT-Tiny-MOT/annotations'
out_path = '/media/jianbo/ioe/motdata/VT-Tiny-MOT/annotations/GT/'
split = ['test']


save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{cls},1\n'

for s in split: # train  val
    videos_labels_path = os.path.join(labels_path, s)
    v_anns_list = [i for i in os.listdir(videos_labels_path) if '.json' in i]

    out_dir = out_path + s
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for v_anns in v_anns_list:
        frame_id = 0 # 对于 每一个视频标注 初始化帧id信息

        v_anns_path = os.path.join(videos_labels_path, v_anns)# 视频标注 json路径

        with open(v_anns_path, 'r') as f:
            v_labels=json.load(f) # 每个视频标注anns


        categories = v_labels['categories']
        anno_ = v_labels['annotations']
        num_instancs = v_labels['num_instances']
        images = v_labels['images']
        videos =v_labels['videos']

        videos_name = []
        for ℹ, video in enumerate(videos):
            video_name = video['name']
            video_name = video_name.split('/')[0]
            videos_name.append(video_name)

        videos_name.sort()

        for video in tqdm.tqdm(videos_name):

            dir_name = os.path.join(out_dir, video)

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            filename = os.path.join(dir_name, 'gt.txt')

            print("processing:", video)

            seq_image = []
            time_count = 0
            with open(filename, 'w') as a:
                for k, image in enumerate(images):
                    file_name = image['file_name']
                    if video in file_name:
                        id = image['id']
                        frame_id = int(float(file_name.split('/')[-1][0:5]))  # 帧数

                        seq_image.append(file_name)

                        for m, anno_signel in enumerate(anno_):
                            anno_image_id = anno_signel['image_id']
                            if id == anno_image_id:

                                time_count += 1

                                bbox = anno_signel['bbox']
                                cls_id = anno_signel['category_id']
                                track_id = anno_signel['track_id']

                                line = save_format.format(frame=frame_id, id=track_id, x1=bbox[0], y1=bbox[1], w=bbox[2], h=bbox[3], s=1,
                                                          cls=cls_id)
                                a.write(line)
                print("time_count:", time_count)
                print('seq:', len(seq_image))
