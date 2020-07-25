import numpy as np
import pandas as pd
from sklearn import metrics
import cv2

class Xray(object):
    
    def __init__(self, actor_id_list, model_path, model_name):
        self.frame = None
        self.scene_num = 0
        self.frame_num = 0
        self.fps = None
        self.last_scene_change_frame_num = 0
        self.face_enc_df = get_face_enc_by_actorids(actor_id_list)
        self.final_df = pd.DataFrame()


    def create_object_detection_df(self, dataframe):
        dataframe['bbox'] = dataframe.apply(lambda x: get_bboxes(x['frame']), axis=1)
        dataframe = create_duplicate_row(dataframe, 'bbox')
        dataframe['obj_img'] = dataframe.apply(lambda x: get_object_image_by_bbox(x['frame'], x['bbox']))
        return dataframe

    def create_face_recognition_df(self, dataframe):
        actor_df = self.face_enc_df.copy()
        dataframe['f_location'] = dataframe.apply(lambda x: do_face_detection(FACE_DETECTION_MODEL,
                                x['obj_img']))
        actor_df['encoding'] = actor_df.apply(lambda x: np.array(x['encoding']).reshape(1,-1), axis=1)
        dataframe['face_encoding'] = dataframe.apply(lambda x: face_encodings(x['obj_img'],
                x['f_location'][0])[0].reshape(1,-1) if x['f_location'] else None, axis=1)
        dataframe['obj_img'] = dataframe.apply(lambda x: is_full_body_image(x['bbox'], 
                                x['f_location']), axis=1)
        dataframe['actor_ids'] = dataframe.apply(lambda x: get_actor_ids(actor_df, x['face_encoding']))
        return dataframe

    def get_actor_ids(self, face_encoding_df, query_encoding):
        face_encoding_df['distace'] = face_encoding_df.apply(lambda x: 
                metrics.pairwise.euclidean_distance(x['encoding'], query_encoding), axis=1)
        face_encoding_df = face_encoding_df.sort_values(by='distance')
        if face_encoding_df.iloc[0]['distance'] <= [[FACE_REC_THRESG]]:
            return face_encoding_df.iloc[0]['name']

    
    def create_apparel_extraction_df(self, dataframe):
        dataframe['top_n'] = dataframe.apply(lambda x: deep_fashion_retrival(x['obj_img']))
        dataframe['obj_img'] = dataframe.apply(lambda x: x['obj_img'] if x['top_n'] is not None 
                    else None, axis=1)
        return dataframe

    def process_df(self, dataframe, apparel_sim, params):
        if params['obj_det']:
            dataframe = self.create_object_detection_df(dataframe)
        if params['face_rec']:
            dataframe = self.create_face_recognition_df(dataframe)
        if params['aprl_ext']:
            dataframe = self.create_apparel_extraction_df(dataframe)
        self.final_df = self.final_df.append(dataframe, ignore_index=True)

    def process_video(self, path, out_csv_path, apparel_sim=0.5, params=
            {
                "obj_det": OBJECT_DETECTION,
                "face_rec": FACE_REC,
                "aprl_ext": APPRL_EXT
            }):
        cap = cv2.VideoCapture(path)
        obj = {
            "scene_num":[],
            "frame": [],
            "frame_num": []
        }
        while True:
            ret, frame = cap.read()
            if self.frame_num > 0:
                prev_frame = obj['frame'][-1]
            else:
                prev_frame = frame
            if ret:
                if self.is_scene_changes(frame, prev_frame):
                    scene_df = pd.DataFrame(obj)
                    self.process_df(scene_df, apparel_sim, params)
                    self.scene_num +=1
                    self.last_scene_change_frame_num = self.frame_num
                    obj = {
                        "scene_num":[self.scene_num],
                        "frame": [self.frame],
                        "frame_num": [self.frame_num]
                    }
                else:
                    obj['scene_num'].append(self.scene_num)
                    obj['frame'].append(self.frame)
                    obj['frame_num'].append(self.frame_num)
            else:
                break
        scene_df = pd.DataFrame(obj)
        self.process_df(scene_df, apparel_sim, params)
        self.final_df.to_pickle(out_csv_path)
