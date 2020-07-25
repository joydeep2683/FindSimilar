from find_similar.config import *
from find_similar.util import *
import signal
import sys
import pandas as pd
from find_similar.ui import ShowCastInUI
from torch_models.deepfasion1.config import *
import os

def xray_video(vid_path, metadata_file, apparel_sim=APPAREL_SIM_THRES, model_path=MODEL_DUMP_FOLDER,
		model_name=DUMPED_MODEL):
	xray_obj = Xray(actor_ids, model_path, model_name)
	xray_obj.process_video(vid_path, metada_file, apparel_sim)


def xray_image(img_path, metadata_file, apparel_sim=APPAREL_SIM_THRES, model_path=MODEL_DUMP_FOLDER,
		model_name=DUMPED_MODEL):
	xray_obj = Xray(actor_ids, model_path, model_name)
	xray_obj.process_image(img_path, metada_file, apparel_sim)


def compress_video(video_path, delete_original=True):
	compression_path = video_path.replace('.avi', '.mp4')
	os.system(f"ffmpeg -y -i {video_path} -vcodec libx264 -crf 28 {compression_path}")
	if delete_original:
		os.remove(video_path)


def get_file_path(folfer_name, file_name, extension, extra_param=''):
	if not os.path.exists(MODEL_DUMP_FOLDER+f"{folder_name}/"):
		os.makedirs(MODEL_DUMP_FOLDER+f"{folder_name}/")
	file_path = MODEL_DUMP_FOLDER+f"{folder_name}/{file_name}_cw{COLOR_WEIGHT}.{extension}"
	return file_path

def main(argv):
	video = argv[0]
	video_name = video.split('/')[-1].split('.')[0]
	video_folder = video.split('/')[-2]

	metada_file = get_file_path('metadata', video_name, 'pkl')
	ui_overlay = ShowCastInUI(FACE_PATH)
	df = pd.read_pickle(metadata_file)

	if IS_IMAGE:
		save_path = get_file_path('images', video_name, 'jpg')
		ui_overlay.image_overlay(save_path, video, None, df)
	else:
		save_path = get_file_path('videos', video_name, 'avi')
		ui_overlay.video_overlay(save_path, video, None, df)
		if COMPRESS_VIDEO:
			compress_video(save_path)


if __name__ == "__main__":
	main(sys.argv[1:])














