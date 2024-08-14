import os
import pandas as pd
import argparse
from pathlib import Path
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def video_process(video_file_path, dst_dir_path, name, fps=5, size=240):
    
    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                   '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                   'stream=width,height,avg_frame_rate,duration').split()
    
    ffprobe_cmd.append(str(video_file_path))

    p = subprocess.run(ffprobe_cmd, capture_output=True)
    res = p.stdout.decode('utf-8').splitlines()
    if len(res) < 4:
        return 0, 1

    frame_rate = [float(r) for r in res[2].split('/')]
    frame_rate = frame_rate[0] / frame_rate[1]
 

    width = int(res[0])
    height = int(res[1])

    if width > height:
        vf_param = 'scale=-1:{}'.format(size)
    else:
        vf_param = 'scale={}:-1'.format(size)

    if fps > 0:
        vf_param += ',minterpolate={}'.format(fps)

    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
    ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
    print(ffmpeg_cmd)
    out_ffmpeg = subprocess.run(ffmpeg_cmd)
    extracted_frames = len(os.listdir(dst_dir_path))
    return extracted_frames, out_ffmpeg.returncode


def process_row(args):
    _, row = args
    current_subcategory = row["label"]
    current_id = row["youtube_id"]
    current_start = row["time_start"]
    current_end = row["time_end"]
    current_split = row["split"]
    
    video_name = '%s_%s_%s' % (current_id, '%06d'% current_start, '%06d'% current_end)
    current_filename = "id_"+video_name
    
    current_category = None
    for k,v in categories_dict.items():
        if current_subcategory in v:
            current_category = k
 
    current_filename_outpath = os.path.join(video_dst_path, current_filename) 
    if not os.path.exists(current_filename_outpath) and not os.path.exists(os.path.join(current_filename_outpath, "category.csv")):
        os.mkdir(os.path.join(video_dst_path, current_filename))
        
        video_file_path = os.path.join(original_path, current_subcategory, video_name + ".mp4")
        
        current_frame_dir = os.path.join(video_dst_path, current_filename, "jpgs")
        if not os.path.exists(current_frame_dir):
            os.mkdir(current_frame_dir)
            
        extracted_frames, success = video_process(video_file_path,  current_frame_dir, current_filename)
        category_df = pd.DataFrame([[current_category, current_subcategory, current_filename]], columns=["Category", "Sub-behavior", "Filename"])
        category_df.to_csv(os.path.join(video_dst_path, current_filename, "category.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'kinetics_downloaded_path', default=None, type=Path, help='Directory path of videos to move.')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Destination path for Kinetics videos. Inside, it has to be already available the Info directory with \
            train.csv, test.csv and validation.csv files.')
    
    args = parser.parse_args()

    src_path = args.kinetics_downloaded_path
    dst_path = args.dst_path

    video_dst_path = os.path.join(dst_path,'Videos')
    if not os.path.exists(video_dst_path):
       os.mkdir(video_dst_path)


    split = 'train'
    k700_path = os.path.join(src_path,'k700-2020')
    original_path = os.path.join(k700_path,split)
    
    categories_dict = {
                'food': [
                    'eating burger', 'eating cake', 'eating carrots', 'eating chips', 'eating doughnuts',
                    'eating hotdog', 'eating ice cream', 'eating spaghetti', 'eating watermelon',
                    'sucking lolly', 'tasting beer', 'tasting food', 'tasting wine', 'sipping cup'
                ],
                'phone': [
                    'texting', 'talking on cell phone', 'looking at phone'
                ],
                'smoking': [
                    'smoking', 'smoking hookah', 'smoking pipe'
                ],
                'fatigue': [
                    'sleeping', 'yawning', 'headbanging', 'headbutting', 'shaking head'
                ],
                'selfcare': [
                    'scrubbing face', 'putting in contact lenses', 'putting on eyeliner', 'putting on foundation',
                    'putting on lipstick', 'putting on mascara', 'brushing hair', 'brushing teeth', 'braiding hair',
                    'combing hair', 'dyeing eyebrows', 'dyeing hair'
                ]
    }
    
    

    info_path = os.path.join(dst_path,'Info')

    # training set

    train_csv_path = os.path.join(info_path,'train.csv')

    train_csv = pd.read_csv(train_csv_path)

    with Pool(processes=cpu_count()) as pool:  # Adjust the number of processes as needed
        list(tqdm(pool.imap(process_row, train_csv.iterrows()), total=len(train_csv)))

    # validation set

    validation_csv_path = os.path.join(info_path,'validation.csv')

    validation_csv = pd.read_csv(validation_csv_path)

    with Pool(processes=cpu_count()) as pool:  # Adjust the number of processes as needed
        list(tqdm(pool.imap(process_row, validation_csv.iterrows()), total=len(validation_csv)))


    # test set

    test_csv_path = os.path.join(info_path,'test.csv')

    test_csv = pd.read_csv(test_csv_path)

    with Pool(processes=cpu_count()) as pool:  # Adjust the number of processes as needed
        list(tqdm(pool.imap(process_row, test_csv.iterrows()), total=len(test_csv)))