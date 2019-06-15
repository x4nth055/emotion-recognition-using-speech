import glob
import os
import subprocess

def convert_audio(audio_path, target_path, remove=True):
    """This function sets the audio `audio_path` to:
        - 16000Hz Sampling rate
        - one number of audio channels ( mono )
            Params:
                audio_path (str): the path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system."""

    # os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")
    NULL = subprocess.DEVNULL
    subprocess.Popen(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}", shell=True, stdout=NULL, stderr=NULL).communicate()
    if remove:
        os.remove(audio_path)



print(glob.glob("*.wav"))
for file in glob.glob("*.wav"):
    if " " in file:
        new_file = file.replace(" ", "")
        os.rename(file, new_file)
        file = new_file
    print(file)
    if "1" in file:
        filename, ext = file.split(".")
        filename += f"_sad.{ext}"
        filename = ''.join([ c for c in filename if not c.isdigit() ])
        print(f"Converting {file} to {filename}")
        convert_audio(file, filename)
    elif "2" in file:
        filename, ext = file.split(".")
        filename += f"_neutral.{ext}"
        filename = ''.join([ c for c in filename if not c.isdigit() ])
        print(f"Converting {file} to {filename}")
        convert_audio(file, filename)
    elif "3" in file:
        filename, ext = file.split(".")
        filename += f"_happy.{ext}"
        filename = ''.join([ c for c in filename if not c.isdigit() ])
        print(f"Converting {file} to {filename}")
        convert_audio(file, filename)
    
