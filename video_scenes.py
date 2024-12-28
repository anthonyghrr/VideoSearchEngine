import os
import cv2
import json
import re
from rapidfuzz import fuzz
from dotenv import load_dotenv
from PIL import Image
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
import moondream as md
import google.generativeai as genai
import time

try:
    model = md.vl(model="/Users/anthonyghandour/Desktop/moondream-2b-int8.mf")
except ValueError as e:
    exit(1)

# DOESNT WORK
# load_dotenv()
# genai.configure(api_key='GOOGLE_API_KEY')
genai.configure(api_key='AIzaSyCptXoo8SdvDEzTW_d692hG3ZaM1KYBOBo')

# preprocess words
def preprocess_word(word):
    return re.sub(r'\W+', '', word.lower())

# detect scenes and save scene images
def detect_scenes_and_save_images(video_path, output_folder="scenes"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    scene_images = []
    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        print(f"Detected {len(scene_list)} scenes.")
        cap = cv2.VideoCapture(video_path)
        for i, (start, end) in enumerate(scene_list):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start.get_frames())
            ret, frame = cap.read()
            if ret:
                scene_filename = os.path.join(output_folder, f"scene_{i + 1}_start.jpg")
                cv2.imwrite(scene_filename, frame)
                scene_images.append(scene_filename)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, end.get_frames())
            ret, frame = cap.read()
            if ret:
                scene_filename = os.path.join(output_folder, f"scene_{i + 1}_end.jpg")
                cv2.imwrite(scene_filename, frame)
                scene_images.append(scene_filename)
               
    finally:
        video_manager.release()
        cap.release()

    return scene_images

# generate captions for scene images
def generate_captions_for_scenes(scene_images, captions_file="scene_captions.json"):
    if os.path.exists(captions_file):
        print("Loading captions from existing file...")
        with open(captions_file, "r") as file:
            captions = json.load(file)
        return captions

    captions = {}
    for image_path in scene_images:
        caption = generate_caption(image_path)
        scene_number = os.path.basename(image_path).split("_")[1]  
        captions[scene_number] = caption

    with open(captions_file, "w") as file:
        json.dump(captions, file, indent=4)

    return captions

# generate caption using moondream2
def generate_caption(image_path):
    print(f"Generating caption for {image_path} using moondream2...")
    image = Image.open(image_path)
    encoded_image = model.encode_image(image)
    caption = model.caption(encoded_image)["caption"]
    return caption

# search scenes using normalized captions
def search_scenes(captions, search_word):
    search_word = preprocess_word(search_word)
    found_scenes = []
    for scene_num, caption in captions.items():
        normalized_caption = " ".join(preprocess_word(word) for word in caption.split())
        if search_word in normalized_caption:
            found_scenes.append(scene_num)
        elif fuzz.partial_ratio(search_word, normalized_caption) > 70:
            found_scenes.append(scene_num)
    return found_scenes

# create a collage of images
def create_collage(image_paths, output_collage="collage.png"):
    images = [Image.open(image_path) for image_path in image_paths]
    num_images = len(images)
    max_images_per_row = 4 
    num_rows = (num_images // max_images_per_row) + (1 if num_images % max_images_per_row != 0 else 0)
    total_width = sum(img.width for img in images[:max_images_per_row])
    total_height = sum(img.height for img in images[::max_images_per_row])
    collage = Image.new("RGB", (total_width, total_height))
    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        if i % max_images_per_row == 0 and i != 0:
            x_offset = 0
            y_offset += images[i - 1].height
        collage.paste(img, (x_offset, y_offset))
        x_offset += img.width

    collage.save(output_collage)
    print(f"Collage saved to {output_collage}")

    try:
        if os.name == "posix":  # macOS/Linux
            os.system(f"open {output_collage}")
        elif os.name == "nt":  # Windows
            os.startfile(output_collage)
    except Exception as e:
        print(f"Could not open collage: {e}")

# Process video with Gemini model
def process_video_with_gemini(video_path, search_word):
    print("Uploading file...")
    video_file = genai.upload_file(path=video_path)
    print(f"Completed upload: {video_file.uri}")

    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Video upload failed.")

    prompt = "Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions."
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    print("Making LLM inference request...")
    response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
    
    print("Processing LLM response...")
    response_text = response.text
    frames_with_word = []
    for line in response_text.split('\n'):
        if search_word.lower() in line.lower():
            timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                frames_with_word.append(timestamp)

    if not frames_with_word:
        print(f"No frames found with the word '{search_word}'.")
        return

    output_folder = "gemini_frames"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    image_paths = []

    for timestamp in frames_with_word:
        h, m, s = map(int, timestamp.split(':'))
        frame_number = int((h * 3600 + m * 60 + s) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_folder, f"frame_{timestamp.replace(':', '-')}.jpg")
            cv2.imwrite(frame_filename, frame)
            image_paths.append(frame_filename)

    cap.release()

    if image_paths:
        create_collage(image_paths)

def main():
    video_path = "video.mp4"
    captions_file = "scene_captions.json"

    print("Hello! I can help you find specific scenes in your video.")
    user_choice = input("Would you like to search by images (1) or by video transcript (2)? Please type 1 or 2: ").strip()
    
    if user_choice == "1":
        print("\nGreat! I'll start by detecting scenes in your video and generating captions for them.")
        scene_images = detect_scenes_and_save_images(video_path)
        captions = generate_captions_for_scenes(scene_images, captions_file)

        all_words = set()
        for caption in captions.values():
            words = caption.split()
            normalized_words = [preprocess_word(word) for word in words]
            all_words.update(normalized_words)
        word_completer = WordCompleter(list(all_words), ignore_case=True)
        session = PromptSession(completer=word_completer)

        try:
            print("\nNow, let's search for a specific word in the video.")
            search_word = session.prompt("Please enter the word you'd like to search for: ").strip()

            found_scenes = search_scenes(captions, search_word)
            if found_scenes:
                print(f"Found {len(found_scenes)} scenes with the word '{search_word}':")
                scene_images_to_collage = [os.path.join("scenes", f"scene_{scene}_start.jpg") for scene in found_scenes]
                create_collage(scene_images_to_collage)
            else:
                print(f"Oops! I couldn't find any scenes with the word '{search_word}'.")
        except KeyboardInterrupt:
            print("\nNo worries, search interrupted. Feel free to try again!")
        except EOFError:
            print("\nSearch aborted. Let me know if you'd like to try something else.")
    
    elif user_choice == "2":
        print("\nAwesome! I'll process the video and search for the word in the video transcript.")
        search_word = input("What word would you like me to search for in the video? ").strip()
        process_video_with_gemini(video_path, search_word)
    else:
        print("\nHmm, I didn't quite catch that. Please type 1 for images or 2 for the video model.")
        print("No worries, I’ll be here to help when you're ready!")

if __name__ == "__main__":
    main()
