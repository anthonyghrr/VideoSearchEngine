import yt_dlp

def download_video(search_term="Super Mario movie trailer", output_path="video.mp4"):
    ydl_opts = {
        'format': 'best',  
        'outtmpl': output_path, 
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Searching for '{search_term}' on YouTube and downloading the video...")
        ydl.download([f"ytsearch:{search_term}"])

# Call the function to download "Super Mario movie trailer"
download_video()
