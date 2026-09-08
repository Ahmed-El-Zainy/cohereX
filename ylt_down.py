import yt_dlp

youtube_url = "https://youtu.be/HI7yJc0waTM"

ydl_opts = {
    "format": "bv*+ba/b",

    "outtmpl": "saudi_business_03min.%(ext)s",

    "download_ranges": lambda info, ydl: [
        {
            "start_time": 720,
            "end_time": 900,
        }
    ],

    "extractor_args": {
        "youtube": {
            "player_client": ["web"],
        }
    },

    "merge_output_format": "mp4",

    "postprocessors": [
        {
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }
    ],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([youtube_url])

print("Downloaded: saudi_business_03min.mp4")