import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yt_dlp
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment
import argparse
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

def get_playlist_tracks(playlist_id):
    """Retrieve track names and artists from a Spotify playlist"""
    results = sp.playlist_tracks(playlist_id)
    tracks = []
    for item in results['items']:
        track = item['track']
        name = track['name']
        artist = track['artists'][0]['name']
        tracks.append(f"{name} - {artist}")
    return tracks

def search_youtube(song_name):
    """Use yt-dlp to search YouTube for the best lyric video match"""
    search_query = f"{song_name} lyric video"
    
    ydl_opts = {
        "quiet": True,
        "default_search": "ytsearch5",  # Search and return 5 results
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_results = ydl.extract_info(search_query, download=False)
    
    # Extract first relevant video URL
    if "entries" in search_results and search_results["entries"]:
        for entry in search_results["entries"]:
            if "lyric" in entry.get("title", "").lower():
                return entry["webpage_url"]  # Prioritize lyric videos

        return search_results["entries"][0]["webpage_url"]  # Fallback to first result

    return None  # No results found

def download_audio(video_url, save_path):
    """Download YouTube video directly as .wav audio"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': save_path + ".wav",  # Save as .wav directly
        'ffmpeg_location': "/opt/miniconda3/envs/aidj/bin/ffmpeg",  # Ensure this path is correct
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Download as .wav directly
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    print(f"Downloaded and saved as: {save_path}.wav")

"""def main():
    ""Prompts the user for a song name, searches YouTube, and downloads the audio""
    song_name = input("Enter the song name: ").strip()
    save_directory = "data/raw-wavs"
    os.makedirs(save_directory, exist_ok=True)

    print(f"Searching for: {song_name}...")
    video_url = search_youtube(song_name)

    if video_url:
        print(f"Found: {video_url}")
        save_path = os.path.join(save_directory, song_name.replace(" ", "_"))
        print("Downloading and converting to WAV...")
        download_audio(video_url, save_path)
        print(f"Download complete: {save_path}.wav")
    else:
        print("No suitable video found.")

if __name__ == "__main__":
    main()"""


def main(playlist_id):
    tracks = get_playlist_tracks(playlist_id)
    save_directory = "data/raw-wavs"
    os.makedirs(save_directory, exist_ok=True)

    for track in tracks:
        print(f"Processing: {track}")
        video_url = search_youtube(track)
        if video_url:
            print(f"Found: {video_url}")
            save_path = os.path.join(save_directory, track.replace(" ", "_"))
            download_audio(video_url, save_path)
            print(f"Downloaded: {track}.wav")
        else:
            print(f"Could not find: {track}")

if __name__ == "__main__":
    # Set up argument parser to read playlist ID from the command line
    parser = argparse.ArgumentParser(description="Encode audio and save codes for a playlist.")
    parser.add_argument("playlist_id", type=str, help="Playlist ID to identify the encoded codes")
    
    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the playlist ID from the argument
    main(args.playlist_id)

