# =================================libraries=================================
import os
import json
import glob
import re
import pandas as pd
from tqdm import tqdm
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# =================================parameters=================================
# API Keys
CLIENT_ID = "376a1acf3d214e52a70c56a3c648b66b"
CLIENT_SECRET = "a99c371f5d1243e397ba2bf0c325cda3"
BASE_PATH = "C:/KC/NCU/Intro_to_Data_Science/Kpop-lyric-datasets/melon/monthly-chart"

# =================================functions=================================
def calculate_english_line_ratio(lines_list):
    if not lines_list:
        return 0.0
    
    english_line_count = 0
    valid_line_count = 0
    
    for line in lines_list:
        clean_line = str(line).strip()
        
        if not clean_line:
            continue
            
        valid_line_count += 1

        if re.match(r'^[a-zA-Z]', clean_line):
            english_line_count += 1
            
    if valid_line_count == 0:
        return 0.0
        
    return english_line_count / valid_line_count

def calculate_english_word_ratio(line_list):
    if not line_list:
        return 0.0

    all_words = " ".join(line_list).split()
    total_words = len(all_words)
    if total_words == 0:
        return 0.0

    english_words = [word for word in all_words if re.match(r'^[a-zA-Z]+$', word)]
    english_word_count = len(english_words)
    # print(f"Total words: {total_words}, English words: {english_word_count}")

    return (english_word_count / total_words)

def get_lyrics(output_file):
    all_files = glob.glob(os.path.join(BASE_PATH, "*", "*", "*.json"))
    print(f"找到 {len(all_files)} 個 JSON 檔案")

    lyrics_cache = {}

    song_data_buffer = []

    print("正在處理 JSON 檔案...")
    for file_path in tqdm(all_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                info = data.get('info', [])
                
                rank_val = info[0].get('rank')
                if rank_val:
                    rank_val = int(rank_val)

                song_id = data.get('song_id')
                if song_id is None:
                    continue
                
                line_ratio = 0.0
                word_ratio = 0.0
                if song_id in lyrics_cache:
                    line_ratio = lyrics_cache[song_id]['line_ratio']
                    word_ratio = lyrics_cache[song_id]['word_ratio']
                else:
                    lines = data.get('lyrics', {}).get('lines', [])
                    line_ratio = calculate_english_line_ratio(lines)
                    word_ratio = calculate_english_word_ratio(lines)
                    lyrics_cache[song_id] = {'line_ratio':line_ratio, 'word_ratio': word_ratio}

                song_data_buffer.append({
                    'song_id': song_id,
                    'song_name': data.get('song_name'),
                    'artist': data.get('artist'),
                    'rank': rank_val,
                    'eng_line_ratio': line_ratio,
                    'eng_word_ratio': word_ratio,
                    'year' : info[0].get('year'),
                    'month' : info[0].get('month')
                })

        except Exception as e:
            pass

    df_raw = pd.DataFrame(song_data_buffer)
    df_raw.to_csv(f"dataset/{output_file}", index=False)
    return df_raw

def clean_artist_name(name, artist_map):
    # strip whitespace
    name = str(name).strip()

    # Eng + (Kor) → keep Eng
    match_eng_kor = re.match(r"^([A-Za-z0-9 .,'&/-]+)\s*\(([가-힣 ]+)\)$", name)
    if match_eng_kor:
        return match_eng_kor.group(1).strip()

    # Kor + (Eng) → keep Eng
    match = re.search(r"[(]([A-Za-z0-9 .,'/-]+)[)]", name)
    if match:
        return match.group(1).strip()

    # Eng → keep
    if re.fullmatch(r"[A-Za-z0-9 .,&'-]+", name):
        return name

    # Kor → translate using map
    if name in artist_map:
        return artist_map[name]
    
    return name

def clean_song_title(title):
    title = str(title).strip()

    # not Kor → keep
    if not re.search(r"[가-힣]", title):
        return title

    # Kor + (Eng) → keep Eng
    match_parens = re.search(r"[(]([A-Za-z0-9 .,&'!\?/-]+)[)]", title)
    if match_parens:
        # filter out "feat."
        extracted = match_parens.group(1).strip()
        if not extracted.lower().startswith("feat.") and "inst." not in extracted.lower():
            return extracted

    # Eng + Kor → keep Eng
    match_eng_kor = re.match(r"^([A-Za-z0-9 .,'&/-]+)\s*([가-힣 ]+)$", title)
    if match_eng_kor:
        return match_eng_kor.group(1).strip()

    return title

def artist_name_mapping(rank_dataset, artist_map):
    rank_dataset["artist"] = rank_dataset["artist"].apply(lambda x: clean_artist_name(x, artist_map))
    rank_dataset["song_name"] = rank_dataset["song_name"].apply(clean_song_title)
    return rank_dataset

def clean_ranking_dataset(file_name, output_file):
    kpop_rank = pd.read_csv(f"dataset/{file_name}")
    print("reading " + f"dataset/{file_name}")
    print(f"# origin dataset: {len(kpop_rank)}")

    # read dict for artist name mapping
    print("reading dataset/kpopgroups_edited.csv")
    groups_name = pd.read_csv("dataset/kpopgroups_edited.csv")
    artist_map = dict(zip(groups_name["Korean Name"], groups_name["Name"]))

    kpop_rank = artist_name_mapping(kpop_rank, artist_map)

    kpop_map = groups_name.set_index("Name")[["Company", "Gender"]].to_dict(orient="index")
    for index, row in kpop_rank.iterrows():
        artist = row["artist"]
        if artist in kpop_map:
            kpop_rank.at[index, "company"] = kpop_map[artist]["Company"]
            kpop_rank.at[index, "gender"] = kpop_map[artist]["Gender"]
        else:
            kpop_rank.at[index, "company"] = "Unknown"
            kpop_rank.at[index, "gender"] = "Unknown"
    
    kpop_rank.to_csv(f"dataset/{output_file}", index=False)
    return

def solo_artist_filter(file_name, output_file):
    kpop_rank = pd.read_csv(f"dataset/{file_name}")
    print("reading " + f"dataset/{file_name}")
    print(f"# origin dataset: {len(kpop_rank)}")

    idol_data = pd.read_csv("dataset/kpopidolsv3.csv")
    idol_data["Gender"] = idol_data["Gender"].replace("M", "Male")
    idol_data["Gender"] = idol_data["Gender"].replace("F", "Female")

    master_lookup = {}
    for index, row in idol_data.iterrows():
        standard_info = {
            "stage_name": row["Stage Name"],
            "company":  row["Company"],
            "gender":   row["Gender"]
        }
        
        potential_names = [
            row["Stage Name"], 
            row["Korean Name"], 
            row["K Stage Name"]
        ]

        for name in potential_names:
            if pd.notna(name) and str(name).strip() != "":
                clean_name = str(name).strip()
                master_lookup[clean_name] = standard_info
    
    for index, row in kpop_rank.iterrows():
        artist = row["artist"]
        if artist in master_lookup:
            kpop_rank.at[index, "artist"] = master_lookup[artist]["stage_name"]
            kpop_rank.at[index, "company"] = master_lookup[artist]["company"]
            kpop_rank.at[index, "gender"] = master_lookup[artist]["gender"]

    kpop_rank.to_csv(f"dataset/{output_file}", index=False)
    
    return

def get_spotify_client():
    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp

def fetch_spotify_info(sp, artist, song_kor):
    query = f"{artist} {song_kor}"
    try:
        results = sp.search(q=query, type='track', limit=1)
        items = results['tracks']['items']
        if items:
            return {
                'official_title': items[0]['name'],
                'spotify_id': items[0]['id']
            }
    except Exception:
        pass
    return None

def get_info_from_map(row, key_type, song_map):
    lookup_key = f"{row['artist']}_{row['song_name']}"

    data = song_map.get(lookup_key) 
    if data:
        return data.get(key_type)
    return None

def group_ranking_per_song(kpop_file):
    kpop_rank = pd.read_csv(f'dataset/{kpop_file}')

    df_grouped = kpop_rank.groupby("song_id").agg({
        'artist': 'first',
        'song_name': 'first', 
        'rank': 'mean',
        'month': 'count',
        'year': 'first',
        'eng_line_ratio': 'first',
        'eng_word_ratio': 'first'
    }).reset_index()

    df_grouped = df_grouped.rename(columns={
        'rank': 'avg_rank',
        'month': 'months_on_chart'
    })

    print(f"# unique song: {len(df_grouped)}")
    return df_grouped

def get_spotify_id(cache_file, kpop_file, output_name):
    kpop_df = pd.read_csv(f"dataset/{kpop_file}")
    song_map = {}

    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            song_map = json.load(f)
        print(f"read existed dict, including {len(song_map)} songs")

    unique_pairs = kpop_df[['artist', 'song_name']].drop_duplicates().values.tolist()
    print(f"# unique songs: {len(unique_pairs)} (times to call API)")

    sp = get_spotify_client()
    is_updated = False

    print("start calling Spotify API...")
    for artist, song in tqdm(unique_pairs):
        lookup_key = f"{artist}_{song}"
        
        if lookup_key in song_map:
            continue 
        
        result = fetch_spotify_info(sp, artist, song)
        
        if result:
            song_map[lookup_key] = result
        else:
            song_map[lookup_key] = {'official_title': None, 'spotify_id': None}
        
        is_updated = True
        time.sleep(0.1) 
        
        if len(song_map) % 100 == 0:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(song_map, f, ensure_ascii=False)

    if is_updated:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(song_map, f, ensure_ascii=False, indent=4)
        print("update and save dict")

    print("mapping to rank data...")
    kpop_df['spotify_title'] = kpop_df.apply(lambda row: get_info_from_map(row, 'official_title', song_map), axis=1)
    kpop_df['spotify_id'] = kpop_df.apply(lambda row: get_info_from_map(row, 'spotify_id', song_map), axis=1)

    print(kpop_df)
    kpop_df.to_csv(f"dataset/{output_name}", index=False)
    return kpop_df

def merge_datasets(tracks_file, rank_file, output_name):
    kpop_rank = pd.read_csv(f"dataset/{rank_file}")
    tracks_data = pd.read_csv(f"dataset/{tracks_file}")
    COL = ['Artist', 'Artist_Id', 'Track_Title', 'Track_Id', 'danceability', 
           'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
           'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'eng_line_ratio', 'eng_word_ratio', 'avg_rank', 'months_on_chart', 'company', 'gender']

    final_dataset = tracks_data.merge(kpop_rank, left_on="Track_Id", right_on="spotify_id", how="inner", suffixes=('_track', '_rank'))
    final_dataset = final_dataset[[col for col in COL if col in final_dataset.columns]]
    if 'spotify_id' in final_dataset.columns:
        final_dataset = final_dataset.drop(columns=['spotify_id'])

    final_dataset["company"] = final_dataset["company"].fillna("Unknown")
    final_dataset["gender"] = final_dataset["gender"].fillna("Unknown")
    final_dataset = final_dataset.dropna()
    final_dataset = final_dataset.reset_index(drop=True)

    print(f"# final unique songs: {len(final_dataset)}")
    print(final_dataset.head())
    final_dataset.to_csv(f"dataset/{output_name}", index=False)
    return

# =================================main code=================================
# clean_ranking_dataset("kpop_lyrics_rank.csv", "kpop_lyrics_rank_cleaned.csv")
# solo_artist_filter("kpop_lyrics_rank_cleaned.csv", "kpop_lyrics_rank_cleaned.csv")
# get_spotify_id("ver3_cache.json", "kpop_lyrics_rank_cleaned.csv", "kpop_lyrics_rank_spotify_v3.csv")
merge_datasets("single_album_track_data.csv", "kpop_lyrics_rank_spotify_v3.csv", "final_kpop_dataset_v3.csv")