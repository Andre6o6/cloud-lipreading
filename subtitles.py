import ffmpeg

def render_subtitles(video_path, subtitles, offsets=[]):
    """
    params:
        video_path: str  - Path to video
        subtitle: [str]  - List of subtitle strings
        offsets: [int]   - List of subtitle offsets in frames
    """
    #FIXME
    sub = subtitles[0]
    name, ext = video_path.split('.')
    name = name+"_subbed."+ext
    
    file = ffmpeg.input(video_path)
    audio = file.audio
    video = file.drawtext(
            text=sub,
            box=1,
            boxborderw=2,
            boxcolor="black",
            fontcolor="white",
            x="(w-tw)/2",
            y="h-1.5*th",
        )
    ffmpeg.output(video, audio, name).global_args( '-loglevel', 'error').run()