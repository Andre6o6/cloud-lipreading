import ffmpeg
import cv2

def render_line(video_path, subtitles):
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


def render_subtitles(video_path, out_path, subs, offsets=75):
    """
    params:
        video_path: str  - Path to video
        subtitle: [str]  - List of subtitle strings
        offsets: [int]   - subtitle offsets in frames

    """
    vidcap = cv2.VideoCapture(video_path)
    ret, image = vidcap.read()

    #Output stream of frames
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    size = (image.shape[1], image.shape[0])
    fps = 29
    out = cv2.VideoWriter(out_path, fourcc, fps, size)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x,y = (10,500)

    n_frame = 0
    n_line = 0
    while ret:
        #text_w, text_h = cv2.getTextSize(subs[n_line], font, 1, 2)[0]
        cv2.putText(image, subs[n_line],(x,y), font, 1, 255)
        out.write(image)

        ret,image = vidcap.read()
        n_frame +=1
        if n_frame>offsets:
            n_line += 1
            n_frame = 0
    out.release()