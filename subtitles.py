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


def render_subtitles(video_path, out_path, subs, window=75, color=(255,255,255), box=True):
    """
    params:
        video_path: str  - Path to video
        subtitle: [str]  - List of subtitle strings
        window: int   - subtitle window in frames

    """
    vidcap = cv2.VideoCapture(video_path)
    ret, image = vidcap.read()

    #Output stream of frames
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    size = (image.shape[1], image.shape[0])
    fps = 29
    out = cv2.VideoWriter(out_path, fourcc, fps, size)

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    box_color = (0,0,0) if color[0] > 127 else (255,255,255)
    
    offset_from_bottom = 5
    box_offset = 2
    font_scale = 1

    n_frame = 0
    n_line = 0
    while ret:
        text_w, text_h = cv2.getTextSize(subs[n_line], font, font_scale, 1)[0]
        x = size[0]//2 - text_w//2
        y = size[1] - offset_from_bottom
        if box:
            cv2.rectangle(image, (x,y+box_offset), (x+text_w,y-text_h-box_offset), box_color, -1)
        cv2.putText(image, subs[n_line],(x,y), font, font_scale, color)
        out.write(image)

        ret,image = vidcap.read()
        n_frame +=1
        if n_frame>window:
            n_line += 1
            n_frame = 0
    out.release()