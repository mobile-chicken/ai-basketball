"""Pure detector + trajectory state machine pipeline.

No manually selected events are read. Input is a video and rim calibration;
output is events.json plus full-frame clips around state-machine confirmations.
"""
import argparse, json, subprocess, time, os
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent
(ROOT/'.cache'/'ultralytics').mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache'/'ultralytics')
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache'/'matplotlib')
os.environ['TORCH_HOME']=str(ROOT/'.cache'/'torch')
from ultralytics import YOLO
from shot_logic import confirm, associate, review_candidate
from calibration import auto_calibrate
FFMPEG = r'D:/software/FFmpeg/ffmpeg-9.0.2-full_build/bin/ffmpeg.exe'
FFPROBE = str(Path(FFMPEG).with_name('ffprobe.exe'))

def clip_window(t, duration, before, after):
    return max(0., t-before), min(duration, t+after)

def export_event(video, out, event, index, duration, before, after):
    event['event_id']=f'score_{index:03}'
    event['clip_start'],event['clip_end']=clip_window(event['score_time'],duration,before,after)
    target=out/f'{index:03}_{event["score_time"]:.2f}.mp4'
    subprocess.run([FFMPEG,'-v','error','-ss',str(event['clip_start']),'-i',str(video),'-t',str(event['clip_end']-event['clip_start']),'-map','0:0','-map','0:a:0?','-c','copy','-avoid_negative_ts','make_zero','-y',str(target)],check=True)
    event['file']=target.name
    (out/'latest_event.json').write_text(json.dumps(event,ensure_ascii=False,indent=2),encoding='utf-8')

def run(args):
    video=Path(args.video); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    meta=json.loads(subprocess.check_output([FFPROBE,'-v','error','-show_format','-show_streams','-of','json',str(video)],encoding='utf-8'))
    duration=float(meta['format']['duration']);
    vstream=next(s for s in meta['streams'] if s.get('codec_type')=='video')
    video_w=int(vstream['width']); video_h=int(vstream['height'])
    num,den=(vstream.get('avg_frame_rate') or vstream.get('r_frame_rate') or '50/1').split('/')
    if float(den)<=0 or float(num)<=0: raise RuntimeError('Invalid source frame rate')
    fps=float(num)/float(den)
    if min(video_w,video_h)<640: raise RuntimeError('Video must be at least 640 pixels on both axes')
    model=YOLO(str(args.model))
    rim=auto_calibrate(video,out) if args.rim=='auto' else tuple(map(float,args.rim.split(',')))
    ox=max(0,min(video_w-640,int((rim[0]+rim[1])/2-320)))
    oy=max(0,min(video_h-640,int(rim[2]-280)))
    cmd=[FFMPEG,'-v','error','-i',str(video),'-map','0:0','-vf',f'crop=640:640:{ox}:{oy}','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','bgr24','pipe:1']
    tracks=[]; events=[]; pending=[]; attempts=[]; frame=0; started=time.time()
    print(f'篮筐标定: {rim}',flush=True)
    def save_report(status):
        report={'schema_version':2,'status':status,'video':str(video),'model':str(args.model),'rim':list(rim),'crop_origin':[ox,oy],'fps':fps,'before':args.before,'after':args.after,'frames':frame,'elapsed_seconds':time.time()-started,'events':events,'manual_events_used':False}
        temp=out/'events.tmp'
        temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(out/'events.json')
    save_report('running')
    with (out/'detections.jsonl').open('w',encoding='utf-8') as cache, subprocess.Popen(cmd,stdout=subprocess.PIPE) as dec:
        while True:
            raw=dec.stdout.read(640*640*3)
            if not raw: break
            if len(raw)!=640*640*3: raise RuntimeError('partial decoded frame')
            image=np.frombuffer(raw,np.uint8).reshape(640,640,3)
            result=model.predict(image,imgsz=args.imgsz,conf=args.conf,classes=[32],device=0,verbose=False)[0]
            now=frame/fps; points=[]
            for box,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                x1,y1,x2,y2=box; points.append({'time':now,'x':(x1+x2)/2+ox,'y':(y1+y2)/2+oy,'radius':(x2-x1)/2,'confidence':score})
            cache.write(json.dumps({'time':now,'points':points})+'\n')
            tracks=associate(tracks,points,now,rim[1]-rim[0],rim)
            for track in tracks:
                proposal=review_candidate(track,rim)
                if proposal is not None and all(abs(proposal-t)>2 for t in attempts): attempts.append(proposal)
                score=confirm(track,rim)
                if score is not None and all(abs(score-e['score_time'])>2 for e in events):
                    event={'score_time':score,'confidence':'model_state_machine','review_status':'pending','evidence':list(track)}
                    events.append(event); pending.append(event)
            frame+=1
            now=frame/fps
            ready=[e for e in pending if now >= e['score_time']+args.after]
            for e in ready:
                export_event(video,out,e,len([x for x in events if 'file' in x])+1,duration,args.before,args.after)
                pending.remove(e)
                save_report('running')
                print(f'已输出进球片段: {e["file"]}',flush=True)
            if frame % 250 == 0:
                save_report('running')
                print(f'识别进度: {min(frame/fps/duration*100,100):.1f}% ({frame/fps:.1f}/{duration:.1f} 秒)', flush=True)
        if dec.wait()!=0: raise RuntimeError('Video decoding failed')
    if frame==0: raise RuntimeError('No video frames decoded')
    events.sort(key=lambda e:e['score_time'])
    for e in list(pending):
        export_event(video,out,e,len([x for x in events if 'file' in x])+1,duration,args.before,args.after)
        print(f'已输出进球片段: {e["file"]}',flush=True)
    save_report('complete')
    review_out=out/'review_candidates'; review_out.mkdir(exist_ok=True)
    review=[]
    for t in attempts:
        if any(abs(t-e['score_time'])<=2 for e in events): continue
        e={'score_time':t,'review_status':'unlabeled','note':'Near-hoop attempt; NOT a confirmed goal'}
        export_event(video,review_out,e,len(review)+1,duration,args.before,args.after)
        review.append(e)
    (review_out/'events.json').write_text(json.dumps({'video':str(video),'events':review,'purpose':'human_labeling_only'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'待标注近筐片段: {len(review)}，目录: {review_out}',flush=True)
    print(json.dumps({'frames':frame,'events':len(events),'elapsed_seconds':time.time()-started},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('video'); p.add_argument('--output',default='output/model_only'); p.add_argument('--model',default='models/yolo11s.pt'); p.add_argument('--rim',default='auto',help='auto or left,right,y center'); p.add_argument('--before',type=float,default=4); p.add_argument('--after',type=float,default=2); p.add_argument('--imgsz',type=int,default=1280); p.add_argument('--conf',type=float,default=.08); run(p.parse_args())
