import argparse, json, subprocess, sys, time, re, os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VIDEO_EXT={'.mp4','.mov','.m4v','.avi','.mkv'}
RIM_DEFAULT='auto'
FFPROBE= r'D:\software\FFmpeg\ffmpeg-9.0.2-full_build\bin\ffprobe.exe'

def get_duration(path):
    try:
        r=subprocess.run([FFPROBE,'-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)], capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='input'); p.add_argument('--output',default='output'); p.add_argument('--rim',default=RIM_DEFAULT); p.add_argument('--before',type=float,default=4); p.add_argument('--after',type=float,default=2); a=p.parse_args()
    src=ROOT/a.input; dst=ROOT/a.output; dst.mkdir(exist_ok=True)
    src.mkdir(parents=True,exist_ok=True)
    videos=sorted(x for x in src.iterdir() if x.is_file() and x.suffix.lower() in VIDEO_EXT)
    if not videos: print('input 文件夹中没有视频'); return 1
    started=time.time(); total=sum(x.stat().st_size for x in videos)
    durations={x:get_duration(x) for x in videos}; total_duration=sum(durations.values()); completed_duration=0.0
    print(f'共 {len(videos)} 个视频，总大小 {total/1024/1024/1024:.2f} GB，总时长 {total_duration/60:.2f} 分钟')
    failures=0
    for i,video in enumerate(videos,1):
        out=dst/video.stem
        if out.exists() and any(out.iterdir()):
            out=out/('run_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000))
        out.mkdir(parents=True,exist_ok=True)
        pct=(completed_duration/total_duration*100) if total_duration else ((i-1)/len(videos)*100)
        print(f'[{i}/{len(videos)}] 开始: {video.name}，时长 {durations[video]/60:.2f} 分钟，整体进度 {pct:.1f}%',flush=True)
        print('篮筐标定: 自动',flush=True)
        cmd=[sys.executable,'-u',str(ROOT/'run_pipeline.py'),str(video),'--rim',a.rim,'--before',str(a.before),'--after',str(a.after),'--output',str(out)]
        env=dict(os.environ,PYTHONIOENCODING='utf-8')
        with (out/'run.log').open('w',encoding='utf-8') as log, subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env) as result:
            for line in result.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
                match=re.search(r'\(([\d.]+)/[\d.]+ 秒\)',line)
                if match and total_duration:
                    done=completed_duration+min(float(match[1]),durations[video])
                    print(f'总扫描进度: {done/total_duration*100:.1f}%',flush=True)
            result.wait()
        if result.returncode:
            failures+=1
            print(f'[{i}/{len(videos)}] 失败: {video.name}，详见 {out / "run.log"}')
        else:
            try:
                r=json.loads((out/'events.json').read_text(encoding='utf-8')); print(f'[{i}/{len(videos)}] 完成: {len(r["events"])} 个候选，耗时 {r["elapsed_seconds"]:.1f} 秒',flush=True)
            except Exception: print(f'[{i}/{len(videos)}] 完成，但无法读取结果')
        completed_duration += durations[video]
        pct=(completed_duration/total_duration*100) if total_duration else (i/len(videos)*100)
        print(f'整体进度 {pct:.1f}%',flush=True)
    print(f'全部处理完成，总耗时 {time.time()-started:.1f} 秒',flush=True)
    return 1 if failures else 0

if __name__=='__main__': raise SystemExit(main())
