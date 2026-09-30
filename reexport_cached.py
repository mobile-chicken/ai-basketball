"""Apply current shot rules to cached detector output and export review clips."""
import argparse
import json
from pathlib import Path
from run_pipeline import export_event, FFPROBE
from shot_logic import associate, confirm, review_candidate
import subprocess


def run(source, out):
    source=Path(source); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    report=json.loads((source/'events.json').read_text(encoding='utf-8'))
    rim=report['rim']; video=Path(report['video']); tracks=[]; events=[]; attempts=[]
    duration=float(subprocess.check_output([FFPROBE,'-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(video)]))
    with (source/'detections.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            tracks=associate(tracks,row['points'],row['time'],rim[1]-rim[0],rim)
            for track in tracks:
                t=review_candidate(track,rim)
                if t is not None and all(abs(t-x)>2 for x in attempts): attempts.append(t)
                t=confirm(track,rim)
                if t is not None and all(abs(t-e['score_time'])>2 for e in events):
                    events.append(dict(score_time=t,evidence=list(track),review_status='pending'))
    for i,e in enumerate(events,1): export_event(video,out,e,i,duration,4,2)
    review=[]; review_out=out/'review_candidates'; review_out.mkdir()
    for t in attempts:
        if any(abs(t-e['score_time'])<=2 for e in events): continue
        e=dict(score_time=t,review_status='unlabeled',note='NOT a confirmed goal')
        export_event(video,review_out,e,len(review)+1,duration,4,2); review.append(e)
    report.update(status='complete',events=events,before=4,after=2,replayed_from=str(source),mode='cached_detections_current_rules')
    (out/'events.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (review_out/'events.json').write_text(json.dumps(dict(video=str(video),events=review,purpose='human_labeling_only'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(events=len(events),review_candidates=len(review),times=[round(e['score_time'],2) for e in events])))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('output'); a=p.parse_args(); run(a.source,a.output)
