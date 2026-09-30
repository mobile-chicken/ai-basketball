"""Replay recorded detections through production logic, without model reruns."""
import argparse
import json
from pathlib import Path
from shot_logic import associate, confirm


def replay(cache, rim):
    tracks=[]; events=[]
    with Path(cache).open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            tracks=associate(tracks,row['points'],row['time'],rim[1]-rim[0],rim)
            for track in tracks:
                t=confirm(track,rim)
                if t is not None and all(abs(t-e['score_time'])>2 for e in events):
                    events.append(dict(score_time=t,evidence=list(track)))
    return events


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('cache'); p.add_argument('--rim',required=True); p.add_argument('--report',required=True)
    a=p.parse_args(); rim=tuple(map(float,a.rim.split(',')))
    events=replay(a.cache,rim)
    report=dict(rim=rim,events=events,note='Detection replay, not a new detector run; candidates need review.')
    Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print([round(e['score_time'],3) for e in events])
