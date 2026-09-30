"""Conservative image-plane evidence; no filename or event-time special cases."""
import math
from statistics import median


def review_candidate(track,rim):
    """Broad near-hoop attempt for HUMAN labeling, not a made-shot decision."""
    if len(track)<3: return None
    left,right,y=rim; w=right-left; cx=(left+right)/2
    last=track[-1]
    recent=[p for p in track if last['time']-p['time']<=.8]
    if len(recent)<3: return None
    if abs(last['x']-cx)>w*1.25 or abs(last['y']-y)>w*.8: return None
    if max(p['y'] for p in recent)-min(p['y'] for p in recent)<w*.25: return None
    return last['time']


def associate(tracks, points, now, width, rim=None):
    # A ball/net merged box must not start a rival track and steal later
    # genuine ball detections at the net exit.
    points=[p for p in points if .045*width<p['radius']<.45*width]
    tracks = [t for t in tracks if now-t[-1]['time'] <= .60]
    edges = []
    for i,t in enumerate(tracks):
        a=t[-1]; dt=now-a['time']
        if dt<=0: continue
        vx=vy=0
        if len(t)>1:
            b=t[-2]; delta=a['time']-b['time']
            vx=(a['x']-b['x'])/delta; vy=(a['y']-b['y'])/delta
        for j,p in enumerate(points):
            ratio=p['radius']/max(a['radius'],1)
            if not .55<ratio<1.8: continue
            distance=math.hypot(p['x']-a['x']-vx*dt,p['y']-a['y']-vy*dt)
            # The net changes velocity abruptly. Only around the actual rim,
            # allow a size-consistent below-net reappearance after occlusion.
            if rim is not None and dt>.08:
                left,right,y=rim; cx=(left+right)/2
                if (abs(a['x']-cx)<width*.65 and abs(p['x']-cx)<width*.45
                        and y-width<a['y']<y+width*.3 and y<p['y']<y+width*1.1):
                    distance=min(distance,math.hypot(p['x']-a['x'],p['y']-a['y'])*.65)
            gate=width*(.35+2*dt)
            if distance<gate:
                edges.append((distance/gate+abs(math.log(ratio))*.2,i,j))
    used_t=set(); used_p=set()
    for _,i,j in sorted(edges):
        if i in used_t or j in used_p: continue
        tracks[i].append(points[j]); used_t.add(i); used_p.add(j)
        tracks[i][:]=[p for p in tracks[i] if now-p['time']<=1.8]
    tracks.extend([[p] for j,p in enumerate(points) if j not in used_p])
    return tracks


def confirm(track, rim):
    left,right,y=rim; w=right-left; cx=(left+right)/2
    if len(track)<5 or w<=0: return None
    for i,(a,b) in enumerate(zip(track,track[1:])):
        dt=b['time']-a['time']
        if not 0<dt<=.60 or not a['y']<y<=b['y']: continue
        r=median([a['radius'],b['radius']])
        if not .05*w<r<.48*w: continue
        frac=(y-a['y'])/(b['y']-a['y'])
        t=a['time']+frac*dt; x=a['x']+frac*(b['x']-a['x'])
        margin=max(.1*w,.55*r)
        if not left+margin<x<right-margin: continue
        before=[p for p in track[:i+1] if t-p['time']<=.65]
        if len(before)<2 or min(p['y'] for p in before)>y-max(r*.4,.08*w): continue
        # Require a measured observation close to the rim; a long interpolated
        # jump from above to far below the hoop is insufficient evidence.
        if min(abs(p['y']-y) for p in (a,b))>max(r,.45*w): continue
        after=[p for p in track[i+1:] if p['time']<=t+1.20]
        exits=[j for j,p in enumerate(after) if p['y']>=y+max(2*r,.65*w)]
        if not exits: continue
        path=after[:exits[0]+1]
        if len(path)<2 or path[-1]['time']-t<.04: continue
        # Inspect ALL points through net exit; never discard a rebound or
        # sideways escape and then accept the remaining favorable points.
        prev=a; valid=True
        for p in path:
            if p['time']-prev['time']>.60 or p['y']<prev['y']-max(.3*r,.06*w): valid=False
            if abs(p['x']-cx)>w*.45 or not .55< p['radius']/r <1.8: valid=False
            prev=p
        if not valid: continue
        # In a single camera, an airball in front of/behind the hoop can have
        # exactly the same 2-D crossing. Require evidence of net interaction:
        # measured descent through the net must slow relative to approach.
        # This conservative gate may miss fast clean swishes; absence of
        # slowdown is insufficient to assert a made basket.
        approach=before[-min(4,len(before)):]
        span=approach[-1]['time']-approach[0]['time']
        if span<=0: continue
        vy=(approach[-1]['y']-approach[0]['y'])/span
        net_vy=(path[-1]['y']-y)/(path[-1]['time']-t)
        occluded=dt>=.12 and a['y']<y and b['y']>y
        if not occluded and (vy<=0 or net_vy>vy*.85): continue
        if median(p['confidence'] for p in before+path)<.12: continue
        return t
    return None
