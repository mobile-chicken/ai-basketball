"""Multi-frame red-rim localization for fixed-camera gym recordings.

Color/geometry heuristic, not a trained hoop detector. Fail on ambiguity.
"""
import cv2
import numpy as np


def backboard_score(background,left,right,rim_y):
    """Require dark backboard side rails around and above the red rim.

    This structural check replaces the preference for the image center.
    """
    width=right-left
    if width<=0: return 0.
    h,w=background.shape[:2]
    x0=max(0,int(left-3*width)); x1=min(w,int(right+2*width))
    y0=max(0,int(rim_y-3*width)); y1=min(h,int(rim_y+.7*width))
    gray=cv2.cvtColor(background[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    dark=(gray<95).astype(np.uint8)*255
    vertical=cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((max(5,int(width*.6)),1),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(vertical)
    rails=[]
    for x,y,cw,ch,area in stats[1:]:
        px=x0+x+cw/2; top=y0+y; bottom=top+ch
        if cw>width*.35 or ch<width*.65: continue
        if top>rim_y-width*.55 or bottom<rim_y-width*.15: continue
        if bottom>rim_y+width*.7: continue
        rails.append((px,top,bottom,ch))
    lefts=[r for r in rails if left-width*2.8<r[0]<left-width*.15]
    rights=[r for r in rails if right+width*.05<r[0]<right+width*1.5]
    score=0.
    for a in lefts:
        for b in rights:
            if not 1.4*width<b[0]-a[0]<4.5*width: continue
            if abs(a[2]-b[2])>width*.45: continue
            score=max(score,min(a[3],b[3])/width)
    return float(score)


def auto_calibrate(video, out=None):
    cap=cv2.VideoCapture(str(video)); frames=[]
    duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/max(cap.get(cv2.CAP_PROP_FPS),1)
    for sec in np.linspace(0,min(12,max(0,duration-.1)),9):
        cap.set(cv2.CAP_PROP_POS_MSEC,float(sec)*1000)
        ok,frame=cap.read()
        if ok: frames.append(frame)
    cap.release()
    if len(frames)<5: raise RuntimeError('Insufficient frames for hoop calibration')
    background=np.median(frames,axis=0).astype(np.uint8)
    h,w=background.shape[:2]; scale=w/2688
    masks=[]
    for frame in frames:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        masks.append((((hsv[:,:,0]<22)|(hsv[:,:,0]>170)) & (hsv[:,:,1]>60) & (hsv[:,:,2]>35)))
    mask=(np.mean(masks,axis=0)>=.55).astype(np.uint8)*255
    # Bridge compression gaps, but never double the apparent rim width.
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((1,max(2,round(3*scale))),np.uint8))
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    candidates=[]
    for label,(x,y,cw,ch,area) in enumerate(stats[1:],1):
        cx=x+cw/2; cy=y+ch/2
        # Do not assume the hoop is near the image center. The camera may
        # pan so the target hoop is on either side of the frame.
        if not (0.0 <= cx < w and 0.0 <= cy < h): continue
        if not (20*scale<cw<160*scale and 3*scale<ch<65*scale and cw/ch>1.4): continue
        region=(labels[y:y+ch,x:x+cw]==label)
        rows=region.sum(axis=1); peak=int(rows.argmax())
        if rows[peak]<cw*.5: continue
        # Use the upper rim band, not the mount/net below it.
        band=region[max(0,peak-2):min(ch,peak+3)]
        xs=np.where(band.any(axis=0))[0]
        left=float(x); right=float(x+cw-1); rim_y=float(y+peak)
        if right-left<18*scale: continue
        structure=backboard_score(background,left,right,rim_y)
        if structure<.8: continue
        candidates.append((-structure,(left,right,rim_y)))
    candidates.sort()
    if not candidates or (len(candidates)>1 and candidates[1][0]-candidates[0][0]<.15):
        raise RuntimeError('Hoop calibration ambiguous; no clips exported. Needs clearer fixed-camera view.')
    rim=candidates[0][1]
    if out is not None:
        cv2.rectangle(background,(int(rim[0]),int(rim[2]-4)),(int(rim[1]),int(rim[2]+4)),(0,255,0),2)
        cv2.imencode('.jpg',background)[1].tofile(str(out/'calibration.jpg'))
    return rim
