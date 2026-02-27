import cv2
import mediapipe as mp
import numpy as np
import pulsectl
import subprocess
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── AUDIO ───────────────────────────────────────────────────
pulse        = pulsectl.Pulse('gesture-volume-control')
sink         = pulse.sink_list()[0]
is_muted     = False
current_vol  = pulse.volume_get_all_chans(sink)
print(f"Audio: {sink.description}")

# ─── MEDIAPIPE ───────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector     = vision.HandLandmarker.create_from_options(options)

# ─── CAMERA ──────────────────────────────────────────────────
# ─── CAMERA ──────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera at index 0. Falling back to index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera at index 1 either. Exiting.")
        exit(1)

# ─── MEDIA CONTROL (DBUS/MPRIS) ──────────────────────────────
def _mpris_player():
    try:
        out = subprocess.check_output(
            ["dbus-send","--session","--print-reply",
             "--dest=org.freedesktop.DBus","/org/freedesktop/DBus",
             "org.freedesktop.DBus.ListNames"],
            stderr=subprocess.DEVNULL, text=True)
        for ln in out.splitlines():
            ln = ln.strip().strip('"')
            if ln.startswith("org.mpris.MediaPlayer2."):
                return ln
    except Exception:
        pass
    return None

def media(action):          # action: "Next" | "Previous" | "PlayPause"
    player = _mpris_player()
    if player:
        subprocess.Popen(
            ["dbus-send","--session","--type=method_call",
             f"--dest={player}","/org/mpris/MediaPlayer2",
             f"org.mpris.MediaPlayer2.Player.{action}"],
            stderr=subprocess.DEVNULL)
    else:
        print(f"[media] no active MPRIS player for: {action}")

# ─── GESTURE CLASSIFIER ──────────────────────────────────────
def fingers_up(hand, label="Right"):
    """[thumb, index, middle, ring, pinky] — 1=extended (flipped frame)."""
    # Thumb detection: Tip (4) vs Knuckle (2) distance to Pinky base (17)
    # This is more robust than x-coord comparison for different hands/mirroring.
    d_tip = np.hypot(hand[4].x - hand[17].x, hand[4].y - hand[17].y)
    d_knk = np.hypot(hand[2].x - hand[17].x, hand[2].y - hand[17].y)
    f = [1 if d_tip > d_knk else 0]

    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        f.append(1 if hand[tip].y < hand[pip].y else 0)
    return f

def classify(hand, label="Right"):
    """Return one gesture label, or None.
    Priority order matters — check exact matches first, relaxed VOLUME last."""
    f = fingers_up(hand, label)
    n = sum(f)
    # Exact matches first (strict)
    if n == 0:                return "FIST"     # ✊ all fingers down
    if f == [0,1,0,0,0]:     return "SWIPE"    # 👉 index only
    if f == [0,1,1,0,0]:     return "PEACE"    # ✌️
    if n == 5:                return "PALM"     # 🖐
    if f == [1,0,0,0,0]:     return "THUMBUP"  # 👍
    # Relaxed: thumb + index BOTH up — other fingers can be anything
    # (covers natural pinch where middle/ring/pinky aren't perfectly down)
    if f[0] == 1 and f[1] == 1:  return "VOLUME"  # 👌
    return None

# ─── STATE MACHINE ───────────────────────────────────────────
# Only ONE state is active at a time.
# A gesture must appear in STABILITY_FRAMES consecutive frames before activating.

STABILITY_FRAMES = 4        # frames gesture must be stable before activating
HOLD_FRAMES      = 14       # frames to hold a non-swipe gesture to fire it (~0.5s@30fps)
COOLDOWN_SEC     = 1.2      # min seconds between any two triggered actions

state            = "IDLE"   # IDLE | LOCKED
locked_gesture   = None     # the gesture currently locked in
stable_buf       = deque(maxlen=STABILITY_FRAMES)
hold_count       = 0        # frames held while in LOCKED
last_triggered   = 0.0

# Swipe tracking (only valid in LOCKED=="SWIPE")
swipe_x_hist     = deque(maxlen=20)   # (time, norm_x)

# Feedback
fb_text  = ""
fb_color = (255,255,255)
fb_t     = -999
FB_DUR   = 2.0

quit_stable_count = 0  # Stability for crossed-hands quit
show_help = True

def feedback(text, color=(80,255,180)):
    global fb_text, fb_color, fb_t
    fb_text, fb_color, fb_t = text, color, time.time()
    print(f"[GESTURE] {text}")

# ─── DRAW HELPERS ────────────────────────────────────────────
def rrect(img, p1, p2, col, a=0.55, r=12):
    ov = img.copy()
    x1,y1=p1; x2,y2=p2
    for pts in [((x1+r,y1),(x2-r,y2)),((x1,y1+r),(x2,y2-r))]:
        cv2.rectangle(ov, pts[0], pts[1], col, -1)
    for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov,(cx,cy),r,col,-1)
    cv2.addWeighted(ov,a,img,1-a,0,img)

def label(frame, text, pos, sc=0.52, th=1, fg=(255,255,255), bg=(20,20,20)):
    font=cv2.FONT_HERSHEY_SIMPLEX
    (tw,th2),bl=cv2.getTextSize(text,font,sc,th)
    x,y=pos
    rrect(frame,(x-5,y-th2-5),(x+tw+5,y+bl+3),bg,a=0.6,r=6)
    cv2.putText(frame,text,(x,y),font,sc,fg,th,cv2.LINE_AA)

def vol_bar(frame, vol, muted):
    bw,bh,bx,by=26,170,18,75
    rrect(frame,(bx-4,by-4),(bx+bw+4,by+bh+4),(15,15,15),a=0.65,r=6)
    fill=int(vol*bh)
    col=(80,80,220) if muted else (30,220,100)
    if fill>0:
        cv2.rectangle(frame,(bx,by+bh-fill),(bx+bw,by+bh),col,-1)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(160,160,160),1)
    lbl="MUTE" if muted else f"{int(vol*100)}%"
    cv2.putText(frame,lbl,(bx-2,by-8),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,"VOL",(bx+1,by+bh+14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1,cv2.LINE_AA)

def hold_bar(frame, label_txt, progress, W, H):
    bfw=200; bx=W//2-bfw//2; by=H-48
    rrect(frame,(bx-4,by-4),(bx+bfw+4,by+22),(15,15,15),a=0.6,r=6)
    cv2.rectangle(frame,(bx,by),(bx+int(progress*bfw),by+18),(0,210,160),-1)
    cv2.rectangle(frame,(bx,by),(bx+bfw,by+18),(140,140,140),1)
    cv2.putText(frame,f"Hold: {label_txt}",(bx,by-6),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,255,200),1,cv2.LINE_AA)

def draw_feedback(frame, now, W, H):
    if now-fb_t >= FB_DUR:
        return
    a=1.0-(now-fb_t)/FB_DUR
    font=cv2.FONT_HERSHEY_SIMPLEX; sc=1.2; th=3
    # Remove emojis from feedback text for better rendering
    clean_text = fb_text.replace("⏯","|").replace("🔇","Mute").replace("🔊","Vol").replace("🖐","50%").replace("👍","+20%").replace("⏭",">>").replace("⏮","<<")
    (tw,th2),_=cv2.getTextSize(clean_text,font,sc,th)
    x=(W-tw)//2; y=H//2+20
    rrect(frame,(x-18,y-th2-18),(x+tw+18,y+18),(10,10,10),a=0.65,r=12)
    c=tuple(int(v*a) for v in fb_color)
    cv2.putText(frame,clean_text,(x,y),font,sc,c,th,cv2.LINE_AA)

def draw_help(frame,W):
    rows=[
        ("Thumb+Index",     "Volume"),
        ("Index swipe R",    "Next track"),
        ("Index swipe L",    "Prev track"),
        ("Fist (hold)",     "Play/Pause"),
        ("Peace (hold)",    "Mute toggle"),
        ("Palm (hold)",     "Vol reset 50%"),
        ("Thumb (hold)",    "Vol +20%"),
        ("H=Help | X-Hands=Quit",""),
    ]
    px,py=W-290,12; rh=21
    rrect(frame,(px-8,py-8),(px+284,py+len(rows)*rh+18),(12,12,12),a=0.72,r=10)
    cv2.putText(frame,"GESTURE GUIDE",(px,py+13),cv2.FONT_HERSHEY_SIMPLEX,0.48,(100,210,255),1,cv2.LINE_AA)
    for i,(g,a) in enumerate(rows):
        y=py+13+(i+1)*rh
        cv2.putText(frame,g,   (px,y),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,255,200),1,cv2.LINE_AA)
        cv2.putText(frame,a,(px+168,y),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,220,90),1,cv2.LINE_AA)

# ─── MAIN LOOP ───────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]
    now   = time.time()

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_img)
    
    # ── Check for Quit Gesture (Crossed Hands) ────────────────
    if len(result.hand_landmarks) == 2:
        lbl1 = result.handedness[0][0].category_name
        lbl2 = result.handedness[1][0].category_name
        
        h1_x = result.hand_landmarks[0][0].x
        h2_x = result.hand_landmarks[1][0].x
        
        right_hand_x = h1_x if lbl1 == "Right" else h2_x
        left_hand_x  = h1_x if lbl1 == "Left" else h2_x
        
        # In the non-mirrored view (flipped frame 178):
        # Left hand is normally on the left (low x), Right hand on the right (high x).
        # CROSS = Left hand moves to the right of the Right hand.
        if left_hand_x > right_hand_x:
            quit_stable_count += 1
            if quit_stable_count > 15: # ~0.5s stability
                print("[QUIT] Hands crossed detected. Exiting...")
                break
        else:
            quit_stable_count = 0
    else:
        quit_stable_count = 0

    cooldown_ok = (now - last_triggered) > COOLDOWN_SEC

    # ── Detect current frame gesture ──────────────────────────
    if result.hand_landmarks:
        hand    = result.hand_landmarks[0]
        label_  = result.handedness[0][0].category_name
        current = classify(hand, label_)

        def px(i, h=hand):
            return (int(h[i].x*W), int(h[i].y*H))

        # ── Draw Landmarks for Feedback ───────────────────────
        for h_lms in result.hand_landmarks:
            for lmk in h_lms:
                cx, cy = int(lmk.x * W), int(lmk.y * H)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # ── Stability buffer: only lock in a gesture once it's
        #    been the same for STABILITY_FRAMES consecutive frames ─
        stable_buf.append(current)

        if state == "IDLE":
            # Require unanimous stability
            if (len(stable_buf) == STABILITY_FRAMES
                    and len(set(stable_buf)) == 1
                    and current is not None):
                state          = "LOCKED"
                locked_gesture = current
                hold_count     = 0
                swipe_x_hist.clear()

        elif state == "LOCKED":
            # If the gesture changes, drop back to IDLE so there's
            # always a clean transition (no accidental blends)
            if current != locked_gesture:
                # Grace: only exit if it's been different for 3+ frames
                if stable_buf.count(locked_gesture) < 3:
                    state          = "IDLE"
                    locked_gesture = None
                    stable_buf.clear()
                    swipe_x_hist.clear()
            else:
                hold_count += 1

        # ── Act on locked gesture ─────────────────────────────
        if state == "LOCKED":
            g = locked_gesture

            # --- VOLUME ---
            if g == "VOLUME":
                t = px(4); i8 = px(8)
                dist = np.hypot(i8[0]-t[0], i8[1]-t[1])
                vol  = float(np.interp(dist,[30,200],[0.0,1.0]))
                if not is_muted:
                    pulse.volume_set_all_chans(sink, vol)
                current_vol = vol
                mid=((t[0]+i8[0])//2,(t[1]+i8[1])//2)
                cv2.circle(frame,t,12,(255,80,220),-1)
                cv2.circle(frame,i8,12,(255,80,220),-1)
                cv2.line  (frame,t,i8,(255,80,220),3)
                cv2.circle(frame,mid,8,(0,235,255),-1)
                label(frame,f"VOL {int(vol*100)}%",(mid[0]+14,mid[1]-4),
                      fg=(0,235,255),sc=0.55)

            # --- SWIPE (index only) ---
            elif g == "SWIPE":
                ix_norm = hand[8].x
                swipe_x_hist.append((now, ix_norm))
                i8 = px(8)
                cv2.circle(frame, i8, 14, (0,200,255), -1)
                label(frame,"SWIPE MODE",(W//2-55,H-16),fg=(0,200,255),sc=0.52)

                if cooldown_ok and len(swipe_x_hist) >= 6:
                    ot, ox = swipe_x_hist[0]
                    dt = now - ot
                    dx = ix_norm - ox
                    speed = abs(dx) / max(dt, 0.001)
                    if dt < 0.55 and abs(dx) > 0.18 and speed > 0.35:
                        if dx > 0:
                            media("Next")
                            feedback("⏭  NEXT TRACK", (80,255,180))
                        else:
                            media("Previous")
                            feedback("⏮  PREV TRACK", (80,200,255))
                        last_triggered = now
                        swipe_x_hist.clear()

            # --- HOLD GESTURES (fist / peace / palm / thumbup) ---
            else:
                cfg = {
                    "FIST":    ("Play / Pause",  (255,220,60)),
                    "PEACE":   ("Mute toggle",   (255,100,100)),
                    "PALM":    ("Vol reset 50%", (160,255,100)),
                    "THUMBUP": ("Vol  +20%",     (60,255,220)),
                }
                if g in cfg:
                    txt, col = cfg[g]
                    progress = hold_count / HOLD_FRAMES
                    if progress < 1.0:
                        hold_bar(frame, txt, progress, W, H)
                    elif cooldown_ok:
                        # Fire the action
                        if g == "FIST":
                            media("PlayPause")
                            feedback("⏯  PLAY / PAUSE", col)
                        elif g == "PEACE":
                            is_muted = not is_muted
                            pulse.mute(sink, is_muted)
                            feedback("🔇  MUTED" if is_muted else "🔊  UNMUTED", col)
                        elif g == "PALM":
                            current_vol = 0.5
                            pulse.volume_set_all_chans(sink, 0.5)
                            if is_muted:
                                is_muted = False
                                pulse.mute(sink, False)
                            feedback("🖐  VOLUME → 50%", col)
                        elif g == "THUMBUP":
                            current_vol = min(1.0, current_vol + 0.2)
                            pulse.volume_set_all_chans(sink, current_vol)
                            feedback(f"👍  VOL +20%  →  {int(current_vol*100)}%", col)
                        last_triggered = now
                        # Reset so gesture must be released before firing again
                        state = "IDLE"
                        stable_buf.clear()

        # Show active mode badge
        if state == "LOCKED" and locked_gesture:
            badge_colors = {
                "VOLUME":"(255,80,220)","SWIPE":"(0,200,255)",
                "FIST":"(255,220,60)","PEACE":"(255,100,100)",
                "PALM":"(160,255,100)","THUMBUP":"(60,255,220)",
            }
            bc = eval(badge_colors.get(locked_gesture,"(200,200,200)"))
            label(frame, locked_gesture, (72,56), fg=bc, sc=0.48)

    else:
        # No hand → full reset
        state = "IDLE"
        stable_buf.clear()
        swipe_x_hist.clear()

    # ── HUD ──────────────────────────────────────────────────
    vol_bar(frame, current_vol, is_muted)
    status = "MUTED 🔇" if is_muted else f"VOL {int(current_vol*100)}%"
    label(frame, status, (72,30), sc=0.58, fg=(255,255,180))
    draw_feedback(frame, now, W, H)
    if show_help:
        draw_help(frame, W)

    cv2.imshow("Gesture Control", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        show_help = not show_help

cap.release()
cv2.destroyAllWindows()
pulse.close()
