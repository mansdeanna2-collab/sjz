# -*- coding: utf-8 -*-
"""
v21 平衡优化版 - 减少左右移动晃动 / 提升稳定性
升级版：启用头部锁定（类别2），强化双目标防晃动
"""
import os, sys, json, time, ctypes, threading
from ctypes import wintypes
from collections import deque, Counter

import numpy as np
import cv2
import dxcam
import mss
import torch
import tensorrt as trt
from torchvision.ops import nms

import win32gui, win32con, win32api

# ---------- ctypes ----------
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
_gdi32.CreateFontW.restype = wintypes.HFONT
_gdi32.CreateFontW.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
    wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
    wintypes.DWORD, wintypes.LPCWSTR
]
def create_font_ctypes(h=14, w=700, face="Consolas"):
    return _gdi32.CreateFontW(h,0,0,0,w,0,0,0,
        win32con.ANSI_CHARSET, win32con.OUT_DEFAULT_PRECIS,
        win32con.CLIP_DEFAULT_PRECIS, win32con.CLEARTYPE_QUALITY,
        win32con.FF_DONTCARE|win32con.DEFAULT_PITCH, face)
_user32 = ctypes.WinDLL("user32", use_last_error=True)


# ==================== 用户配置 ====================
ENGINE_PATH   = r"C:\Users\Administrator\PyCharmMiscProject\runs\detect\my_experiment39\weights\Dawan.engine"
IMG_SIZE, CENTER_REGION, NUM_CLASSES = 256, 256, 4

CONF_KEEP, CONF_NEW = 0.1, 0.55
CONF_DISPLAY_ON, CONF_DISPLAY_OFF = 0.60, 0.35
DISPLAY_STICKY = 6
IOU_THRES = 0.45
TARGET_CLASS = None

TRK_IOU_MATCH_BASE, TRK_IOU_MIN = 0.25, 0.08
TRK_SIZE_REF = 80.0
TRK_MAX_LOST, TRK_MIN_HITS, TRK_SHOW_LOST = 8, 2, 3
TRK_CONF_DECAY, TRK_MAX_VEL = 0.90, 20.0
CONF_SMOOTH_ALPHA = 0.2
EMA_ALPHA_MIN, EMA_ALPHA_MAX = 0.15, 0.70
EMA_MOTION_LOW, EMA_MOTION_HIGH = 3.0, 20.0

MAX_INFER_FPS = 300
CAPTURE_FPS = 200
LOG_INTERVAL = 2.0

SHOW_OVERLAY, OVERLAY_FPS = True, 120
DRAW_CROSSHAIR, DRAW_STATS, DRAW_FOV_CIRCLE = True, True, False
DRAW_BOXES = True
STATS_UPDATE_MS = 100

# ==================== 瞄准配置 (头部锁定 + 防双目标晃动) ====================
AIM_ENABLE = True
AIM_HOTKEYS  = ["RBUTTON"]
AIM_FIRE_KEY = "LBUTTON"

AIM_FIRE_TRIGGERS_AIM = True
AIM_FIRE_ONLY_FOV_MULT = 0.9
AIM_FIRE_ONLY_STRENGTH = 0.95

AIM_CLASSES = [0]                     # 身体类别
AIM_HEAD_CLASS = 2                    # 头部类别（启用）
AIM_OFFSETS = {0: 0.30}               # 身体瞄准偏移（胸部）
AIM_DEFAULT_OFFSET = 0.30
AIM_FOV_RADIUS = 200                  # 略微缩小，减少干扰

# P控制 - 降低过冲
AIM_KP_X = 0.18
AIM_KP_Y = 0.16
AIM_FIRE_KP_X_MULT = 1.35
AIM_FIRE_KP_Y_MULT = 1.90

# 快速锁定
AIM_ACQUIRE_BOOST_MS = 300
AIM_ACQUIRE_KP_MULT  = 2.2
AIM_ACQUIRE_CAP_MULT = 2.0

# 输出平滑 - 增加稳定性
AIM_OUTPUT_SMOOTH = 0.25

# 瞄准点平滑
AIM_POINT_SMOOTH_NEW  = 0.45
AIM_POINT_SMOOTH_FIRE = 0.35

# 锁定位置平滑
AIM_LOCK_SMOOTH_ALPHA = 0.90          # 提高平滑，减少跳动

# 步长 - 适度减小
AIM_STEP_CAP_NEAR = 16
AIM_STEP_CAP_FAR  = 36
AIM_STEP_CAP_FIRE_Y = 24
AIM_NEAR_DIST = 35
AIM_FAR_DIST  = 150

# 死区 / 抖动
AIM_DEAD_ZONE = 2
AIM_JITTER_THRESH = 10
AIM_JITTER_COOLDOWN_MS = 35
AIM_JITTER_DISABLE_ON_FIRE = True
AIM_MIN_STEP = 1
AIM_DEBUG = True

# 过冲刹车
AIM_OVERSHOOT_BRAKE = True
AIM_BRAKE_FACTOR = 0.35
AIM_BRAKE_FIRE_X_FACTOR = 0.25

# 压枪
AIM_ANTI_RECOIL = True
AIM_RECOIL_Y_BASE = 1.2
AIM_RECOIL_Y_MAX  = 4.5
AIM_RECOIL_X      = 0.0
AIM_RECOIL_START_DELAY_MS = 30

# 目标评分
AIM_SCORE_DIST_WEIGHT = 0.6
AIM_SCORE_SIZE_BONUS  = 130
AIM_SCORE_CENTER_BONUS = 110

# 位置锁 - 强化粘性，防止双目标摇晃
AIM_POS_LOCK = True
AIM_LOCK_BASE_TOL    = 360
AIM_LOCK_MOVE_FACTOR = 25.0
AIM_LOCK_FIRE_BONUS  = 4.0
AIM_LOCK_STICKY_MS   = 3000           # 延长粘性时间
AIM_LOCK_FIRE_NO_SWITCH = True
AIM_LOCK_SIZE_WEIGHT = 0.08
AIM_LOCK_SCORE_BIAS  = -200.0         # 评分偏置，保留锁定目标

# 目标速度 - 降低激进程度
AIM_TARGET_VEL_TRACK  = True
AIM_TARGET_VEL_TOL    = 0.20
AIM_TARGET_VEL_ALPHA  = 0.90
AIM_TARGET_PRED_MS    = 30            # 减小预测时间，避免过冲
AIM_TARGET_VEL_MAX    = 600
AIM_PRED_DISABLE_ON_FIRE = True

# 滑行
AIM_COAST_MS = 120
AIM_COAST_DECAY = 0.65

# 限速
AIM_SEND_RATE_HZ = 120

AIM_CALIB_KEYS = ["9"]
AIM_PROBE_KEYS = ["F2"]
AIM_MOUSE_DRIVER = "logitech"

SET_AFFINITY, HIGH_PRIORITY = True, True
AFFINITY_CPUS = [6, 7]
DEVICE = torch.device("cuda:0")
# =================================================

VK_CODES = {
    "LBUTTON":0x01,"RBUTTON":0x02,"MBUTTON":0x04,"XBUTTON1":0x05,"XBUTTON2":0x06,
    "LSHIFT":0xA0,"RSHIFT":0xA1,"LCTRL":0xA2,"RCTRL":0xA3,"LALT":0xA4,"RALT":0xA5,
    "CAPS":0x14,"F1":0x70,"F2":0x71,"F3":0x72,"F4":0x73,
    "9":0x39,
}
def is_key_down(key):
    vk=VK_CODES.get(str(key).upper())
    if vk is None: return False
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)
def is_any_key_down(keys): return any(is_key_down(k) for k in keys)


# ==================== 鼠标驱动 ====================
class MOUSEINPUT(ctypes.Structure):
    _fields_=[("dx",ctypes.c_long),("dy",ctypes.c_long),
              ("mouseData",wintypes.DWORD),("dwFlags",wintypes.DWORD),
              ("time",wintypes.DWORD),("dwExtraInfo",ctypes.POINTER(wintypes.ULONG))]
class _IU(ctypes.Union): _fields_=[("mi",MOUSEINPUT)]
class INPUT(ctypes.Structure):
    _anonymous_=("u",); _fields_=[("type",wintypes.DWORD),("u",_IU)]
MOUSEEVENTF_MOVE=0x0001; INPUT_MOUSE=0
_user32.SendInput.argtypes=[wintypes.UINT,ctypes.POINTER(INPUT),ctypes.c_int]
_user32.SendInput.restype=wintypes.UINT

class MouseDriver:
    def __init__(self, kind="sendinput"):
        self.kind=kind.lower(); self._lg=None
        if self.kind=="logitech": self._init_logitech()
        elif self.kind=="makcu": self._init_makcu()
        elif self.kind=="kmbox": self._init_kmbox()
        print(f"[mouse] driver = {self.kind}")
    def move_rel(self,dx,dy):
        if dx==0 and dy==0: return
        if self.kind=="sendinput": self._send_sendinput(int(dx),int(dy))
        elif self.kind=="logitech": self._send_logitech(int(dx),int(dy))
        elif self.kind=="makcu": self._send_makcu(int(dx),int(dy))
        elif self.kind=="kmbox": self._send_kmbox(int(dx),int(dy))
    def close(self): pass
    def _send_sendinput(self,dx,dy):
        inp=INPUT(); inp.type=INPUT_MOUSE
        inp.mi=MOUSEINPUT(dx,dy,0,MOUSEEVENTF_MOVE,0,None)
        _user32.SendInput(1,ctypes.byref(inp),ctypes.sizeof(INPUT))
    def _init_logitech(self):
        for p in [r"DLLs\LGmouseControl\MouseControl.dll", r"MouseControl.dll",
                  r"C:\Users\Administrator\PyCharmMiscProject\DLLs\LGmouseControl\MouseControl.dll"]:
            try: self._lg=ctypes.CDLL(p); print(f"[mouse] loaded: {p}"); break
            except OSError: continue
        if self._lg is None:
            print("[mouse] ❌ fallback"); self.kind="sendinput"; return
        if not all(hasattr(self._lg,fn) for fn in ["move_R","click_Left_down","click_Left_up"]):
            print("[mouse] ❌ DLL missing"); self.kind="sendinput"; self._lg=None; return
        self._lg.move_R.argtypes=[ctypes.c_int,ctypes.c_int]
        self._lg.move_R.restype=ctypes.c_int
        print("[mouse] logitech ready")
    def _send_logitech(self,dx,dy):
        try: self._lg.move_R(int(dx),int(dy))
        except Exception as e: print(f"[mouse] fail: {e}")
    def _init_makcu(self):
        try:
            import serial; self._mk=serial.Serial("COM5",115200,timeout=0.01)
        except Exception as e: print(f"[mouse] makcu fail: {e}"); self.kind="sendinput"
    def _send_makcu(self,dx,dy): self._mk.write(f"km.move({dx},{dy})\r\n".encode())
    def _init_kmbox(self):
        try:
            import serial; self._km=serial.Serial("COM6",115200,timeout=0.01)
        except Exception as e: print(f"[mouse] kmbox fail: {e}"); self.kind="sendinput"
    def _send_kmbox(self,dx,dy): self._km.write(f"km.move({dx},{dy})\n".encode())


# ==================== Aimbot (头部锁定 + 防双目标晃动) ====================
class Aimbot:
    def __init__(self, mouse, region_xyxy):
        self.mouse = mouse
        self.region = region_xyxy
        self.crosshair = ((region_xyxy[0]+region_xyxy[2])//2,
                          (region_xyxy[1]+region_xyxy[3])//2)
        self._debug_cnt = 0; self._probe_cnt = 0; self._calib_cnt = 0
        self._was_pressed = False
        self._press_start_t = 0.0

        self._locked_cx = None; self._locked_cy = None
        self._locked_w  = 0.0;  self._locked_h  = 0.0
        self._locked_cls = None
        self._last_seen_t = 0.0

        self._aim_x_sm = None; self._aim_y_sm = None
        self._target_vx = 0.0; self._target_vy = 0.0
        self._last_err_x = 0.0; self._last_err_y = 0.0
        self._has_last_err = False
        self._last_mx_raw = 0.0; self._last_my_raw = 0.0
        self._last_mx = 0.0; self._last_my = 0.0

        self._next_send_t = 0.0
        self._send_interval = 1.0 / max(1, AIM_SEND_RATE_HZ)

        self._fire_start_t = 0.0
        self._was_firing = False
        self._fire_frames = 0

        self._last_send_t = 0.0

        self._coast_mx = 0.0; self._coast_my = 0.0
        self._coast_until = 0.0

        self.aim_point = None
        self.locked_box_screen = None

    def _aim_point_of(self, box, cls):
        k = int(cls)
        x1, y1, x2, y2 = box
        ax = (x1 + x2) * 0.5 + self.region[0]
        if AIM_HEAD_CLASS is not None and k == AIM_HEAD_CLASS:
            ay = (y1 + y2) * 0.5 + self.region[1]
        else:
            offset = AIM_OFFSETS.get(k, AIM_DEFAULT_OFFSET)
            ay = y1 + (y2 - y1) * offset + self.region[1]
        return ax, ay

    def _screen_box(self, box):
        return (box[0]+self.region[0], box[1]+self.region[1],
                box[2]+self.region[0], box[3]+self.region[1])

    def _pick_target(self, boxes, confs, clss, now, firing, fov_radius):
        if len(boxes) == 0: return None
        cx_s, cy_s = self.crosshair
        cands = []
        fov_sq = fov_radius * fov_radius

        head_cands = []
        body_cands = []

        for b, c, k in zip(boxes, confs, clss):
            k_i = int(k)
            if k_i not in AIM_CLASSES and (AIM_HEAD_CLASS is None or k_i != AIM_HEAD_CLASS):
                continue
            sx1 = b[0]+self.region[0]; sy1 = b[1]+self.region[1]
            sx2 = b[2]+self.region[0]; sy2 = b[3]+self.region[1]
            scx = (sx1+sx2)*0.5; scy = (sy1+sy2)*0.5
            sw = sx2-sx1; sh = sy2-sy1
            ax, ay = self._aim_point_of(b, k_i)
            d2 = (ax-cx_s)**2 + (ay-cy_s)**2
            if d2 > fov_sq: continue
            cand = (d2, b, k_i, scx, scy, sw, sh, ax, ay)
            if AIM_HEAD_CLASS is not None and k_i == AIM_HEAD_CLASS:
                head_cands.append(cand)
            else:
                body_cands.append(cand)

        primary_cands = head_cands if head_cands else body_cands
        if not primary_cands:
            return None

        # ---- 位置锁：优先保留当前锁定目标 ----
        if AIM_POS_LOCK and self._locked_cx is not None:
            age_ms = (now - self._last_seen_t) * 1000.0
            if age_ms < AIM_LOCK_STICKY_MS:
                dt = max(now - self._last_seen_t, 0.0)
                if firing and AIM_PRED_DISABLE_ON_FIRE:
                    pred_cx = self._locked_cx; pred_cy = self._locked_cy
                else:
                    pred_cx = self._locked_cx + self._target_vx * dt
                    pred_cy = self._locked_cy + self._target_vy * dt
                move_mag = abs(self._last_mx) + abs(self._last_my)
                vel_mag  = abs(self._target_vx) + abs(self._target_vy)
                tol = (AIM_LOCK_BASE_TOL
                       + move_mag * AIM_LOCK_MOVE_FACTOR
                       + vel_mag * AIM_TARGET_VEL_TOL)
                if firing: tol *= AIM_LOCK_FIRE_BONUS

                best = None; best_s = float('inf')
                for c in primary_cands:
                    _d2, b, k, scx, scy, sw, sh, ax, ay = c
                    if k != self._locked_cls: continue
                    cd = float(np.hypot(scx - pred_cx, scy - pred_cy))
                    if cd > tol: continue
                    sz = abs(sw - self._locked_w) + abs(sh - self._locked_h)
                    score = cd + sz * AIM_LOCK_SIZE_WEIGHT
                    if score < best_s:
                        best_s = score; best = c
                if best is not None:
                    return best
                if firing and AIM_LOCK_FIRE_NO_SWITCH:
                    return None

        # ---- 评分选择，对锁定目标施加偏置 ----
        best = None; best_s = float('inf')
        for c in primary_cands:
            d2, b, k, scx, scy, sw, sh, ax, ay = c
            dist = float(np.sqrt(d2))
            size = min(sw, sh)
            size_bonus = min(size, AIM_SCORE_SIZE_BONUS)
            center_bonus = max(0.0, AIM_SCORE_CENTER_BONUS - dist * 0.5)
            score = dist * AIM_SCORE_DIST_WEIGHT - size_bonus * 0.3 - center_bonus

            # 如果存在锁定目标且类别相同，且候选与锁定目标接近，则给予额外偏置（降低分数=更容易被选中）
            if (AIM_POS_LOCK and self._locked_cx is not None and k == self._locked_cls):
                dt = max(now - self._last_seen_t, 0.0)
                if firing and AIM_PRED_DISABLE_ON_FIRE:
                    pred_cx = self._locked_cx; pred_cy = self._locked_cy
                else:
                    pred_cx = self._locked_cx + self._target_vx * dt
                    pred_cy = self._locked_cy + self._target_vy * dt
                cd = float(np.hypot(scx - pred_cx, scy - pred_cy))
                if cd < AIM_LOCK_BASE_TOL * 0.8:   # 足够近
                    score += AIM_LOCK_SCORE_BIAS    # 负值使分数更低，优先保留

            if score < best_s:
                best_s = score; best = c
        return best

    def _probe(self, clss):
        if not is_any_key_down(AIM_PROBE_KEYS): return
        self._probe_cnt += 1
        if self._probe_cnt % 30 != 0: return
        cnt = Counter(int(k) for k in clss)
        print(f"[probe] 类别: {dict(cnt)} 总数 {len(clss)}")

    def _recoil_y(self):
        frames = self._fire_frames
        if frames < 15:
            return AIM_RECOIL_Y_BASE + frames * 0.08
        elif frames < 40:
            return AIM_RECOIL_Y_BASE + 1.2 + (frames - 15) * 0.03
        else:
            return min(AIM_RECOIL_Y_BASE + 1.95 + (frames - 40) * 0.008, AIM_RECOIL_Y_MAX)

    def tick(self, boxes, confs, clss, ids):
        self._probe(clss)
        calibrating = is_any_key_down(AIM_CALIB_KEYS)
        ads_pressed = is_any_key_down(AIM_HOTKEYS)
        firing = is_key_down(AIM_FIRE_KEY)

        fire_aim = firing and AIM_FIRE_TRIGGERS_AIM
        aim_pressed = ads_pressed or fire_aim or calibrating
        fire_only_mode = fire_aim and not ads_pressed and not calibrating

        if not AIM_ENABLE or not aim_pressed:
            if self._was_pressed: self._reset_state()
            self._was_pressed = False
            self._was_firing = False
            self._fire_frames = 0
            return False

        now = time.perf_counter()
        if not self._was_pressed:
            self._reset_state()
            self._press_start_t = now
        self._was_pressed = True

        if firing:
            if not self._was_firing:
                self._fire_start_t = now
                self._fire_frames = 0
            else:
                self._fire_frames += 1
        else:
            self._fire_frames = 0
        self._was_firing = firing

        fov_r = AIM_FOV_RADIUS
        if fire_only_mode:
            fov_r = int(fov_r * AIM_FIRE_ONLY_FOV_MULT)

        pick = self._pick_target(boxes, confs, clss, now, firing, fov_r)

        recoil_x = 0.0
        recoil_y = 0.0

        if pick is None:
            send_x = 0.0
            send_y = 0.0
            if now < self._coast_until and (abs(self._coast_mx) > 0 or abs(self._coast_my) > 0):
                cmx = self._coast_mx * AIM_COAST_DECAY
                cmy = self._coast_my * AIM_COAST_DECAY
                self._coast_mx = cmx
                self._coast_my = cmy
                send_x += cmx
                send_y += cmy
            if send_x != 0 or send_y != 0:
                self._maybe_send(now, send_x, send_y, "coast", force=False)
            self.aim_point = None
            self.locked_box_screen = None
            return False

        if AIM_ANTI_RECOIL and firing:
            fire_age_ms = (now - self._fire_start_t) * 1000.0
            if fire_age_ms >= AIM_RECOIL_START_DELAY_MS:
                recoil_x = AIM_RECOIL_X
                recoil_y = self._recoil_y()

        d2, box, k, scx, scy, sw, sh, ax, ay = pick

        # 更新目标速度
        if (AIM_TARGET_VEL_TRACK
            and self._locked_cx is not None
            and self._locked_cls == k
            and not firing):
            dt = max(now - self._last_seen_t, 1e-3)
            if dt < 0.2:
                vx_new = (scx - self._locked_cx) / dt
                vy_new = (scy - self._locked_cy) / dt
                if (abs(vx_new) < AIM_TARGET_VEL_MAX
                    and abs(vy_new) < AIM_TARGET_VEL_MAX):
                    a = AIM_TARGET_VEL_ALPHA
                    self._target_vx = (1-a) * self._target_vx + a * vx_new
                    self._target_vy = (1-a) * self._target_vy + a * vy_new
                else:
                    self._target_vx *= 0.7; self._target_vy *= 0.7
            else:
                self._target_vx = 0.0; self._target_vy = 0.0
        elif firing:
            self._target_vx *= 0.85; self._target_vy *= 0.85
        else:
            self._target_vx = 0.0; self._target_vy = 0.0

        # 更新锁定位置
        if (self._locked_cx is not None and self._locked_cls == k
            and (now - self._last_seen_t) * 1000.0 < AIM_LOCK_STICKY_MS):
            a = AIM_LOCK_SMOOTH_ALPHA
            self._locked_cx = a * scx + (1-a) * self._locked_cx
            self._locked_cy = a * scy + (1-a) * self._locked_cy
            self._locked_w  = a * sw  + (1-a) * self._locked_w
            self._locked_h  = a * sh  + (1-a) * self._locked_h
        else:
            self._locked_cx = scx; self._locked_cy = scy
            self._locked_w = sw; self._locked_h = sh
            self._aim_x_sm = None; self._aim_y_sm = None
        self._locked_cls = k
        self._last_seen_t = now

        press_age_ms = (now - self._press_start_t) * 1000.0
        acquiring = press_age_ms < AIM_ACQUIRE_BOOST_MS
        if firing and AIM_PRED_DISABLE_ON_FIRE:
            aim_x_raw = ax; aim_y_raw = ay
        else:
            if acquiring:
                aim_x_raw = ax; aim_y_raw = ay
            else:
                pred_dt = AIM_TARGET_PRED_MS / 1000.0
                aim_x_raw = ax + self._target_vx * pred_dt
                aim_y_raw = ay + self._target_vy * pred_dt

        if self._aim_x_sm is None:
            self._aim_x_sm = aim_x_raw; self._aim_y_sm = aim_y_raw
        else:
            sm_new = AIM_POINT_SMOOTH_FIRE if firing else AIM_POINT_SMOOTH_NEW
            self._aim_x_sm = sm_new * aim_x_raw + (1-sm_new) * self._aim_x_sm
            self._aim_y_sm = sm_new * aim_y_raw + (1-sm_new) * self._aim_y_sm

        aim_x = self._aim_x_sm; aim_y = self._aim_y_sm
        self.aim_point = (aim_x, aim_y)
        self.locked_box_screen = self._screen_box(box)

        cx, cy = self.crosshair
        ex = aim_x - cx; ey = aim_y - cy
        dist = float(np.hypot(ex, ey))

        if dist <= AIM_DEAD_ZONE:
            self._last_err_x = ex; self._last_err_y = ey
            self._has_last_err = True
            if recoil_x != 0 or recoil_y != 0:
                self._maybe_send(now, recoil_x, recoil_y, "recoil-dead", force=True)
            return True

        if (dist <= AIM_JITTER_THRESH
            and not (firing and AIM_JITTER_DISABLE_ON_FIRE)):
            since_ms = (now - self._last_send_t) * 1000.0
            if since_ms < AIM_JITTER_COOLDOWN_MS:
                self._last_err_x = ex; self._last_err_y = ey
                self._has_last_err = True
                if recoil_x != 0 or recoil_y != 0:
                    self._maybe_send(now, recoil_x, recoil_y, "recoil-jit", force=True)
                return True

        boost_kp  = AIM_ACQUIRE_KP_MULT if acquiring else 1.0
        boost_cap = AIM_ACQUIRE_CAP_MULT if acquiring else 1.0
        mode_mult = AIM_FIRE_ONLY_STRENGTH if fire_only_mode else 1.0

        kp_x = AIM_KP_X * (AIM_FIRE_KP_X_MULT if firing else 1.0) * boost_kp * mode_mult
        kp_y = AIM_KP_Y * (AIM_FIRE_KP_Y_MULT if firing else 1.0) * boost_kp * mode_mult
        mx_raw = kp_x * ex
        my_raw = kp_y * ey

        aim_mx = (1.0 - AIM_OUTPUT_SMOOTH) * mx_raw + AIM_OUTPUT_SMOOTH * self._last_mx_raw
        aim_my = (1.0 - AIM_OUTPUT_SMOOTH) * my_raw + AIM_OUTPUT_SMOOTH * self._last_my_raw

        if AIM_OVERSHOOT_BRAKE and self._has_last_err:
            if self._last_err_x * ex < 0:
                brake_x = AIM_BRAKE_FIRE_X_FACTOR if firing else AIM_BRAKE_FACTOR
                aim_mx *= brake_x
            if self._last_err_y * ey < 0:
                aim_my *= AIM_BRAKE_FACTOR

        self._coast_mx = aim_mx
        self._coast_my = aim_my
        self._coast_until = now + AIM_COAST_MS / 1000.0

        mx = aim_mx + recoil_x
        my = aim_my + recoil_y

        if dist <= AIM_NEAR_DIST:
            cap_base = AIM_STEP_CAP_NEAR
        elif dist >= AIM_FAR_DIST:
            cap_base = AIM_STEP_CAP_FAR
        else:
            t = (dist - AIM_NEAR_DIST) / (AIM_FAR_DIST - AIM_NEAR_DIST)
            cap_base = AIM_STEP_CAP_NEAR + (AIM_STEP_CAP_FAR - AIM_STEP_CAP_NEAR) * t
        cap_base *= boost_cap

        cap_x = cap_base
        cap_y = max(cap_base, AIM_STEP_CAP_FIRE_Y) if firing else cap_base

        mx = max(-cap_x, min(cap_x, mx))
        my = max(-cap_y, min(cap_y, my))

        if 0 < abs(mx) < AIM_MIN_STEP: mx = AIM_MIN_STEP if mx > 0 else -AIM_MIN_STEP
        if 0 < abs(my) < AIM_MIN_STEP: my = AIM_MIN_STEP if my > 0 else -AIM_MIN_STEP

        self._last_err_x = ex; self._last_err_y = ey
        self._has_last_err = True
        self._last_mx_raw = mx_raw; self._last_my_raw = my_raw

        if calibrating:
            self._calib_cnt += 1
            if self._calib_cnt % 5 == 0:
                mode = "ADS" if ads_pressed else ("FIRE" if fire_only_mode else "CAL")
                print(f"[calib] err=({ex:+6.1f},{ey:+6.1f}) d={dist:5.1f} "
                      f"tv=({self._target_vx:+5.0f},{self._target_vy:+5.0f}) "
                      f"rec={recoil_y:.2f} cap=({cap_x:.0f},{cap_y:.0f}) "
                      f"would=({int(mx):+3d},{int(my):+3d}) fire={int(firing)}")
            return True

        mode_tag = "ads" if ads_pressed else ("fire" if fire_only_mode else "")
        self._maybe_send(now, mx, my,
                         f"err=({ex:+6.1f},{ey:+6.1f}) d={dist:5.1f} "
                         f"cap=({cap_x:.0f},{cap_y:.0f}) "
                         f"rec={recoil_y:.1f} [{mode_tag}] "
                         f"fire={int(firing)} strk={self._fire_frames}")
        self._last_mx = mx; self._last_my = my
        return True

    def _maybe_send(self, now, mx, my, tag="", force=False):
        if not force and now < self._next_send_t: return
        self._next_send_t = now + self._send_interval
        if AIM_DEBUG:
            self._debug_cnt += 1
            if self._debug_cnt % 5 == 0:
                print(f"[aim] {tag} send=({int(mx):+3d},{int(my):+3d})")
        if int(mx) == 0 and int(my) == 0: return
        self._last_send_t = now
        self.mouse.move_rel(mx, my)

    def _reset_state(self):
        self._locked_cx = None; self._locked_cy = None
        self._locked_w = 0.0; self._locked_h = 0.0
        self._locked_cls = None
        self._last_seen_t = 0.0
        self._aim_x_sm = None; self._aim_y_sm = None
        self._target_vx = 0.0; self._target_vy = 0.0
        self._last_mx = 0.0; self._last_my = 0.0
        self._last_mx_raw = 0.0; self._last_my_raw = 0.0
        self._last_err_x = 0.0; self._last_err_y = 0.0
        self._has_last_err = False
        self._next_send_t = 0.0
        self._last_send_t = 0.0
        self._fire_start_t = 0.0
        self._fire_frames = 0
        self._press_start_t = 0.0
        self._coast_mx = 0.0; self._coast_my = 0.0
        self._coast_until = 0.0
        self.aim_point = None
        self.locked_box_screen = None

    def reset(self):
        self._reset_state()
        self._was_pressed = False
        self._was_firing = False


# ==================== 进程 ====================
def tune_process():
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if SET_AFFINITY: p.cpu_affinity(AFFINITY_CPUS)
        if HIGH_PRIORITY and sys.platform == "win32":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        print(f"[proc] affinity={p.cpu_affinity()} nice={p.nice()}")
    except Exception as e: print(f"[proc] skip: {e}")


# ==================== TensorRT ====================
class TRTEngine:
    _NP2TORCH={np.float32:torch.float32,np.float16:torch.float16,
               np.int32:torch.int32,np.int8:torch.int8,
               np.uint8:torch.uint8,np.bool_:torch.bool}
    def __init__(self, engine_path, device):
        self.device=device
        self.logger=trt.Logger(trt.Logger.WARNING)
        self.names_map={}
        print(f"[trt] runtime: {trt.__version__} loading")
        with open(engine_path,"rb") as f:
            meta_len=int.from_bytes(f.read(4),byteorder="little")
            if 0<meta_len<10_000:
                try:
                    meta=json.loads(f.read(meta_len).decode("utf-8"))
                    self.names_map=meta.get("names",{})
                    print(f"[trt] names={self.names_map}")
                except Exception: f.seek(0)
            else: f.seek(0)
            engine_bytes=f.read()
        with trt.Runtime(self.logger) as rt:
            self.engine=rt.deserialize_cuda_engine(engine_bytes)
        self.context=self.engine.create_execution_context()
        self.stream=torch.cuda.Stream(device=device)
        self.io={}
        for i in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(i)
            shape=tuple(self.engine.get_tensor_shape(name))
            np_dtype=trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype=self._NP2TORCH[np.dtype(np_dtype).type]
            is_input=(self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT)
            buf=torch.empty(shape,dtype=torch_dtype,device=device).contiguous()
            self.io[name]={"buf":buf,"is_input":is_input,"shape":shape,"dtype":torch_dtype}
            self.context.set_tensor_address(name,buf.data_ptr())
            print(f"[trt] {'IN ' if is_input else 'OUT'} {name} {shape}")
        self.input_name=next(n for n,v in self.io.items() if v["is_input"])
        self.output_names=[n for n,v in self.io.items() if not v["is_input"]]
    @torch.inference_mode()
    def infer(self,gi):
        self.io[self.input_name]["buf"].copy_(gi,non_blocking=True)
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return {n:self.io[n]["buf"] for n in self.output_names}


class Preprocessor:
    def __init__(self,size,device,dtype=torch.float32):
        self.size=size; self.device=device; self.dtype=dtype
        self.pinned=torch.empty((size,size,3),dtype=torch.uint8,pin_memory=True)
        self.gpu_in=torch.empty((1,3,size,size),dtype=dtype,device=device)
    @torch.inference_mode()
    def __call__(self,frame):
        if frame.shape[0]!=self.size or frame.shape[1]!=self.size:
            frame=cv2.resize(frame,(self.size,self.size),interpolation=cv2.INTER_LINEAR)
        np.copyto(self.pinned.numpy(),frame)
        gpu_u8=self.pinned.to(self.device,non_blocking=True)
        gpu=gpu_u8[...,[2,1,0]].permute(2,0,1).unsqueeze(0)
        self.gpu_in.copy_(gpu.to(self.dtype).mul_(1.0/255.0),non_blocking=True)
        return self.gpu_in


@torch.inference_mode()
def postprocess(raw,conf_thres,iou_thres,nc,target_class=None):
    pred=raw[0].transpose(0,1)
    boxes_xywh=pred[:,:4]; cls_scores=pred[:,4:4+nc]
    conf,cls=cls_scores.max(dim=1)
    mask=conf>conf_thres
    if target_class is not None: mask&=(cls==target_class)
    if not mask.any():
        return (np.empty((0,4),np.float32),np.empty((0,),np.float32),np.empty((0,),np.int32))
    boxes_xywh=boxes_xywh[mask]; conf=conf[mask]; cls=cls[mask]
    xy=boxes_xywh[:,:2]; wh=boxes_xywh[:,2:]
    xyxy=torch.cat([xy-wh*0.5,xy+wh*0.5],dim=1).float()
    keep=nms(xyxy,conf.float(),iou_thres)
    return (xyxy[keep].cpu().numpy(),conf[keep].cpu().numpy(),
            cls[keep].cpu().numpy().astype(np.int32))


def iou_matrix(a,b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)),np.float32)
    a=a[:,None,:]; b=b[None,:,:]
    ix1=np.maximum(a[...,0],b[...,0]); iy1=np.maximum(a[...,1],b[...,1])
    ix2=np.minimum(a[...,2],b[...,2]); iy2=np.minimum(a[...,3],b[...,3])
    iw=np.clip(ix2-ix1,0,None); ih=np.clip(iy2-iy1,0,None)
    inter=iw*ih
    aa=(a[...,2]-a[...,0])*(a[...,3]-a[...,1])
    ab=(b[...,2]-b[...,0])*(b[...,3]-b[...,1])
    return inter/(aa+ab-inter+1e-6)

def box_size(b): return float(min(b[2]-b[0],b[3]-b[1]))
def adaptive_iou_threshold(b):
    s=box_size(b); r=min(1.0,s/TRK_SIZE_REF)
    return TRK_IOU_MIN+(TRK_IOU_MATCH_BASE-TRK_IOU_MIN)*r
def adaptive_ema_alpha(m):
    if m<=EMA_MOTION_LOW: return EMA_ALPHA_MIN
    if m>=EMA_MOTION_HIGH: return EMA_ALPHA_MAX
    t=(m-EMA_MOTION_LOW)/(EMA_MOTION_HIGH-EMA_MOTION_LOW)
    return EMA_ALPHA_MIN+t*(EMA_ALPHA_MAX-EMA_ALPHA_MIN)


class Track:
    __slots__=("id","box","cls","conf","conf_smooth","hits","lost","age","vel","displayed","sticky_until")
    def __init__(self,tid,box,cls,conf):
        self.id=tid; self.box=box.astype(np.float32); self.cls=int(cls)
        self.conf=float(conf); self.conf_smooth=float(conf)
        self.hits=1; self.lost=0; self.age=1
        self.vel=np.zeros(4,np.float32); self.displayed=False; self.sticky_until=0
    def update(self,box,conf):
        nb=box.astype(np.float32); diff=nb-self.box
        m=np.hypot((diff[0]+diff[2])*0.5,(diff[1]+diff[3])*0.5)
        alpha=adaptive_ema_alpha(m)
        v=0.5*self.vel+0.5*diff
        self.vel=np.clip(v,-TRK_MAX_VEL,TRK_MAX_VEL)
        self.box=alpha*nb+(1-alpha)*self.box
        self.conf=float(conf)
        self.conf_smooth=CONF_SMOOTH_ALPHA*float(conf)+(1-CONF_SMOOTH_ALPHA)*self.conf_smooth
        self.hits+=1; self.lost=0; self.age+=1
    def mark_lost(self):
        self.box=self.box+self.vel; self.conf_smooth*=TRK_CONF_DECAY
        self.lost+=1; self.age+=1
    def confirmed(self): return self.hits>=TRK_MIN_HITS


class IoUTracker:
    def __init__(self): self._next_id=1; self.tracks=[]
    def update(self,det_xyxy,det_conf,det_cls):
        if len(self.tracks)==0:
            for b,c,k in zip(det_xyxy,det_conf,det_cls):
                if c>=CONF_NEW:
                    self.tracks.append(Track(self._next_id,b,k,c)); self._next_id+=1
        elif len(det_xyxy)==0:
            for t in self.tracks: t.mark_lost()
        else:
            tb=np.stack([t.box for t in self.tracks],axis=0)
            iou=iou_matrix(tb,det_xyxy)
            tc=np.array([t.cls for t in self.tracks])[:,None]
            iou=np.where(tc==det_cls[None,:],iou,0.0)
            thr=np.array([adaptive_iou_threshold(t.box) for t in self.tracks],dtype=np.float32)
            mt,md=set(),set()
            order=np.dstack(np.unravel_index(np.argsort(-iou,axis=None),iou.shape))[0]
            for ti,di in order:
                if iou[ti,di]<thr[ti]: break
                if ti in mt or di in md: continue
                self.tracks[ti].update(det_xyxy[di],det_conf[di])
                mt.add(ti); md.add(di)
            for ti,t in enumerate(self.tracks):
                if ti not in mt: t.mark_lost()
            for di in range(len(det_xyxy)):
                if di not in md and det_conf[di]>=CONF_NEW:
                    self.tracks.append(Track(self._next_id,det_xyxy[di],det_cls[di],det_conf[di]))
                    self._next_id+=1
        self.tracks=[t for t in self.tracks if t.lost<=TRK_MAX_LOST and t.conf_smooth>=CONF_KEEP]
    def get_display(self):
        B,C,K,I=[],[],[],[]
        for t in self.tracks:
            if not t.confirmed(): continue
            if t.lost>TRK_SHOW_LOST: t.displayed=False; continue
            show=False
            if t.displayed:
                if t.conf_smooth>=CONF_DISPLAY_OFF or t.age<t.sticky_until: show=True
                else: t.displayed=False
            else:
                if t.conf_smooth>=CONF_DISPLAY_ON and t.lost==0:
                    t.displayed=True; t.sticky_until=t.age+DISPLAY_STICKY; show=True
            if show:
                B.append(t.box); C.append(t.conf_smooth); K.append(t.cls); I.append(t.id)
        if B:
            return (np.stack(B,0),np.array(C,np.float32),np.array(K,np.int32),np.array(I,np.int32))
        return (np.empty((0,4),np.float32),np.empty((0,),np.float32),
                np.empty((0,),np.int32),np.empty((0,),np.int32))


class Grabber(threading.Thread):
    def __init__(self,cam):
        super().__init__(daemon=True); self.cam=cam
        self._lock=threading.Lock()
        self._frame=None; self._ts=0.0; self._seq=0
        self._running=True; self._evt=threading.Event()
    def run(self):
        while self._running:
            f=self.cam.get_latest_frame()
            if f is None: continue
            with self._lock:
                self._frame=f; self._ts=time.perf_counter(); self._seq+=1
            self._evt.set()
    def wait_new(self,last_seq,timeout=0.05):
        if self._seq==last_seq:
            self._evt.clear(); self._evt.wait(timeout)
        with self._lock: return self._frame,self._ts,self._seq
    def stop(self):
        self._running=False; self._evt.set()


# ==================== Overlay (修复绘制错误) ====================
class Overlay(threading.Thread):
    CLASS_NAME="TRTGameOverlayWin_v21"; BG_COLORKEY=0x000000; TIMER_ID=1
    def __init__(self,region_xyxy,names_map,fps=120):
        super().__init__(daemon=True)
        self.x,self.y,x2,y2=region_xyxy
        self.w=x2-self.x; self.h=y2-self.y
        self.names=names_map
        self.timer_ms=max(1,int(1000/fps))
        self._lock=threading.Lock()
        self._boxes=None; self._confs=None; self._clss=None; self._ids=None
        self._stats=""; self._aim_active=False
        self._aim_point=None; self._locked_box=None
        self._dirty=False; self._running=True; self._hwnd=None
        self._mem_dc=None; self._mem_bmp=None; self._old_bmp=None
        self._font=None; self._bg_brush=None
        self._cross_brush=None; self._cross_brush_active=None
        self._pen_prio=None; self._pen_other=None
        self._pen_fov=None; self._pen_lock=None; self._pen_aim=None
        self._null_brush=None

    def push(self,boxes,confs,clss,ids,stats,aim_active=False,aim_point=None,locked_box=None):
        with self._lock:
            self._boxes=None if len(boxes)==0 else boxes.astype(np.int32).copy()
            self._confs=None if len(confs)==0 else confs.copy()
            self._clss=None if len(clss)==0 else clss.copy()
            self._ids=None if len(ids)==0 else ids.copy()
            self._stats=stats; self._aim_active=aim_active
            self._aim_point=aim_point; self._locked_box=locked_box
            self._dirty=True
        if self._hwnd:
            try: win32gui.InvalidateRect(self._hwnd,None,False)
            except Exception: pass

    def _wnd_proc(self,hwnd,msg,wp,lp):
        if msg==win32con.WM_PAINT: self._on_paint(hwnd); return 0
        elif msg==win32con.WM_ERASEBKGND: return 1
        elif msg==win32con.WM_TIMER:
            with self._lock: need=self._dirty
            if need:
                try: win32gui.InvalidateRect(hwnd,None,False)
                except Exception: pass
            return 0
        elif msg==win32con.WM_DESTROY: win32gui.PostQuitMessage(0); return 0
        return win32gui.DefWindowProc(hwnd,msg,wp,lp)

    def _init_gdi(self,ref_hdc):
        self._mem_dc=win32gui.CreateCompatibleDC(ref_hdc)
        self._mem_bmp=win32gui.CreateCompatibleBitmap(ref_hdc,self.w,self.h)
        self._old_bmp=win32gui.SelectObject(self._mem_dc,self._mem_bmp)
        try:
            self._font=create_font_ctypes(14,700,"Consolas")
            if not self._font: self._font=create_font_ctypes(14,700,"Arial")
        except Exception: self._font=None
        if not self._font:
            self._font = win32gui.GetStockObject(win32con.DEFAULT_GUI_FONT)
        self._bg_brush=win32gui.CreateSolidBrush(self.BG_COLORKEY)
        self._cross_brush=win32gui.CreateSolidBrush(win32api.RGB(0,255,255))
        self._cross_brush_active=win32gui.CreateSolidBrush(win32api.RGB(255,0,0))
        self._pen_prio=win32gui.CreatePen(win32con.PS_SOLID,2,win32api.RGB(0,255,0))
        self._pen_other=win32gui.CreatePen(win32con.PS_SOLID,2,win32api.RGB(255,140,0))
        self._pen_fov=win32gui.CreatePen(win32con.PS_DOT,1,win32api.RGB(255,255,0))
        self._pen_lock=win32gui.CreatePen(win32con.PS_SOLID,3,win32api.RGB(255,50,50))
        self._pen_aim=win32gui.CreatePen(win32con.PS_SOLID,2,win32api.RGB(255,0,0))
        self._null_brush=win32gui.GetStockObject(win32con.NULL_BRUSH)

    def _sd(self,o):
        if not o: return
        try: win32gui.DeleteObject(o)
        except Exception: pass

    def _cleanup_gdi(self):
        try:
            for o in [self._font,self._bg_brush,self._cross_brush,self._cross_brush_active,
                      self._pen_prio,self._pen_other,self._pen_fov,self._pen_lock,self._pen_aim]:
                self._sd(o)
            if self._mem_dc:
                if self._old_bmp:
                    try: win32gui.SelectObject(self._mem_dc,self._old_bmp)
                    except Exception: pass
                self._sd(self._mem_bmp)
                try: win32gui.DeleteDC(self._mem_dc)
                except Exception: pass
        except Exception: pass

    def _box_iou(self, a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _on_paint(self,hwnd):
        hdc,ps=win32gui.BeginPaint(hwnd)
        try:
            if self._mem_dc is None: self._init_gdi(hdc)
            mdc=self._mem_dc
            win32gui.FillRect(mdc,(0,0,self.w,self.h),self._bg_brush)
            with self._lock:
                boxes=self._boxes
                confs=self._confs
                clss=self._clss
                ids=self._ids
                stats=self._stats
                aim_active=self._aim_active
                aim_point=self._aim_point
                locked_box=self._locked_box
                self._dirty=False
            win32gui.SetBkMode(mdc,win32con.TRANSPARENT)
            old_font=win32gui.SelectObject(mdc,self._font) if self._font else None
            tf=(win32con.DT_LEFT|win32con.DT_TOP|win32con.DT_SINGLELINE|win32con.DT_NOCLIP)

            if DRAW_BOXES and boxes is not None and len(boxes) > 0:
                for i, (box, conf, cls, tid) in enumerate(zip(boxes, confs, clss, ids)):
                    x1, y1, x2, y2 = box
                    x1 = max(0, min(self.w, x1))
                    y1 = max(0, min(self.h, y1))
                    x2 = max(0, min(self.w, x2))
                    y2 = max(0, min(self.h, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    is_locked = False
                    if locked_box is not None:
                        lx1, ly1, lx2, ly2 = locked_box
                        lx1 -= self.x; ly1 -= self.y
                        lx2 -= self.x; ly2 -= self.y
                        iou = self._box_iou((x1,y1,x2,y2), (lx1,ly1,lx2,ly2))
                        if iou > 0.5:
                            is_locked = True

                    if is_locked:
                        pen = self._pen_lock
                    elif cls in AIM_CLASSES:
                        pen = self._pen_prio
                    else:
                        pen = self._pen_other
                    old_pen = win32gui.SelectObject(mdc, pen)
                    old_brush = win32gui.SelectObject(mdc, self._null_brush)
                    win32gui.Rectangle(mdc, x1, y1, x2, y2)
                    name = self.names.get(str(cls), f"cls{cls}")
                    label = f"{name} {conf:.2f} ID:{tid}"
                    tw, th = win32gui.GetTextExtentPoint32(mdc, label)
                    win32gui.SelectObject(mdc, self._bg_brush)
                    win32gui.Rectangle(mdc, x1, y1 - th - 4, x1 + tw + 4, y1)
                    if is_locked:
                        win32gui.SetTextColor(mdc, win32api.RGB(255, 50, 50))
                    elif cls in AIM_CLASSES:
                        win32gui.SetTextColor(mdc, win32api.RGB(0, 255, 0))
                    else:
                        win32gui.SetTextColor(mdc, win32api.RGB(255, 140, 0))
                    win32gui.DrawText(mdc, label, -1, (x1+2, y1-th-2, x1+tw+2, y1-2), tf)
                    win32gui.SelectObject(mdc, old_pen)
                    win32gui.SelectObject(mdc, old_brush)

            if DRAW_CROSSHAIR:
                cx,cy=self.w//2,self.h//2
                cb=self._cross_brush_active if aim_active else self._cross_brush
                win32gui.FillRect(mdc,(cx-10,cy-1,cx+10,cy+1),cb)
                win32gui.FillRect(mdc,(cx-1,cy-10,cx+1,cy+10),cb)

            if DRAW_STATS and stats:
                win32gui.SetTextColor(mdc,win32api.RGB(0,255,255))
                win32gui.DrawText(mdc,stats,-1,(5,5,self.w-5,25),tf)

            if aim_point is not None:
                ax, ay = aim_point
                ax -= self.x; ay -= self.y
                if 0 <= ax < self.w and 0 <= ay < self.h:
                    old_pen = win32gui.SelectObject(mdc, self._pen_aim)
                    win32gui.MoveToEx(mdc, int(ax)-5, int(ay))
                    win32gui.LineTo(mdc, int(ax)+5, int(ay))
                    win32gui.MoveToEx(mdc, int(ax), int(ay)-5)
                    win32gui.LineTo(mdc, int(ax), int(ay)+5)
                    win32gui.SelectObject(mdc, old_pen)

            if old_font: win32gui.SelectObject(mdc,old_font)
            win32gui.BitBlt(hdc,0,0,self.w,self.h,mdc,0,0,win32con.SRCCOPY)
        finally:
            win32gui.EndPaint(hwnd,ps)

    def run(self):
        hInst=win32api.GetModuleHandle(None)
        try:
            wc=win32gui.WNDCLASS()
            wc.lpszClassName=self.CLASS_NAME; wc.lpfnWndProc=self._wnd_proc
            wc.hInstance=hInst; wc.hbrBackground=0
            wc.hCursor=win32gui.LoadCursor(0,win32con.IDC_ARROW)
            wc.style=win32con.CS_HREDRAW|win32con.CS_VREDRAW
            win32gui.RegisterClass(wc)
        except Exception: pass
        style=win32con.WS_POPUP
        ex=(win32con.WS_EX_LAYERED|win32con.WS_EX_TRANSPARENT|
            win32con.WS_EX_TOPMOST|win32con.WS_EX_TOOLWINDOW|win32con.WS_EX_NOACTIVATE)
        self._hwnd=win32gui.CreateWindowEx(ex,self.CLASS_NAME,"Overlay",style,
            self.x,self.y,self.w,self.h,0,0,hInst,None)
        win32gui.SetLayeredWindowAttributes(self._hwnd,self.BG_COLORKEY,0,win32con.LWA_COLORKEY)
        win32gui.ShowWindow(self._hwnd,win32con.SW_SHOWNOACTIVATE)
        win32gui.SetWindowPos(self._hwnd,win32con.HWND_TOPMOST,self.x,self.y,self.w,self.h,
            win32con.SWP_NOACTIVATE|win32con.SWP_SHOWWINDOW)
        _user32.SetTimer(wintypes.HWND(self._hwnd),self.TIMER_ID,self.timer_ms,None)
        print(f"[overlay] at ({self.x},{self.y}) {self.w}x{self.h}")
        msg=wintypes.MSG(); pMsg=ctypes.byref(msg); PM_REMOVE=0x0001
        while self._running:
            while _user32.PeekMessageW(pMsg,0,0,0,PM_REMOVE):
                if msg.message==win32con.WM_QUIT: self._running=False; break
                _user32.TranslateMessage(pMsg); _user32.DispatchMessageW(pMsg)
            if not self._running: break
            _user32.WaitMessage()
        try: _user32.KillTimer(wintypes.HWND(self._hwnd),self.TIMER_ID)
        except Exception: pass
        self._cleanup_gdi()
        try: win32gui.DestroyWindow(self._hwnd)
        except Exception: pass

    def stop(self):
        self._running=False
        if self._hwnd:
            try: win32gui.PostMessage(self._hwnd,win32con.WM_CLOSE,0,0)
            except Exception: pass


# ==================== 主程序 ====================
def main():
    tune_process()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark=False

    with mss.mss() as sct:
        mon=sct.monitors[1]; sw,sh=mon["width"],mon["height"]
    cx,cy=sw//2,sh//2; half=CENTER_REGION//2
    region=(cx-half,cy-half,cx+half,cy+half)
    print(f"[region] {sw}x{sh} region={region}")

    cam=dxcam.create(output_idx=0,output_color="BGR")
    cam.start(region=region,target_fps=CAPTURE_FPS,video_mode=False)

    engine=TRTEngine(ENGINE_PATH,DEVICE)
    input_dtype=engine.io[engine.input_name]["dtype"]
    preproc=Preprocessor(IMG_SIZE,DEVICE,dtype=input_dtype)
    out_name=engine.output_names[0]

    mouse=MouseDriver(AIM_MOUSE_DRIVER)
    aimbot=Aimbot(mouse,region)
    print(f"[aim] 开镜键={AIM_HOTKEYS} 开火键={AIM_FIRE_KEY} fov={AIM_FOV_RADIUS}")
    print(f"[aim] 开火触发吸附={AIM_FIRE_TRIGGERS_AIM} fov倍率={AIM_FIRE_ONLY_FOV_MULT} 强度={AIM_FIRE_ONLY_STRENGTH}")
    print(f"[aim] P: kp=({AIM_KP_X},{AIM_KP_Y}) 快锁 {AIM_ACQUIRE_BOOST_MS}ms kp*{AIM_ACQUIRE_KP_MULT}")
    print(f"[aim] 移动增强: 速度alpha={AIM_TARGET_VEL_ALPHA} 预测={AIM_TARGET_PRED_MS}ms")
    print(f"[aim] 压枪=True 仅锁定时生效, max={AIM_RECOIL_Y_MAX}")
    print(f"[aim] 锁定 tol={AIM_LOCK_BASE_TOL} sticky={AIM_LOCK_STICKY_MS}ms")
    print(f"[aim] 头部优先类别: {AIM_HEAD_CLASS} (身体偏移: {AIM_OFFSETS.get(0, AIM_DEFAULT_OFFSET):.2f})")
    print(f"[aim] F2 探测, 9 校准")

    tracker=IoUTracker()
    print("[warmup]...")
    dummy=np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8)
    for _ in range(30): engine.infer(preproc(dummy))
    torch.cuda.synchronize(); print("[warmup] done")

    grabber=Grabber(cam); grabber.start()
    overlay=None
    if SHOW_OVERLAY:
        overlay=Overlay(region,engine.names_map,fps=OVERLAY_FPS)
        overlay.start(); time.sleep(0.2)

    min_interval=1.0/MAX_INFER_FPS
    next_allowed=time.perf_counter()
    infer_cnt=0; infer_ms_hist=deque(maxlen=240)
    t_start=time.perf_counter(); t_log=t_start; t_stats=t_start
    last_seq=-1; last_disp_n=0; cached_stats=""

    print(f"[run] IMG={IMG_SIZE} NC={NUM_CLASSES}")
    try:
        while True:
            now=time.perf_counter()
            if now<next_allowed:
                s=next_allowed-now
                if s>0.0005: time.sleep(s)
            frame,_,seq=grabber.wait_new(last_seq,timeout=0.05)
            if frame is None or seq==last_seq: continue
            last_seq=seq
            t0=time.perf_counter()
            gi=preproc(frame); outs=engine.infer(gi)
            raw_xyxy,raw_conf,raw_cls=postprocess(outs[out_name],CONF_KEEP,IOU_THRES,NUM_CLASSES,TARGET_CLASS)
            tracker.update(raw_xyxy,raw_conf,raw_cls)
            disp_boxes,disp_confs,disp_clss,disp_ids=tracker.get_display()
            last_disp_n=len(disp_boxes)
            aim_active=aimbot.tick(raw_xyxy,raw_conf,raw_cls,np.arange(len(raw_xyxy)))
            t1=time.perf_counter()
            infer_ms_hist.append((t1-t0)*1000.0); infer_cnt+=1
            next_allowed=t0+min_interval

            if (t1-t_stats)*1000.0>=STATS_UPDATE_MS:
                dt=max(t1-t_start,1e-6)
                arr=np.fromiter(infer_ms_hist,dtype=np.float32)
                cached_stats=(f"FPS={infer_cnt/dt:.0f} "
                              f"p50={float(np.percentile(arr,50)):.1f}ms "
                              f"trk={last_disp_n}/{len(tracker.tracks)} "
                              f"{'AIM' if aim_active else '---'}")
                t_stats=t1

            if overlay is not None and overlay._running:
                overlay.push(disp_boxes,disp_confs,disp_clss,disp_ids,
                             cached_stats,aim_active,
                             aimbot.aim_point,aimbot.locked_box_screen)

            if t1-t_log>=LOG_INTERVAL:
                dt=max(t1-t_start,1e-6)
                arr=np.fromiter(infer_ms_hist,dtype=np.float32)
                print(f"[stat] FPS={infer_cnt/dt:6.1f} "
                      f"p50={float(np.percentile(arr,50)):.2f} "
                      f"p99={float(np.percentile(arr,99)):.2f} "
                      f"trk={last_disp_n}/{len(tracker.tracks)} "
                      f"aim={'ON' if aim_active else 'off'}")
                t_log=t1
    except KeyboardInterrupt: pass
    finally:
        print("[run] shutting down...")
        grabber.stop()
        if overlay is not None: overlay.stop()
        try: mouse.close()
        except Exception: pass
        try: cam.stop()
        except Exception: pass
        print("[run] bye")

if __name__=="__main__":
    main()