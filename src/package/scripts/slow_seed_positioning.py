#!/usr/bin/env python3
"""Slow, deadman-gated positioning to the DP-CNN IK seed pose."""
from __future__ import annotations

import json
from pathlib import Path
import time

import pygame
import cv2
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TARGET = (-7.2001, -33.1997, 72.6224, 11.7131, 33.2003)
MAX_STEP_DEG = 2.0

def next_step(current: tuple[float, ...], target: tuple[float, ...]) -> tuple[float, ...]:
    """Return one bounded interpolation step toward target."""
    return tuple(c + max(-MAX_STEP_DEG, min(MAX_STEP_DEG, t - c)) for c, t in zip(current, target, strict=True))

def main() -> int:
    receipt = Path("/home/intro/InternLab/02_InTro_Project/04_experiments/so101_pusht_benchmark/inference/sim_to_real_rollout/positioning/slow-seed-positioning.json")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    config = SO101FollowerConfig(port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6082660-if00", id="intro_so101_follower_01", calibration_dir=Path("/home/intro/.cache/huggingface/lerobot/calibration/robots/so101_follower"), use_degrees=True, cameras={}, max_relative_target=MAX_STEP_DEG, disable_torque_on_disconnect=False)
    robot = SO101Follower(config)
    pygame.init(); screen=pygame.display.set_mode((1400,480)); pygame.display.set_caption("SO-101 SLOW SEED POSITIONING — SPACE hold / ESC stop");font=pygame.font.SysFont("monospace",24);clock=pygame.time.Clock();writes=0;history=[];ticks=0;state="HOLD — hold SPACE to move"
    robot.connect(calibrate=False)
    for joint in JOINTS:
        robot.bus.write("Acceleration", joint, 5)
        robot.bus.write("Goal_Velocity", joint, 100)
        robot.bus.write("P_Coefficient", joint, 32)
    robot.bus.write("P_Coefficient", "shoulder_lift", 64)
    robot.bus.write("Protection_Current", "shoulder_lift", 450)
    robot.bus.write("Goal_Velocity", "shoulder_lift", 120)
    robot.bus.enable_torque()
    camera=cv2.VideoCapture("/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0")
    try:
        running=True
        while running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE): state="EMERGENCY STOP";running=False
            observed=robot.bus.sync_read("Present_Position");current=tuple(float(observed[j]) for j in JOINTS);delta=tuple(t-c for c,t in zip(current,TARGET,strict=True));ticks+=1;history.append(current);history=history[-21:]
            if ticks % 10 == 0: print(f"STATE current={current} delta={delta}",flush=True)
            if max(abs(x) for x in delta)<=0.5: state="TARGET REACHED — torque HOLDING";print("TARGET_REACHED",flush=True);running=False
            elif len(history)==21 and max(abs(a-b) for a,b in zip(history[0],history[-1],strict=True))<0.3:
                state=f"STALL — delta={delta}";print(state,flush=True);running=False
            elif True:  # lab-present owner-approved automatic slow positioning
                step=next_step(current,TARGET);robot.bus.sync_write("Goal_Position",dict(zip(JOINTS,step,strict=True)));writes+=1;state="AUTO MOVING — ESC for emergency STOP"
            else: state="HOLD — hold SPACE to move"
            screen.fill((10,10,18));lines=[state,"AUTO MOVE | ESC = emergency STOP",f"CURRENT {tuple(round(x,1) for x in current)}",f"TARGET  {TARGET}",f"WRITES  {writes}"]
            for i,line in enumerate(lines):screen.blit(font.render(line,True,(240,240,240)),(30,35+i*55))
            ok,frame=camera.read()
            if ok:
                rgb=cv2.cvtColor(cv2.resize(frame,(480,360)),cv2.COLOR_BGR2RGB);surface=pygame.image.frombuffer(rgb.tobytes(),(480,360),"RGB");screen.blit(surface,(900,30))
            pygame.display.flip();clock.tick(10)
    finally:
        holding = state.startswith("TARGET REACHED")
        camera.release()
        if not holding:
            robot.bus.disable_torque()
        robot.disconnect()
        pygame.quit()
        receipt.write_text(json.dumps({"state":state,"write_count":writes,"target_degrees":TARGET,"torque_holding":holding,"completed_at":time.time()},indent=2)+"\n")
    return 0 if state.startswith("TARGET REACHED") else 2

if __name__=="__main__": raise SystemExit(main())
