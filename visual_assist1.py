#!/usr/bin/env python3
"""
Visual Assist System for Indoor Navigation of Visually Impaired People
A complete assistive technology system using YOLOv8, DeepSORT, and MiDaS
Enhanced with context-rich announcements for better guidance
"""

import cv2
import numpy as np
import pyttsx3
import threading
import queue
import time
import torch
from collections import deque, defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import speech_recognition as sr

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Camera settings
CAMERA_SOURCE = "http://192.168.1.4:8080/video"  # 0 for webcam, or "rtsp://..." for IP camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# YOLO settings
YOLO_MODEL = "yolov8n.pt"  # nano model for speed
YOLO_CONFIDENCE = 0.5

# Target objects for detection
TARGET_CLASSES = {
    'chair', 'dining table', 'bottle', 'person', 'door', 
    'couch', 'bed', 'laptop', 'cup', 'keyboard', 'mouse'
}

# Simplified class name mapping
CLASS_NAME_MAP = {
    'dining table': 'table',
    'couch': 'sofa'
}

# Known object heights (in meters) for distance estimation
OBJECT_HEIGHTS = {
    'person': 1.7,
    'chair': 0.9,
    'table': 0.75,
    'door': 2.0,
    'sofa': 0.8,
    'bed': 0.6,
    'bottle': 0.25,
    'laptop': 0.02,
    'cup': 0.12,
    'keyboard': 0.03,
    'mouse': 0.04
}

# Distance thresholds (meters)
VERY_NEAR_THRESHOLD = 0.7
NEAR_THRESHOLD = 1.5

# Announcement cooldowns (seconds)
OBJECT_ANNOUNCE_COOLDOWN = 3.0
CLEAR_PATH_COOLDOWN = 5.0
WALL_ANNOUNCE_COOLDOWN = 4.0
APPROACH_ANNOUNCE_COOLDOWN = 4.0

# MiDaS settings
MIDAS_INTERVAL = 3.0  # Run MiDaS every N seconds
WALL_DEPTH_THRESHOLD = 0.3  # Depth value threshold for wall detection
WALL_COVERAGE_THRESHOLD = 0.4  # Fraction of frame that must be close

# Tracking settings
APPROACH_DISTANCE_THRESHOLD = 0.3  # meters closer to trigger approach warning
TRACK_HISTORY_SIZE = 30  # frames

# Screen regions for direction
LEFT_REGION = 0.33
RIGHT_REGION = 0.67

# ============================================================================
# AUDIO SYSTEM
# ============================================================================

class AudioSystem:
    """Thread-safe audio system with priority queue and stop/start control"""
    
    def __init__(self):
        self.normal_queue = queue.Queue()
        self.urgent_queue = queue.Queue()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 0.9)
        self.is_speaking = False
        self.is_enabled = True  # Control flag for stop/start
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        
        # Start TTS thread
        self.thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.thread.start()
    
    def _tts_worker(self):
        """Background worker for text-to-speech with pause control"""
        while not self.stop_flag.is_set():
            try:
                # Wait if paused
                while self.pause_flag.is_set() and not self.stop_flag.is_set():
                    time.sleep(0.1)
                
                if not self.is_enabled:
                    time.sleep(0.1)
                    continue
                
                # Check urgent queue first
                try:
                    text = self.urgent_queue.get(timeout=0.1)
                    self._speak(text)
                    continue
                except queue.Empty:
                    pass
                
                # Then check normal queue
                try:
                    text = self.normal_queue.get(timeout=0.1)
                    self._speak(text)
                except queue.Empty:
                    pass
                    
            except Exception as e:
                print(f"TTS Error: {e}")
    
    def _speak(self, text):
        """Actual speech synthesis"""
        if not self.is_enabled:
            return
        
        try:
            self.is_speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.is_speaking = False
        except Exception as e:
            print(f"Speech error: {e}")
            self.is_speaking = False
    
    def announce(self, text, urgent=False):
        """Add text to speech queue"""
        if not self.is_enabled:
            return
        
        if urgent:
            # Clear normal queue for urgent messages
            while not self.normal_queue.empty():
                try:
                    self.normal_queue.get_nowait()
                except queue.Empty:
                    break
            self.urgent_queue.put(text)
        else:
            # Avoid queue buildup
            if self.normal_queue.qsize() < 3:
                self.normal_queue.put(text)
    
    def pause_announcements(self):
        """Stop all announcements"""
        self.is_enabled = False
        self.pause_flag.set()
        # Clear queues
        while not self.normal_queue.empty():
            try:
                self.normal_queue.get_nowait()
            except queue.Empty:
                break
        while not self.urgent_queue.empty():
            try:
                self.urgent_queue.get_nowait()
            except queue.Empty:
                break
        print("🔇 Audio announcements STOPPED")
    
    def resume_announcements(self):
        """Resume announcements"""
        self.is_enabled = True
        self.pause_flag.clear()
        print("🔊 Audio announcements RESUMED")
    
    def stop(self):
        """Stop the audio system"""
        self.stop_flag.set()
        self.thread.join(timeout=2)

# ============================================================================
# VOICE RECOGNITION SYSTEM
# ============================================================================

class VoiceCommandSystem:
    """Enhanced voice command recognition with better feedback"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening_enabled = True  # Control flag
        
        try:
            self.microphone = sr.Microphone()
            self.recognizer.energy_threshold = 3000
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            print("✓ Microphone initialized successfully")
        except Exception as e:
            print(f"✗ Microphone initialization failed: {e}")
            print("  Voice commands will not be available")
    
    def listen_for_command(self, timeout=3, phrase_limit=5):
        """Listen for a voice command with better error handling"""
        if not self.microphone:
            print("✗ No microphone available")
            return None
        
        try:
            print("\n🎤 Listening... (speak now)")
            with self.microphone as source:
                # Quick ambient noise adjustment
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            
            print("⏳ Processing speech...")
            command = self.recognizer.recognize_google(audio).lower()
            print(f"✓ Recognized: \"{command}\"")
            return command
            
        except sr.WaitTimeoutError:
            print("✗ No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            print("✗ Could not understand audio (unclear speech)")
            return None
        except sr.RequestError as e:
            print(f"✗ Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None

# ============================================================================
# DISTANCE ESTIMATION
# ============================================================================

def estimate_distance(bbox_height, frame_height, object_class):
    """Estimate distance using bounding box height and known object heights"""
    if object_class not in OBJECT_HEIGHTS:
        return None
    
    # Simple pinhole camera model
    # distance = (known_height * focal_length) / bbox_height
    # Focal length approximation for typical webcam
    focal_length = frame_height * 1.2
    
    object_height = OBJECT_HEIGHTS[object_class]
    if bbox_height < 10:  # Avoid division by very small numbers
        return None
    
    distance = (object_height * focal_length) / bbox_height
    return max(0.3, min(distance, 10.0))  # Clamp between 0.3m and 10m

def get_distance_category(distance):
    """Categorize distance into very near, near, or far"""
    if distance is None:
        return "unknown distance"
    
    if distance < VERY_NEAR_THRESHOLD:
        return "very near"
    elif distance < NEAR_THRESHOLD:
        return "near"
    else:
        return "far"

def get_direction(bbox_center_x, frame_width):
    """Determine direction based on bounding box position"""
    normalized_x = bbox_center_x / frame_width
    
    if normalized_x < LEFT_REGION:
        return "on your left"
    elif normalized_x > RIGHT_REGION:
        return "on your right"
    else:
        return "in front"

def get_avoidance_direction(bbox_center_x, frame_width):
    """Get suggested avoidance direction based on object position"""
    normalized_x = bbox_center_x / frame_width
    
    if normalized_x < LEFT_REGION:
        return "Move slightly to your right"
    elif normalized_x > RIGHT_REGION:
        return "Move slightly to your left"
    else:
        # Object in center - suggest moving to the side with more space
        if normalized_x < 0.5:
            return "Move to your right"
        else:
            return "Move to your left"

# ============================================================================
# MiDaS DEPTH ESTIMATION (WALL DETECTION ONLY)
# ============================================================================

class MiDaSWallDetector:
    """MiDaS-based wall detection - FALLBACK ONLY"""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_run_time = 0
        
        try:
            # Load MiDaS small model
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(self.device)
            self.model.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            
            print("MiDaS loaded successfully")
        except Exception as e:
            print(f"MiDaS loading failed: {e}")
    
    def detect_wall(self, frame):
        """Detect if there's a wall ahead - THROTTLED"""
        current_time = time.time()
        
        # Throttle MiDaS execution
        if current_time - self.last_run_time < MIDAS_INTERVAL:
            return False, None
        
        if self.model is None:
            return False, None
        
        try:
            self.last_run_time = current_time
            
            # Prepare input
            input_batch = self.transform(frame).to(self.device)
            
            # Predict depth
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth = prediction.cpu().numpy()
            
            # Normalize depth
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
            # Check central region for close obstacles (potential wall)
            h, w = depth_normalized.shape
            center_region = depth_normalized[h//4:3*h//4, w//4:3*w//4]
            
            # Count pixels that are very close
            close_pixels = np.sum(center_region < WALL_DEPTH_THRESHOLD)
            total_pixels = center_region.size
            coverage = close_pixels / total_pixels
            
            # Wall detected if large portion of center is close
            wall_detected = coverage > WALL_COVERAGE_THRESHOLD
            
            return wall_detected, depth_normalized
            
        except Exception as e:
            print(f"MiDaS error: {e}")
            return False, None

# ============================================================================
# OBJECT TRACKER WITH APPROACH DETECTION
# ============================================================================

class ObjectTracker:
    """Track objects and detect approaching obstacles"""
    
    def __init__(self):
        self.tracker = DeepSort(max_age=30)
        self.track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_SIZE))
        self.last_approach_announce = defaultdict(float)
    
    def update(self, detections, frame):
        """Update tracker with new detections"""
        if len(detections) == 0:
            self.tracker.update_tracks([], frame=frame)
            return []
        
        # Format detections for DeepSORT
        bbs = []
        for det in detections:
            x1, y1, x2, y2, conf, class_name, distance = det
            bbs.append(([x1, y1, x2-x1, y2-y1], conf, class_name))
        
        # Update tracks
        tracks = self.tracker.update_tracks(bbs, frame=frame)
        
        # Process tracks
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()
            class_name = track.get_det_class()
            
            # Store history
            x1, y1, x2, y2 = bbox
            bbox_height = y2 - y1
            
            # Find corresponding detection to get distance
            distance = None
            for det in detections:
                if det[5] == class_name:  # Match by class
                    distance = det[6]
                    break
            
            self.track_history[track_id].append({
                'distance': distance,
                'bbox_height': bbox_height,
                'time': time.time()
            })
            
            tracked_objects.append({
                'track_id': track_id,
                'bbox': bbox,
                'class_name': class_name,
                'distance': distance
            })
        
        return tracked_objects
    
    def check_approaching(self, tracked_objects):
        """Check if any tracked object is approaching"""
        approaching = []
        current_time = time.time()
        
        for obj in tracked_objects:
            track_id = obj['track_id']
            history = self.track_history[track_id]
            
            if len(history) < 10:  # Need enough history
                continue
            
            # Compare current distance with distance from 2 seconds ago
            recent_distances = [h['distance'] for h in list(history)[-5:] if h['distance'] is not None]
            old_distances = [h['distance'] for h in list(history)[:5] if h['distance'] is not None]
            
            if len(recent_distances) < 3 or len(old_distances) < 3:
                continue
            
            avg_recent = np.mean(recent_distances)
            avg_old = np.mean(old_distances)
            
            # Check if getting closer
            if avg_old - avg_recent > APPROACH_DISTANCE_THRESHOLD:
                # Check cooldown
                if current_time - self.last_approach_announce[track_id] > APPROACH_ANNOUNCE_COOLDOWN:
                    approaching.append(obj)
                    self.last_approach_announce[track_id] = current_time
        
        return approaching

# ============================================================================
# MAIN VISUAL ASSIST SYSTEM
# ============================================================================

class VisualAssistSystem:
    """Main system integrating all components"""
    
    def __init__(self):
        print("Initializing Visual Assist System...")
        
        # Initialize components
        self.audio = AudioSystem()
        self.voice = VoiceCommandSystem()
        self.tracker = ObjectTracker()
        self.midas = MiDaSWallDetector()
        
        # Navigation mode flag
        self.navigation_mode = False
        
        # Load YOLO
        print("Loading YOLOv8...")
        self.yolo = YOLO(YOLO_MODEL)
        
        # Cooldown trackers
        self.last_object_announce = defaultdict(float)
        self.last_clear_path_announce = 0
        self.last_wall_announce = 0
        
        # Camera
        self.cap = cv2.VideoCapture(CAMERA_SOURCE)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        print("System ready!")
        print("\n" + "="*60)
        print("VOICE COMMANDS AVAILABLE:")
        print("="*60)
        print("🎯 Path queries:")
        print("  • 'Is the path clear?'")
        print("  • 'What is in front of me?'")
        print("  • 'Any obstacles nearby?'")
        print("  • 'How far is it?'")
        print("\n🧭 Navigation:")
        print("  • 'Guide me' / 'Navigate'")
        print("  • 'Is left clear?' / 'Is right clear?'")
        print("  • 'Where is the door/chair?'")
        print("  • 'Describe surroundings'")
        print("\n🔊 System control:")
        print("  • 'Stop' - Stop announcements")
        print("  • 'Start' - Resume announcements")
        print("="*60 + "\n")
        
        self.audio.announce("Visual assist system ready")
    
    def detect_objects(self, frame):
        """Run YOLO detection and estimate distances"""
        results = self.yolo(frame, conf=YOLO_CONFIDENCE, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                
                # Filter target classes
                if class_name not in TARGET_CLASSES:
                    continue
                
                # Map class name
                display_name = CLASS_NAME_MAP.get(class_name, class_name)
                
                # Estimate distance
                bbox_height = y2 - y1
                distance = estimate_distance(bbox_height, frame.shape[0], display_name)
                
                detections.append((x1, y1, x2, y2, conf, display_name, distance))
        
        return detections
    
    def process_detections(self, detections, frame_width):
        """Process detections and generate CONTEXT-RICH announcements"""
        announcements = []
        current_time = time.time()
        
        # Sort by distance (closest first)
        valid_dets = [d for d in detections if d[6] is not None]
        valid_dets.sort(key=lambda x: x[6])
        
        # Announce closest objects (max 2 per frame)
        announced_count = 0
        for det in valid_dets[:3]:
            x1, y1, x2, y2, conf, class_name, distance = det
            
            # Check cooldown
            key = f"{class_name}_{int(distance*10)}"
            if current_time - self.last_object_announce[key] < OBJECT_ANNOUNCE_COOLDOWN:
                continue
            
            # Get direction and distance category
            bbox_center_x = (x1 + x2) / 2
            direction = get_direction(bbox_center_x, frame_width)
            dist_category = get_distance_category(distance)
            
            # ========================================
            # 🔊 BUILD CONTEXT-RICH MESSAGE
            # ========================================
            # Format: "Object [distance] [direction]. [Action/Guidance]."
            
            base_info = f"{class_name.capitalize()} {dist_category} {direction}."
            
            if dist_category == "very near":
                # Critical warning with immediate action
                avoidance = get_avoidance_direction(bbox_center_x, frame_width)
                message = f"{base_info} Stop. {avoidance}."
                urgent = True
            elif dist_category == "near":
                # Warning with caution
                message = f"{base_info} Walk carefully."
                urgent = False
            else:
                # Information only
                message = base_info
                urgent = False
            
            announcements.append((message, urgent))
            self.last_object_announce[key] = current_time
            
            announced_count += 1
            if announced_count >= 2:
                break
        
        return announcements
    
    def handle_voice_commands(self, detections):
        """Enhanced voice command handler with terminal output and navigation"""
        command = self.voice.listen_for_command()
        
        if command is None:
            return
        
        print("\n" + "="*60)
        print(f"📢 VOICE COMMAND: \"{command}\"")
        print("="*60)
        
        # ========================================
        # SYSTEM CONTROL COMMANDS
        # ========================================
        
        if "stop" in command or "shut up" in command or "quiet" in command:
            print("🔇 RESPONSE: Stopping announcements")
            print("="*60 + "\n")
            self.audio.pause_announcements()
            self.audio.announce("Stopping announcements", urgent=True)
            return
        
        elif "start" in command or "resume" in command or "continue" in command:
            print("🔊 RESPONSE: Resuming announcements")
            print("="*60 + "\n")
            self.audio.resume_announcements()
            self.audio.announce("Resuming announcements", urgent=True)
            return
        
        # ========================================
        # PATH & OBSTACLE QUERIES
        # ========================================
        
        elif "path clear" in command or "clear path" in command or "is it clear" in command:
            if len(detections) == 0:
                response = "✅ YES - Path ahead looks clear"
                speech = "Yes, path ahead looks clear. You may continue forward."
            else:
                nearby_count = sum(1 for d in detections if d[6] and d[6] < NEAR_THRESHOLD)
                response = f"❌ NO - {len(detections)} object(s) detected, {nearby_count} nearby"
                speech = f"No, there are {len(detections)} obstacles detected. Walk carefully."
            
            print(f"🗣️  ANSWER: {response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
        
        elif "what" in command and ("front" in command or "ahead" in command or "see" in command):
            if len(detections) == 0:
                response = "Nothing detected in front"
                speech = "Nothing detected in front. Path is clear."
            else:
                # List up to 3 closest objects
                sorted_dets = sorted([d for d in detections if d[6]], key=lambda x: x[6])[:3]
                objects_list = []
                for det in sorted_dets:
                    class_name = det[5]
                    distance = det[6]
                    dist_cat = get_distance_category(distance)
                    objects_list.append(f"{class_name} {dist_cat} ({distance:.1f}m)")
                
                response = "Objects ahead:\n  " + "\n  ".join([f"• {obj}" for obj in objects_list])
                
                # Enhanced speech with context
                closest = sorted_dets[0]
                speech = f"{len(sorted_dets)} objects detected. Closest is {closest[5]} {get_distance_category(closest[6])} at {closest[6]:.1f} meters."
            
            print(f"🗣️  ANSWER:\n{response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
        
        elif "obstacle" in command or "nearby" in command or "around me" in command:
            nearby = [d for d in detections if d[6] and d[6] < NEAR_THRESHOLD]
            very_near = [d for d in detections if d[6] and d[6] < VERY_NEAR_THRESHOLD]
            
            if len(nearby) == 0:
                response = "✅ No nearby obstacles detected"
                speech = "No nearby obstacles. Path is clear."
            else:
                response = f"⚠️  {len(nearby)} nearby obstacle(s):\n"
                for det in nearby[:3]:
                    response += f"  • {det[5]} at {det[6]:.1f}m ({get_distance_category(det[6])})\n"
                
                if very_near:
                    speech = f"Warning! {len(very_near)} very near obstacles and {len(nearby)} total nearby. Stop and proceed carefully."
                else:
                    speech = f"{len(nearby)} nearby obstacles detected. Walk carefully."
            
            print(f"🗣️  ANSWER:\n{response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
        
        elif "how far" in command or "distance" in command:
            if len(detections) == 0:
                response = "No objects detected to measure"
                speech = "No objects detected. Path is clear."
            else:
                closest = min([d for d in detections if d[6]], key=lambda x: x[6])
                response = f"Closest object: {closest[5]} at {closest[6]:.1f} meters"
                speech = f"Closest {closest[5]} is {get_distance_category(closest[6])} at {closest[6]:.1f} meters."
            
            print(f"🗣️  ANSWER: {response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
        
        # ========================================
        # NAVIGATION COMMANDS
        # ========================================
        
        elif "guide me" in command or "navigate" in command or "help me walk" in command:
            response = "🧭 Navigation mode activated"
            speech = "Navigation mode activated. I will guide you step by step. Walk slowly and listen carefully."
            print(f"🗣️  ANSWER: {response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
            self.navigation_mode = True
        
        elif "left" in command and ("safe" in command or "clear" in command):
            # Check left side
            left_objects = [d for d in detections if d[6] and (d[0] + d[2])/2 < FRAME_WIDTH * LEFT_REGION]
            if len(left_objects) == 0:
                response = "✅ Left side is clear"
                speech = "Left side is clear. You may move left."
            else:
                closest_left = min(left_objects, key=lambda x: x[6])
                response = f"⚠️  {len(left_objects)} object(s) on the left"
                speech = f"{len(left_objects)} obstacles on the left. {closest_left[5]} {get_distance_category(closest_left[6])}. Do not move left."
            
            print(f"🗣️  ANSWER: {response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
        
        elif "right" in command and ("safe" in command or "clear" in command):
            # Check right side
            right_objects = [d for d in detections if d[6] and (d[0] + d[2])/2 > FRAME_WIDTH * RIGHT_REGION]
            if len(right_objects) == 0:
                response = "✅ Right side is clear"
                speech = "Right side is clear. You may move right."
            else:
                closest_right = min(right_objects, key=lambda x: x[6])
                response = f"⚠️  {len(right_objects)} object(s) on the right"
                speech = f"{len(right_objects)} obstacles on the right. {closest_right[5]} {get_distance_category(closest_right[6])}. Do not move right."
            
            print(f"🗣️  ANSWER: {response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
        
        elif "where" in command and "door" in command:
            doors = [d for d in detections if d[5] == "door"]
            if len(doors) == 0:
                response = "No doors detected"
                speech = "No doors visible in current view."
            else:
                door = doors[0]
                direction = get_direction((door[0] + door[2])/2, FRAME_WIDTH)
                dist_cat = get_distance_category(door[6])
                response = f"🚪 Door detected {direction}, {dist_cat} ({door[6]:.1f}m)"
                speech = f"Door {dist_cat} {direction} at {door[6]:.1f} meters."
            
            print(f"🗣️  ANSWER: {response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
        
        elif "where" in command and "chair" in command:
            chairs = [d for d in detections if d[5] == "chair"]
            if len(chairs) == 0:
                response = "No chairs detected"
                speech = "No chairs visible in current view."
            else:
                chair = chairs[0]
                direction = get_direction((chair[0] + chair[2])/2, FRAME_WIDTH)
                dist_cat = get_distance_category(chair[6])
                response = f"🪑 Chair detected {direction}, {dist_cat} ({chair[6]:.1f}m)"
                speech = f"Chair {dist_cat} {direction} at {chair[6]:.1f} meters."
            
            print(f"🗣️  ANSWER: {response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
        
        elif "describe" in command or "what around" in command:
            if len(detections) == 0:
                response = "Environment description: No objects detected in view"
                speech = "No objects detected around you. Environment appears clear."
            else:
                # Group by class
                class_counts = {}
                for det in detections:
                    class_name = det[5]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                response = "📋 Environment description:\n"
                items = []
                for cls, count in class_counts.items():
                    response += f"  • {count} {cls}(s)\n"
                    items.append(f"{count} {cls}")
                
                speech = f"I see {', '.join(items)} in your surroundings."
            
            print(f"🗣️  ANSWER:\n{response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
        
        # ========================================
        # UNKNOWN COMMAND
        # ========================================
        else:
            response = "❓ Command not recognized. Try:\n"
            response += "  • 'Is the path clear?'\n"
            response += "  • 'What is in front of me?'\n"
            response += "  • 'Any obstacles nearby?'\n"
            response += "  • 'Stop' / 'Start'\n"
            response += "  • 'Guide me' / 'Navigate'\n"
            response += "  • 'Is left/right clear?'"
            
            speech = "Command not recognized. Say, is the path clear, or what is in front, or any obstacles nearby."
            
            print(f"🗣️  ANSWER:\n{response}")
            print("="*60 + "\n")
            self.audio.announce(speech, urgent=True)
    
    def run(self):
        """Main processing loop"""
        print("\n" + "="*60)
        print("SYSTEM STARTED - ENHANCED GUIDANCE MODE")
        print("="*60)
        print("Press 'q' to quit")
        print("Press 'v' to ask a voice question")
        print("Press 's' to stop announcements")
        print("Press 'r' to resume announcements")
        print("="*60 + "\n")
        
        voice_thread = None
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_height, frame_width = frame.shape[:2]
                frame_count += 1
                
                # ========================================
                # 1. YOLO DETECTION (HIGHEST PRIORITY)
                # ========================================
                detections = self.detect_objects(frame)
                
                # Print detection summary every 30 frames (about 1 second)
                if frame_count % 30 == 0 and detections:
                    print(f"\n[Frame {frame_count}] {len(detections)} objects detected:")
                    for det in detections[:5]:  # Show first 5
                        cls_name = det[5]
                        dist = det[6]
                        if dist:
                            print(f"  • {cls_name}: {dist:.2f}m ({get_distance_category(dist)})")
                
                # ========================================
                # 2. TRACKING & APPROACH DETECTION
                # ========================================
                tracked_objects = self.tracker.update(detections, frame)
                approaching = self.tracker.check_approaching(tracked_objects)
                
                # ========================================
                # 🔊 ANNOUNCE APPROACHING OBJECTS (CONTEXT-RICH)
                # ========================================
                for obj in approaching:
                    bbox_center = (obj['bbox'][0] + obj['bbox'][2]) / 2
                    direction = get_direction(bbox_center, frame_width)
                    dist_category = get_distance_category(obj['distance'])
                    
                    # Context-rich approaching message
                    message = (
                        f"{obj['class_name'].capitalize()} {dist_category} {direction}. "
                        f"Object approaching. Slow down."
                    )
                    print(f"⚠️  ALERT: {message}")
                    self.audio.announce(message, urgent=True)
                
                # ========================================
                # 3. PROCESS DETECTIONS & ANNOUNCE
                # ========================================
                if len(detections) > 0 and self.audio.is_enabled:
                    announcements = self.process_detections(detections, frame_width)
                    for message, urgent in announcements:
                        self.audio.announce(message, urgent=urgent)
                
                # ========================================
                # 4. MiDaS WALL DETECTION (FALLBACK ONLY)
                # ========================================
                wall_detected = False
                depth_map = None
                
                if len(detections) == 0:  # Only if NO objects detected
                    wall_detected, depth_map = self.midas.detect_wall(frame)
                    
                    if wall_detected and self.audio.is_enabled:
                        current_time = time.time()
                        if current_time - self.last_wall_announce > WALL_ANNOUNCE_COOLDOWN:
                            # Context-rich wall message
                            message = "Wall very near in front. Stop. Turn slightly to your side."
                            print(f"⚠️  WALL DETECTED: {message}")
                            self.audio.announce(message, urgent=True)
                            self.last_wall_announce = current_time
                
                # ========================================
                # 5. CLEAR PATH ANNOUNCEMENT
                # ========================================
                if len(detections) == 0 and not wall_detected and self.audio.is_enabled:
                    current_time = time.time()
                    if current_time - self.last_clear_path_announce > CLEAR_PATH_COOLDOWN:
                        # Context-rich clear path message
                        message = "Path ahead looks clear. You may continue forward."
                        self.audio.announce(message, urgent=False)
                        self.last_clear_path_announce = current_time
                
                # ========================================
                # 6. VISUALIZATION
                # ========================================
                display_frame = frame.copy()
                
                # Draw detections
                for det in detections:
                    x1, y1, x2, y2, conf, class_name, distance = det
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Color based on distance
                    dist_cat = get_distance_category(distance)
                    if dist_cat == "very near":
                        color = (0, 0, 255)  # Red
                    elif dist_cat == "near":
                        color = (0, 165, 255)  # Orange
                    else:
                        color = (0, 255, 0)  # Green
                    
                    # Draw box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label
                    label = f"{class_name}"
                    if distance:
                        label += f" {distance:.1f}m ({dist_cat})"
                    
                    cv2.putText(display_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw track IDs
                for obj in tracked_objects:
                    x1, y1, x2, y2 = [int(v) for v in obj['bbox']]
                    track_id = obj['track_id']
                    cv2.putText(display_frame, f"ID:{track_id}", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Show status
                status_text = f"Objects: {len(detections)}"
                if wall_detected:
                    status_text += " | Wall: YES"
                if not self.audio.is_enabled:
                    status_text += " | Audio: MUTED"
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Voice command indicator
                cv2.putText(display_frame, "Press 'v' for voice command", (10, frame_height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("Visual Assist System", display_frame)
                
                # ========================================
                # 7. KEYBOARD INPUT
                # ========================================
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    # Voice command mode
                    print("\n" + "🎤"*30)
                    self.audio.announce("Listening", urgent=True)
                    voice_thread = threading.Thread(
                        target=self.handle_voice_commands, 
                        args=(detections,),
                        daemon=True
                    )
                    voice_thread.start()
                elif key == ord('s'):
                    # Stop announcements
                    self.audio.pause_announcements()
                elif key == ord('r'):
                    # Resume announcements
                    self.audio.resume_announcements()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.audio.stop()
        print("Shutdown complete")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("Visual Assist System for Indoor Navigation")
    print("Enhanced with Context-Rich Guidance")
    print("=" * 60)
    
    try:
        system = VisualAssistSystem()
        system.run()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
