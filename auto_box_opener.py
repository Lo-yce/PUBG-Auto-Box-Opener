import pyautogui
import cv2
import numpy as np
import time
import os
import json
import logging
import keyboard
from datetime import datetime

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

DEFAULT_CONFIG = {
    "confidence_threshold": 0.7,
    "scan_interval": 0.5,
    "click_delay": 0.2,
    "animation_timeout": 8.0,
    "animation_poll_interval": 0.3,
    "retry_count": 3,
    "retry_delay": 1.0,
    "capture_region": None,
    "open_coords": None,
    "close_coords": None,
    "chest_coords": None,
    "log_to_file": True,
    "multi_scale": True,
    "scale_range": [0.8, 1.2],
    "scale_steps": 5
}

CONFIG_FILE = "box_opener_config.json"
LOG_FILE = "box_opener.log"
TEMPLATES_DIR = "templates"


class AutoBoxOpener:
    def __init__(self, config_path=CONFIG_FILE):
        self.config = self._load_config(config_path)
        self.running = False
        self.paused = False

        self.templates_dir = TEMPLATES_DIR
        self.open_button_path = os.path.join(self.templates_dir, "open_button.png")
        self.close_button_path = os.path.join(self.templates_dir, "close_button.png")
        self.chest_image_path = os.path.join(self.templates_dir, "chest.png")

        self.open_template = None
        self.close_template = None
        self.chest_template = None

        self.total_opened = 0
        self.failed_count = 0
        self.start_time = None

        self._setup_logging()

        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)
            self.log(f"创建模板目录: {self.templates_dir}")

    def _load_config(self, config_path):
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                merged = {**DEFAULT_CONFIG, **saved}
                return merged
            except Exception:
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    def _save_config(self, config_path=CONFIG_FILE):
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.log(f"配置已保存到 {config_path}")
        except Exception as e:
            self.log(f"保存配置失败: {e}")

    def _setup_logging(self):
        self.logger = logging.getLogger("AutoBoxOpener")
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        console_handler.setFormatter(console_fmt)
        self.logger.addHandler(console_handler)

        if self.config.get("log_to_file", True):
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_fmt = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_fmt)
            self.logger.addHandler(file_handler)

    def log(self, msg, level="info"):
        getattr(self.logger, level)(msg)

    def load_templates(self):
        loaded = True
        if os.path.exists(self.open_button_path):
            self.open_template = cv2.imread(self.open_button_path, cv2.IMREAD_GRAYSCALE)
            if self.open_template is None:
                self.log(f"无法读取模板: {self.open_button_path}", "error")
                loaded = False
        else:
            self.log(f"缺少模板: {self.open_button_path}", "warning")
            loaded = False

        if os.path.exists(self.close_button_path):
            self.close_template = cv2.imread(self.close_button_path, cv2.IMREAD_GRAYSCALE)
            if self.close_template is None:
                self.log(f"无法读取模板: {self.close_button_path}", "error")
                loaded = False
        else:
            self.log(f"缺少模板: {self.close_button_path}", "warning")
            loaded = False

        if os.path.exists(self.chest_image_path):
            self.chest_template = cv2.imread(self.chest_image_path, cv2.IMREAD_GRAYSCALE)
            if self.chest_template is None:
                self.log(f"无法读取模板: {self.chest_image_path}", "error")

        return loaded

    def capture_screen(self, region=None):
        if region is None:
            region = self.config.get("capture_region")
        screenshot = pyautogui.screenshot(region=region)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def find_template(self, screen, template, threshold=None):
        if template is None:
            return None, 0
        if threshold is None:
            threshold = self.config["confidence_threshold"]

        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        if self.config.get("multi_scale", True):
            return self._find_template_multiscale(screen_gray, template, threshold)

        result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            h, w = template.shape
            cx = max_loc[0] + w // 2
            cy = max_loc[1] + h // 2
            return (cx, cy), max_val
        return None, 0

    def _find_template_multiscale(self, screen_gray, template, threshold):
        scale_min, scale_max = self.config.get("scale_range", [0.8, 1.2])
        scale_steps = self.config.get("scale_steps", 5)
        scales = np.linspace(scale_min, scale_max, scale_steps)

        best_val = 0
        best_pos = None
        th, tw = template.shape

        for scale in scales:
            new_w = int(tw * scale)
            new_h = int(th * scale)
            if new_w < 10 or new_h < 10:
                continue
            if new_w > screen_gray.shape[1] or new_h > screen_gray.shape[0]:
                continue

            scaled_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(screen_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                cx = max_loc[0] + new_w // 2
                cy = max_loc[1] + new_h // 2
                best_pos = (cx, cy)

        if best_val >= threshold and best_pos is not None:
            return best_pos, best_val
        return None, 0

    def find_by_color(self, screen, target_hsv, tolerance=30):
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        target_hsv = np.array(target_hsv, dtype=np.int32)
        lower = np.array([
            max(0, target_hsv[0] - tolerance),
            max(0, target_hsv[1] - tolerance),
            max(0, target_hsv[2] - tolerance)
        ], dtype=np.uint8)
        upper = np.array([
            min(179, target_hsv[0] + tolerance),
            min(255, target_hsv[1] + tolerance),
            min(255, target_hsv[2] + tolerance)
        ], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                x, y, w, h = cv2.boundingRect(largest)
                return (x + w // 2, y + h // 2)
        return None

    def _wait_for_close_button(self, timeout=None):
        if timeout is None:
            timeout = self.config["animation_timeout"]
        poll_interval = self.config["animation_poll_interval"]
        start = time.time()

        while time.time() - start < timeout:
            if keyboard.is_pressed("F12"):
                return None
            if self.paused:
                time.sleep(0.1)
                continue

            screen = self.capture_screen()
            pos, conf = self.find_template(screen, self.close_template)
            if pos:
                elapsed = time.time() - start
                self.log(f"检测到关闭按钮 (耗时 {elapsed:.1f}s, 置信度 {conf:.2f})")
                return pos
            time.sleep(poll_interval)

        self.log(f"等待关闭按钮超时 ({timeout}s)", "warning")
        return None

    def _click_with_retry(self, x, y, description="目标", retries=None):
        if retries is None:
            retries = self.config["retry_count"]
        for i in range(retries):
            pyautogui.click(x, y)
            time.sleep(self.config["click_delay"])
            screen = self.capture_screen()
            return True
        self.log(f"点击 {description} 失败，已重试 {retries} 次", "warning")
        return False

    def open_box_by_template(self):
        screen = self.capture_screen()

        chest_pos = None
        if self.chest_template is not None:
            chest_pos, chest_conf = self.find_template(screen, self.chest_template)
            if chest_pos:
                self.log(f"找到宝箱，置信度: {chest_conf:.2f}")
                pyautogui.rightClick(chest_pos[0], chest_pos[1])
                time.sleep(0.5)

        screen = self.capture_screen()
        open_pos, open_conf = self.find_template(screen, self.open_template)

        if not open_pos:
            if self.config.get("chest_coords"):
                cx, cy = self.config["chest_coords"]
                self.log("使用配置坐标右键点击宝箱")
                pyautogui.rightClick(cx, cy)
                time.sleep(0.5)
                screen = self.capture_screen()
                open_pos, open_conf = self.find_template(screen, self.open_template)

        if open_pos:
            self.log(f"找到开启按钮，置信度: {open_conf:.2f}")
            pyautogui.click(open_pos[0], open_pos[1])
            time.sleep(self.config["click_delay"])

            close_pos = self._wait_for_close_button()
            if close_pos:
                pyautogui.click(close_pos[0], close_pos[1])
                time.sleep(self.config["click_delay"])
                self.total_opened += 1
                self.log(f"成功开箱! 已开 {self.total_opened} 个")
                return True
            else:
                self.failed_count += 1
                self.log("开箱失败: 未检测到关闭按钮", "warning")
                pyautogui.press("escape")
                time.sleep(0.5)
                return False
        else:
            self.log("未找到开启按钮", "debug")
            return False

    def open_box_by_coordinates(self):
        open_coords = self.config.get("open_coords")
        close_coords = self.config.get("close_coords")
        chest_coords = self.config.get("chest_coords")

        if not open_coords or not close_coords:
            self.log("缺少坐标配置", "error")
            return False

        if chest_coords:
            pyautogui.rightClick(chest_coords[0], chest_coords[1])
            time.sleep(0.5)

        pyautogui.click(open_coords[0], open_coords[1])
        time.sleep(self.config["click_delay"])

        close_pos = self._wait_for_close_button()
        if close_pos:
            pyautogui.click(close_pos[0], close_pos[1])
            time.sleep(self.config["click_delay"])
            self.total_opened += 1
            self.log(f"成功开箱! 已开 {self.total_opened} 个")
            return True
        else:
            pyautogui.click(close_coords[0], close_coords[1])
            time.sleep(self.config["click_delay"])
            self.total_opened += 1
            self.log(f"开箱完成(坐标模式) 已开 {self.total_opened} 个")
            return True

    def print_stats(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            rate = self.total_opened / elapsed * 60 if elapsed > 0 else 0
            self.log(f"统计: 已开 {self.total_opened} 个 | 失败 {self.failed_count} 次 | "
                     f"用时 {mins}分{secs}秒 | 速率 {rate:.1f}个/分钟")

    def start(self, method="template"):
        self.running = True
        self.paused = False
        self.start_time = time.time()
        self.total_opened = 0
        self.failed_count = 0

        self.log("开始自动开宝箱...")
        self.log("按 F12 停止 | 按 F11 暂停/恢复")
        self.log(f"模式: {method}")

        if method == "template" and not self.load_templates():
            self.log("错误: 缺少必要的模板图片，请检查 templates 目录", "error")
            return

        try:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue

                if method == "template":
                    self.open_box_by_template()
                elif method == "coordinates":
                    self.open_box_by_coordinates()

                if keyboard.is_pressed("F12"):
                    self.log("用户按下 F12，停止运行")
                    break

                if keyboard.is_pressed("F11"):
                    self.paused = not self.paused
                    state = "暂停" if self.paused else "恢复"
                    self.log(f"已{state}")
                    time.sleep(0.3)

                time.sleep(self.config["scan_interval"])

        except Exception as e:
            self.log(f"运行错误: {e}", "error")
        finally:
            self.running = False
            self.print_stats()

    def stop(self):
        self.running = False


def get_mouse_position(prompt="移动鼠标到目标位置"):
    print(f"\n{prompt}")
    print("按 Enter 确认坐标 | 按 Esc 取消")
    time.sleep(1)

    while True:
        x, y = pyautogui.position()
        print(f"当前鼠标位置: ({x}, {y})", end="\r")

        if keyboard.is_pressed("enter"):
            print(f"\n已记录坐标: ({x}, {y})")
            time.sleep(0.3)
            return (x, y)
        elif keyboard.is_pressed("esc"):
            print("\n已取消")
            return None
        time.sleep(0.05)


def auto_capture_template(name, save_path):
    print(f"\n--- 截取模板: {name} ---")
    print("将鼠标移到目标区域左上角，按 Enter 键")
    top_left = get_mouse_position(f"移动鼠标到 {name} 的左上角")
    if not top_left:
        return False

    print("将鼠标移到目标区域右下角，按 Enter 键")
    bottom_right = get_mouse_position(f"移动鼠标到 {name} 的右下角")
    if not bottom_right:
        return False

    x1, y1 = top_left
    x2, y2 = bottom_right
    region = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    if region[2] < 5 or region[3] < 5:
        print("选区太小，请重新截取")
        return False

    screenshot = pyautogui.screenshot(region=region)
    screenshot.save(save_path)
    print(f"模板已保存: {save_path}")
    return True


def show_menu():
    print("\n" + "=" * 55)
    print("       PUBG 自动开宝箱工具 v2.0")
    print("=" * 55)
    print("  1. 模板匹配模式 (推荐)")
    print("  2. 固定坐标模式")
    print("  3. 截取模板图片 (辅助工具)")
    print("  4. 获取鼠标坐标 (辅助工具)")
    print("  5. 查看当前配置")
    print("  6. 修改配置参数")
    print("  7. 帮助")
    print("=" * 55)

    choice = input("请输入选项编号: ").strip()
    return choice


def show_config(config):
    print("\n当前配置:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 40)


def edit_config(config):
    print("\n可修改的参数 (直接回车跳过):")
    params = {
        "confidence_threshold": ("置信度阈值 (0-1)", float),
        "scan_interval": ("扫描间隔(秒)", float),
        "click_delay": ("点击延迟(秒)", float),
        "animation_timeout": ("动画超时(秒)", float),
        "retry_count": ("重试次数", int),
        "multi_scale": ("多尺度匹配 (true/false)", lambda x: x.lower() == "true"),
    }

    for key, (desc, type_fn) in params.items():
        val = input(f"  {desc} [{config[key]}]: ").strip()
        if val:
            try:
                config[key] = type_fn(val)
                print(f"    -> 已设为 {config[key]}")
            except ValueError:
                print(f"    -> 输入无效，保持原值")

    return config


def show_help():
    print("\n帮助信息:")
    print("-" * 55)
    print("【模板匹配模式】")
    print("  需要在 templates/ 目录中添加:")
    print("  - open_button.png  开启按钮截图")
    print("  - close_button.png 关闭按钮截图")
    print("  - chest.png        宝箱图标截图 (可选)")
    print("  可使用选项3辅助截取模板")
    print()
    print("【固定坐标模式】")
    print("  需要预先获取按钮坐标")
    print("  可使用选项4辅助获取坐标")
    print("  坐标会自动保存到配置文件")
    print()
    print("【快捷键】")
    print("  F12 - 停止运行")
    print("  F11 - 暂停/恢复")
    print()
    print("【优化说明】")
    print("  - 智能等待: 轮询检测关闭按钮，而非固定等待")
    print("  - 多尺度匹配: 适应不同分辨率")
    print("  - 开箱统计: 实时显示开箱数量和速率")
    print("  - 日志记录: 运行日志保存到 box_opener.log")
    print("  - 配置持久化: 参数保存到 box_opener_config.json")
    print("-" * 55)


if __name__ == "__main__":
    opener = AutoBoxOpener()

    while True:
        choice = show_menu()

        if choice == "1":
            print("\n模板匹配模式")
            if not opener.load_templates():
                print("缺少必要模板图片，是否现在截取? (y/n)")
                if input().strip().lower() == "y":
                    if not auto_capture_template("开启按钮", opener.open_button_path):
                        continue
                    if not auto_capture_template("关闭按钮", opener.close_button_path):
                        continue
                else:
                    continue

            print("3秒后开始...")
            for i in range(3, 0, -1):
                print(f"  {i}...")
                time.sleep(1)
            opener.start(method="template")
            break

        elif choice == "2":
            print("\n固定坐标模式")
            saved_open = opener.config.get("open_coords")
            saved_close = opener.config.get("close_coords")

            if saved_open and saved_close:
                print(f"已保存的坐标: 开启={saved_open}, 关闭={saved_close}")
                print("是否使用已保存的坐标? (y/n)")
                if input().strip().lower() == "y":
                    pass
                else:
                    saved_open = None

            if not saved_open:
                open_coords = get_mouse_position("移动鼠标到开启按钮位置")
                if not open_coords:
                    continue
                opener.config["open_coords"] = list(open_coords)

            if not saved_close:
                close_coords = get_mouse_position("移动鼠标到关闭按钮位置")
                if not close_coords:
                    continue
                opener.config["close_coords"] = list(close_coords)

            chest_coords = get_mouse_position("移动鼠标到宝箱位置 (可选，Esc跳过)")
            if chest_coords:
                opener.config["chest_coords"] = list(chest_coords)

            opener._save_config()

            print("3秒后开始...")
            for i in range(3, 0, -1):
                print(f"  {i}...")
                time.sleep(1)
            opener.start(method="coordinates")
            break

        elif choice == "3":
            print("\n模板截取工具")
            print("请确保游戏窗口在前台显示")
            time.sleep(1)
            auto_capture_template("开启按钮", opener.open_button_path)
            auto_capture_template("关闭按钮", opener.close_button_path)
            print("\n是否截取宝箱图标? (可选，用于自动右键) (y/n)")
            if input().strip().lower() == "y":
                auto_capture_template("宝箱图标", opener.chest_image_path)

        elif choice == "4":
            print("\n坐标获取工具")
            pos = get_mouse_position("移动鼠标到目标位置")
            if pos:
                print(f"坐标: ({pos[0]}, {pos[1]})")
                print("是否保存到配置? (y/n)")
                if input().strip().lower() == "y":
                    label = input("标签 (open/close/chest): ").strip()
                    key = f"{label}_coords"
                    if key in DEFAULT_CONFIG:
                        opener.config[key] = list(pos)
                        opener._save_config()

        elif choice == "5":
            show_config(opener.config)

        elif choice == "6":
            opener.config = edit_config(opener.config)
            opener._save_config()

        elif choice == "7":
            show_help()

        else:
            print("无效选项，请重新输入")
