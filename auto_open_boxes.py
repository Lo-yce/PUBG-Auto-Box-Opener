import pyautogui
import cv2
import numpy as np
import time
import os

# 配置参数
CONFIDENCE_THRESHOLD = 0.7  # 模板匹配的置信度阈值
SCAN_INTERVAL = 0.5  # 扫描间隔（秒）
CLICK_DELAY = 0.2  # 点击延迟（秒）
PROCESSING_DELAY = 1.0  # 处理延迟（秒）

# 模板图片路径
TEMPLATES_DIR = 'templates'
OPEN_BUTTON_TEMPLATE = os.path.join(TEMPLATES_DIR, 'open_button.png')
CLOSE_BUTTON_TEMPLATE = os.path.join(TEMPLATES_DIR, 'close_button.png')

# 确保模板目录存在
if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR)
    print(f"创建模板目录: {TEMPLATES_DIR}")
    print("请在该目录中添加 open_button.png 和 close_button.png 模板图片")
    exit(1)

# 检查模板文件是否存在
if not os.path.exists(OPEN_BUTTON_TEMPLATE) or not os.path.exists(CLOSE_BUTTON_TEMPLATE):
    print("缺少模板图片，请在 templates 目录中添加:")
    print("1. open_button.png - 开啟按钮的截图")
    print("2. close_button.png - 關閉按钮的截图")
    exit(1)

# 加载模板图片
open_button_template = cv2.imread(OPEN_BUTTON_TEMPLATE, cv2.IMREAD_GRAYSCALE)
close_button_template = cv2.imread(CLOSE_BUTTON_TEMPLATE, cv2.IMREAD_GRAYSCALE)

# 函数：查找图像中的模板

def find_template(screen, template, threshold=CONFIDENCE_THRESHOLD):
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= threshold:
        template_height, template_width = template.shape
        center_x = max_loc[0] + template_width // 2
        center_y = max_loc[1] + template_height // 2
        return (center_x, center_y), max_val
    return None, 0

# 函数：截图当前屏幕

def capture_screen():
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# 函数：自动开宝箱

def auto_open_boxes():
    print("开始自动开宝箱...")
    print("按 Ctrl+C 停止")
    
    try:
        while True:
            # 截图当前屏幕
            screen = capture_screen()
            
            # 查找开啟按钮
            open_button_pos, open_confidence = find_template(screen, open_button_template)
            
            if open_button_pos:
                print(f"找到开啟按钮，置信度: {open_confidence:.2f}")
                # 点击开啟按钮
                pyautogui.click(open_button_pos[0], open_button_pos[1])
                time.sleep(CLICK_DELAY)
                
                # 等待开宝箱动画
                time.sleep(3.0)
                
                # 截图检查是否需要关闭
                screen = capture_screen()
                close_button_pos, close_confidence = find_template(screen, close_button_template)
                
                if close_button_pos:
                    print(f"找到關閉按钮，置信度: {close_confidence:.2f}")
                    # 点击關閉按钮
                    pyautogui.click(close_button_pos[0], close_button_pos[1])
                    time.sleep(CLICK_DELAY)
                else:
                    print("未找到關閉按钮，继续...")
            else:
                print("未找到开啟按钮，等待...")
            
            # 等待一段时间后再次扫描
            time.sleep(SCAN_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n停止自动开宝箱")

# 主函数
if __name__ == "__main__":
    # 显示提示信息
    print("自动开宝箱脚本")
    print("请确保:")
    print("1. 游戏窗口在前台显示")
    print("2. 已进入 '造型自订' -> '箱子與輪胎' 页面")
    print("3. 宝箱已显示在屏幕上")
    print("\n3秒后开始...")
    
    # 倒计时
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    # 开始自动开宝箱
    auto_open_boxes()