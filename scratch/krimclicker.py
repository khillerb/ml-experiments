import pyautogui
import time


def hold_mouse_down_and_release():
    while True:
        pyautogui.mouseDown()  # Hold the mouse button down
        time.sleep(2)  # Wait for 2 seconds
        pyautogui.mouseUp()  # Release the mouse button
        time.sleep(1)  # Wait for 1 second


if __name__ == "__main__":
    hold_mouse_down_and_release()
