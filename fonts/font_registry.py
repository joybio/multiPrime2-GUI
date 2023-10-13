# Add all fonts
import dearpygui.dearpygui as dpg

with dpg.font_registry() as main_font_registry:
    regular_font = dpg.add_font('fonts/Roboto/Roboto-Regular.ttf', 16)
    bold_font = dpg.add_font('fonts/Roboto/Roboto-Bold.ttf', 21)
    score_font = dpg.add_font('fonts/ARCADE.ttf', 35)
