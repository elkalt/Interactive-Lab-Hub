import time
import subprocess
import digitalio
import board
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.st7789 as st7789
import numpy as np
cs_pin = digitalio.DigitalInOut(board.D5) 
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = None

BAUDRATE = 64000000
spi = board.SPI()

disp = st7789.ST7789(
    spi,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
    width=135,
    height=240,
    x_offset=53,
    y_offset=40,
)

height = disp.width
width = disp.height
image = Image.new("RGB", (width, height))
rotation = 90
draw = ImageDraw.Draw(image)
try:
    clock_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except IOError:
    try:
        clock_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except IOError:
        clock_font = ImageFont.load_default()

stats_font = ImageFont.load_default()

backlight = digitalio.DigitalInOut(board.D22)
backlight.switch_to_output()
backlight.value = True

button_a = digitalio.DigitalInOut(board.D23)
button_a.direction = digitalio.Direction.INPUT
button_a.pull = digitalio.Pull.UP

button_b = digitalio.DigitalInOut(board.D24)
button_b.direction = digitalio.Direction.INPUT
button_b.pull = digitalio.Pull.UP

TEST_HOURS = [None, 2, 8, 14, 20]
hour_idx = 0

FOOTER_H = 12
CELL_SIZE = 1
GRID_X = 0
GRID_W = width // CELL_SIZE
GRID_H = (height - FOOTER_H) // CELL_SIZE
center_y, center_x = GRID_H / 2.0, GRID_W / 2.0
y_coords, x_coords = np.ogrid[:GRID_H, :GRID_W]
dist_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
max_dist = np.max(dist_from_center)

WAVE_SPEED = 5.0     
PAUSE_CYCLES = 1

transition_phase = 'idle'
wave_radius = 0.0
pause_counter = 0

grid = (np.random.rand(GRID_H, GRID_W) < 0.25).astype(np.uint8)
last_minute = -1
current_clock_str = ""
clock_mask = np.zeros((GRID_H, GRID_W), dtype=bool)

def parse_rule(rule_str: str):
    birth, survival = set(), set()
    for part in rule_str.upper().strip().split('/'):
        if part.startswith('B'):
            birth = {int(c) for c in part[1:] if c.isdigit()}
        elif part.startswith('S'):
            survival = {int(c) for c in part[1:] if c.isdigit()}
    return birth, survival

def step_life(g: np.ndarray, birth: set, survival: set) -> np.ndarray:
    neighbors = (
        np.roll(g, 1, 0) + np.roll(g, -1, 0) +
        np.roll(g, 1, 1) + np.roll(g, -1, 1) +
        np.roll(np.roll(g, 1, 0), 1, 1) +
        np.roll(np.roll(g, 1, 0), -1, 1) +
        np.roll(np.roll(g, -1, 0), 1, 1) +
        np.roll(np.roll(g, -1, 0), -1, 1)
    )
    birth_mask = np.isin(neighbors, list(birth)) & (g == 0)
    survival_mask = np.isin(neighbors, list(survival)) & (g == 1)
    return np.where(birth_mask | survival_mask, 1, 0).astype(np.uint8)

def get_circadian_state(hour: int):
    if 0 <= hour < 6:
        return "B3/S23", "#4A6FA5", 0.15
    elif 6 <= hour < 12:
        return "B356/S23", "#FFB703", 0.30
    elif 12 <= hour < 18:
        return "B357/S1358", "#06D6A0", 0.35
    else:
        return "B378/S235678", "#D88373", 0.20

def generate_clock_mask(text: str) -> np.ndarray:
    """Renders text into a 1-bit monochrome mask aligned with the grid coordinates."""
    mask_img = Image.new("1", (GRID_W, GRID_H), 0)
    m_draw = ImageDraw.Draw(mask_img)
    
    bbox = clock_font.getbbox(text)
    t_width = bbox[2] - bbox[0]
    t_height = bbox[3] - bbox[1]
    
    cx = (GRID_W - t_width) // 2
    cy = (GRID_H - t_height) // 2 - bbox[1]
    
    m_draw.text((cx, cy), text, font=clock_font, fill=1)
    return np.array(mask_img, dtype=bool)

while True:
    now = time.localtime()
    
    if not button_a.value:
        hour_idx = (hour_idx + 1) % len(TEST_HOURS)
        transition_phase = 'clearing'
        wave_radius = 0.0
        time.sleep(0.2)

    if not button_b.value:
        hour_idx = 0
        transition_phase = 'clearing'
        wave_radius = 0.0
        time.sleep(0.2)

    if last_minute == -1:
        last_minute = now.tm_min
    elif now.tm_min != last_minute and transition_phase == 'idle':
        last_minute = now.tm_min
        transition_phase = 'clearing'
        wave_radius = 0.0

    active_hour = TEST_HOURS[hour_idx] if TEST_HOURS[hour_idx] is not None else now.tm_hour
    rule_str, cell_color, seed_density = get_circadian_state(active_hour)
    birth, survival = parse_rule(rule_str)

    time_text = f"{active_hour:02d}:{now.tm_min:02d}"
    if time_text != current_clock_str:
        current_clock_str = time_text
        clock_mask = generate_clock_mask(current_clock_str)

    if transition_phase == 'clearing':
        grid = step_life(grid, birth, survival)
        wave_radius += WAVE_SPEED
        grid[dist_from_center <= wave_radius] = 0

        if wave_radius >= max_dist:
            grid[:] = 0
            transition_phase = 'pause'
            pause_counter = 0

    elif transition_phase == 'pause':
        pause_counter += 1
        if pause_counter >= PAUSE_CYCLES:
            transition_phase = 'seeding'
            wave_radius = 0.0

    elif transition_phase == 'seeding':
        prev_r = wave_radius
        wave_radius += WAVE_SPEED
        
        ring_mask = (dist_from_center > prev_r) & (dist_from_center <= wave_radius)
        spawn_seeds = (np.random.rand(GRID_H, GRID_W) < seed_density).astype(np.uint8)
        grid[ring_mask] = spawn_seeds[ring_mask]

        grid = step_life(grid, birth, survival)

        if wave_radius >= max_dist:
            transition_phase = 'idle'

    else:
        grid = step_life(grid, birth, survival)

    grid[clock_mask] = 1

    draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))

    sim_cells_y, sim_cells_x = np.nonzero(grid & (~clock_mask))
    for r, c in zip(sim_cells_y, sim_cells_x):
        px = GRID_X + c * CELL_SIZE
        py = r * CELL_SIZE
        draw.point((px, py), fill=cell_color)

    clock_y, clock_x = np.nonzero(clock_mask)
    for r, c in zip(clock_y, clock_x):
        px = GRID_X + c * CELL_SIZE
        py = r * CELL_SIZE
        draw.point((px, py), fill="#FFFFFF")

    if TEST_HOURS[hour_idx] is None:
        footer_text = f"Rule: {rule_str} (LIVE)"
        sec_text = time.strftime(":%S")
        sec_bbox = stats_font.getbbox(sec_text)
        draw.text((width - (sec_bbox[2] - sec_bbox[0]) - 2, height - 12), sec_text, font=stats_font, fill="#55FFFF")
    else:
        footer_text = f"Rule: {rule_str} [SIM {active_hour:02d}:00]"

    draw.text((2, height - 12), footer_text, font=stats_font, fill="#AAAAAA")

    disp.image(image, rotation)
    time.sleep(0.01)
