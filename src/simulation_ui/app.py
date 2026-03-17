from __future__ import annotations

import io
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import pandas as pd

import streamlit as st
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import load_yolo, detect_objects, count_summary
from src.graph_routing.grid_builder import (
    build_grid_from_detections,
    pixel_to_cell,
    cell_to_pixel_center,
    nearest_free_cell,
)
from src.graph_routing.path_planner import astar, k_alternative_paths
from src.fuzzy.fuzzy_eval import fuzzy_score

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def default_model_path() -> Path:
    return PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"


def run_dir() -> Path:
    d = PROJECT_ROOT / "runs" / "ui_outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def choose_source_pixel(img_w: int, img_h: int, option: str) -> Tuple[int, int]:
    pad = 20
    if option == "Top-Left":
        return (pad, pad)
    if option == "Top-Right":
        return (img_w - pad, pad)
    if option == "Bottom-Left":
        return (pad, img_h - pad)
    if option == "Bottom-Right":
        return (img_w - pad, img_h - pad)
    return (img_w // 2, img_h // 2)


def pil_png_bytes(img: Image.Image) -> bytes:
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return buff.getvalue()


def draw_source_marker(img: Image.Image, src_xy: Tuple[int, int]) -> Image.Image:
    im = img.copy()
    draw = ImageDraw.Draw(im)
    x, y = src_xy
    draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(0, 255, 255))
    draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 0, 0))
    return im

def highlight_target_houses(
    base_img: Image.Image,
    detections: Dict,
    target_indices: List[int],
    color=(0, 0, 180),  # dark blue
) -> Image.Image:
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    for idx in target_indices:
        if idx >= len(detections["houses"]):
            continue

        h = detections["houses"][idx]
        draw.rectangle(
            [h["x1"], h["y1"], h["x2"], h["y2"]],
            outline=color,
            width=4,
        )

    return img


def overlay_paths(
    base_img: Image.Image,
    paths: List[List[Tuple[int, int]]],
    cell_size: int,
    width: int = 3,
) -> Image.Image:
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    colors = [
        (255, 160, 0),
        (0, 200, 255),
        (255, 120, 120),
        (200, 120, 255),
        (120, 255, 170),
        (255, 255, 0),
        (0, 255, 0),
    ]

    for i, path_cells in enumerate(paths):
        if not path_cells or len(path_cells) < 2:
            continue
        pts = [cell_to_pixel_center(r, c, cell_size) for (r, c) in path_cells]
        col = colors[i % len(colors)]
        draw.line(pts, fill=col, width=width)

    return img


def overlay_best_path(
    base_img: Image.Image,
    path_cells: List[Tuple[int, int]],
    cell_size: int,
) -> Image.Image:
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    if not path_cells:
        return img

    pts = [cell_to_pixel_center(r, c, cell_size) for (r, c) in path_cells]
    if len(pts) >= 2:
        draw.line(pts, fill=(0, 255, 0), width=5)

    sx, sy = pts[0]
    ex, ey = pts[-1]
    draw.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], fill=(0, 255, 0))
    draw.ellipse([ex - 7, ey - 7, ex + 7, ey + 7], fill=(255, 255, 0))
    return img


def build_pdf_report(
    title: str,
    meta: Dict[str, str],
    input_img: Image.Image,
    det_img: Image.Image,
    all_routes_img: Image.Image,
    best_routes_img: Image.Image,
    metrics_rows: List[Dict[str, str]],
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, H - 50, title)
    c.setFont("Helvetica", 10)
    c.drawString(40, H - 66, f"Generated: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}")

    y = H - 95
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Run Details")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in meta.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 14

    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Outputs")
    y -= 10

    def draw_img(label: str, img: Image.Image, x: int, y_top: int, width: int):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y_top, label)

        img_reader = ImageReader(io.BytesIO(pil_png_bytes(img)))
        iw, ih = img.size
        height = int(width * (ih / iw))
        c.drawImage(img_reader, x, y_top - height - 10, width=width, height=height, mask="auto")
        return y_top - height - 25

    left_x, right_x = 40, 300
    small_w = 240

    y1 = draw_img("Input Blueprint", input_img, left_x, y, small_w)
    y2 = draw_img("Detection Output", det_img, right_x, y, small_w)

    y = min(y1, y2)
    y = draw_img("All Routes Overlay", all_routes_img, 40, y, 500)
    y = draw_img("Best Route(s) Overlay", best_routes_img, 40, y, 500)

    if y < 240:
        c.showPage()
        y = H - 70

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Metrics (Per House)")
    y -= 20

    c.setFont("Helvetica-Bold", 9)
    headers = ["House#", "PathCells", "RiskCells", "Cost↓", "Pressure↑", "Safety↑"]
    x = [40, 90, 160, 240, 310, 400]
    for i, hname in enumerate(headers):
        c.drawString(x[i], y, hname)
    y -= 10
    c.line(40, y, 520, y)
    y -= 14

    c.setFont("Helvetica", 9)
    for row in metrics_rows:
        c.drawString(x[0], y, str(row.get("house", "")))
        c.drawString(x[1], y, str(row.get("path_cells", "")))
        c.drawString(x[2], y, str(row.get("risk_cells", "")))
        c.drawString(x[3], y, str(row.get("cost", "")))
        c.drawString(x[4], y, str(row.get("pressure", "")))
        c.drawString(x[5], y, str(row.get("safety", "")))
        y -= 14
        if y < 70:
            c.showPage()
            y = H - 70

    c.save()
    return buf.getvalue()


st.set_page_config(page_title="Smart-Flow Simulation Tool", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
h1 {margin-bottom: 0.2rem;}
small, .stCaption {color: #9aa0a6;}
.card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Smart-Flow")
st.subheader("Explainable Hydraulic Network Simulation Tool")
st.caption("Upload blueprint, detect houses and obstacles, generate routes, evaluate the best route, and export results.")

with st.sidebar:
    st.header("⚙️ Settings")

    model_path_str = str(default_model_path())
    st.sidebar.text_input("Model File", value="best.pt", disabled=True)

    conf = st.slider("Detection Confidence", 0.05, 0.95, 0.25, 0.05, key="conf_slider")

    cell_size = st.select_slider(
        "Grid Cell Size (px)", options=[4, 6, 8, 10, 12, 16], value=8, key="cell_size_slider"
    )

    obstacle_pad = st.slider("Obstacle Padding (px)", 0, 25, 6, 1, key="obstacle_pad_slider")

    source_option = st.selectbox(
        "Water Source (Simulation)",
        ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"],
        key="source_option_select",
    )

    routing_mode = st.radio("Routing Mode", ["Single House", "Multiple Houses"], index=0, key="routing_mode_radio")

    k_routes = st.slider("How many alternative routes to show", 2, 10, 5, 1, key="k_routes_slider")


uploaded = st.file_uploader("Upload a map image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="map_uploader")
if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_w, img_h = pil_img.size

@st.cache_resource
def cached_model(path_str: str):
    return load_yolo(path_str)

if not Path(model_path_str).exists():
    st.error(f" Model not found at:\n{model_path_str}")
    st.stop()

model = cached_model(model_path_str)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader(" Uploaded Blueprint")
    st.image(pil_img, use_container_width=True)
    st.markdown(f"<div class='card'><b>Resolution:</b> {img_w} × {img_h}</div>", unsafe_allow_html=True)

with col2:
    st.subheader("Detected Houses & Obstacles")
    det_res = detect_objects(model, pil_img, conf=conf)
    detections = det_res.detections
    det_img = det_res.annotated
    st.image(det_img, use_container_width=True)

    summary = count_summary(detections)
    st.markdown(
        "<div class='card'>"
        f"<b>Total:</b> {summary['total']}<br>"
        f"<b>Houses:</b> {summary['houses']}<br>"
        f"<b>Obstacles:</b> {summary['obstacles']}"
        "</div>",
        unsafe_allow_html=True,
    )

det_out_path = run_dir() / f"detected_{ts()}.png"
det_img.save(det_out_path)

st.download_button(
    "⬇️ Download Detection Output (PNG)",
    data=pil_png_bytes(det_img),
    file_name=det_out_path.name,
    mime="image/png",
    key="download_detection_btn",
)

st.divider()
st.header("Route Generation")

if len(detections.get("houses", [])) == 0:
    st.warning("No houses detected. Try lowering confidence or upload another map.")
    st.stop()

grid_res = build_grid_from_detections(
    img_w=img_w,
    img_h=img_h,
    detections=detections,
    cell_size=int(cell_size),
    obstacle_padding_px=int(obstacle_pad),
)

src_x, src_y = choose_source_pixel(img_w, img_h, source_option)
start_cell = pixel_to_cell(src_x, src_y, grid_res.cell_size)
start_cell = nearest_free_cell(grid_res.grid, start_cell)

st.markdown(f"<div class='card'><b>Source (simulation tank):</b> ({src_x}, {src_y})</div>", unsafe_allow_html=True)

metrics_rows: List[Dict[str, str]] = []

base_with_source = draw_source_marker(pil_img, (src_x, src_y))

all_routes_img = base_with_source.copy()
best_routes_img = base_with_source.copy()

if routing_mode == "Single House":
    house_index = st.selectbox(
        "Select Target House",
        options=list(range(len(detections["houses"]))),
        format_func=lambda i: f"House #{i+1} (conf {detections['houses'][i]['conf']:.2f})",
        key="single_house_select",
    )
    base_with_source = highlight_target_houses(base_with_source, detections, [house_index], color=(0, 0, 180))

    target = detections["houses"][house_index]
    target_cx = int((target["x1"] + target["x2"]) / 2)
    target_cy = int((target["y1"] + target["y2"]) / 2)

    goal_cell = pixel_to_cell(target_cx, target_cy, grid_res.cell_size)
    goal_cell = nearest_free_cell(grid_res.grid, goal_cell)

    alt_paths = k_alternative_paths(grid_res.grid, start_cell, goal_cell, k=int(k_routes), penalty_step=2.0, spread_radius=2)

    if not alt_paths:
        st.error(" No path found. Try changing source or reduce obstacle padding.")
        st.stop()

    all_routes_img = overlay_paths(base_with_source, alt_paths, grid_res.cell_size, width=3)

    best_path = alt_paths[0]
    best_routes_img = overlay_best_path(base_with_source, best_path, grid_res.cell_size)

    score = fuzzy_score(best_path, grid_res.grid)
    metrics_rows.append(
        {
            "house": str(house_index + 1),
            "path_cells": str(score.get("path_cells", len(best_path))),
            "risk_cells": str(score.get("risk_cells", 0)),
            "cost": f"{score.get('cost_score', 0):.3f}",
            "pressure": f"{score.get('pressure_efficiency', 0):.3f}",
            "safety": f"{score.get('safety_score', 0):.3f}",
        }
    )

    st.subheader(" All Routes (Single House)")
    st.image(all_routes_img, use_container_width=True)

    st.subheader(" Best Route (Single House)")
    st.image(best_routes_img, use_container_width=True)

else:
    max_show = min(len(detections["houses"]), 200)
    selected = st.multiselect(
        "Select Multiple Target Houses",
        options=list(range(max_show)),
        default=list(range(min(3, max_show))),
        format_func=lambda i: f"House #{i+1} (conf {detections['houses'][i]['conf']:.2f})",
        key="multi_house_select",
    )

    base_with_source = highlight_target_houses(base_with_source, detections, selected, color=(0, 0, 180))

    if not selected:
        st.info("Select at least one house.")
        st.stop()

    all_paths_flat: List[List[Tuple[int, int]]] = []
    best_paths: List[List[Tuple[int, int]]] = []

    for idx in selected:
        h = detections["houses"][idx]
        cx = int((h["x1"] + h["x2"]) / 2)
        cy = int((h["y1"] + h["y2"]) / 2)

        goal_cell = pixel_to_cell(cx, cy, grid_res.cell_size)
        goal_cell = nearest_free_cell(grid_res.grid, goal_cell)

        alt = k_alternative_paths(grid_res.grid, start_cell, goal_cell, k=int(k_routes), penalty_step=2.0, spread_radius=2)

        if not alt:
            metrics_rows.append(
                {"house": str(idx + 1), "path_cells": "NO PATH", "risk_cells": "-", "cost": "-", "pressure": "-", "safety": "-"}
            )
            continue

        all_paths_flat.extend(alt)
        best_paths.append(alt[0])

        score = fuzzy_score(alt[0], grid_res.grid)
        metrics_rows.append(
            {
                "house": str(idx + 1),
                "path_cells": str(score.get("path_cells", len(alt[0]))),
                "risk_cells": str(score.get("risk_cells", 0)),
                "cost": f"{score.get('cost_score', 0):.3f}",
                "pressure": f"{score.get('pressure_efficiency', 0):.3f}",
                "safety": f"{score.get('safety_score', 0):.3f}",
            }
        )

    if not best_paths:
        st.error(" No paths found for selected houses. Try another source or reduce obstacle padding.")
        st.stop()

    all_routes_img = overlay_paths(base_with_source, all_paths_flat, grid_res.cell_size, width=2)
    best_routes_img = overlay_paths(base_with_source, best_paths, grid_res.cell_size, width=5)

    st.subheader(" All Routes (Multiple Houses)")
    st.image(all_routes_img, use_container_width=True)

    st.subheader(" Best Routes (Multiple Houses)")
    st.image(best_routes_img, use_container_width=True)


all_routes_out_path = run_dir() / f"all_routes_{ts()}.png"
best_routes_out_path = run_dir() / f"best_routes_{ts()}.png"
all_routes_img.save(all_routes_out_path)
best_routes_img.save(best_routes_out_path)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download All Routes (PNG)",
        data=pil_png_bytes(all_routes_img),
        file_name=all_routes_out_path.name,
        mime="image/png",
        key="download_all_routes_btn",
    )
with c2:
    st.download_button(
        "⬇️ Download Best Route(s) (PNG)",
        data=pil_png_bytes(best_routes_img),
        file_name=best_routes_out_path.name,
        mime="image/png",
        key="download_best_routes_btn",
    )

st.subheader("Detailed Metrics")
df = pd.DataFrame(metrics_rows)
df = df.rename(columns={
    "house": "House No",
    "path_cells": "Path Length",
    "risk_cells": "Risk Cells",
    "cost": "Cost Score",
    "pressure": "Pressure Efficiency",
    "safety": "Safety Score"
})
st.dataframe(df, use_container_width=True)

st.divider()
st.header("Export Results")

meta = {
    "Model": Path(model_path_str).name,
    "Confidence": str(conf),
    "Cell size": str(cell_size),
    "Obstacle padding": str(obstacle_pad),
    "Source": source_option,
    "Routing mode": routing_mode,
    "K alternatives": str(k_routes),
}

pdf_bytes = build_pdf_report(
    title="Smart-Flow Simulation Report",
    meta=meta,
    input_img=pil_img,
    det_img=det_img,
    all_routes_img=all_routes_img,
    best_routes_img=best_routes_img,
    metrics_rows=metrics_rows,
)

st.download_button(
    "⬇️ Download PDF Report",
    data=pdf_bytes,
    file_name=f"smartflow_report_{ts()}.pdf",
    mime="application/pdf",
    key="download_pdf_btn",
)

st.markdown("---")
st.caption("Smart-Flow | AI-based pipeline routing and fuzzy evaluation system for hydraulic network planning.")