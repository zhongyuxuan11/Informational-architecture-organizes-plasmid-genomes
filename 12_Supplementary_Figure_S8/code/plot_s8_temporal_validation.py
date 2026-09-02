"""Plot the host-chromosome and temporal-validation panels shown in V6 S8."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
TABLES = PACKAGE / "tables"
FIGURES = ROOT / "GitHub_upload_V5_figures_20260831_figures_only" / PACKAGE.name
SVG = FIGURES / "Supplementary_Figure_S8_source.svg"
PNG = FIGURES / "Supplementary_Figure_S8_generated.png"
FONT = Path("C:/Windows/Fonts/arial.ttf")
WIDTH_PT, HEIGHT_PT, DPI = 225.2, 140.7, 220
FONT_PT, MARK_PT, AXIS_PT = 7.0, 0.7, 1.0
BLUE, CORAL, INK = "#70a0c3", "#d58168", "#202020"


def sy(value: float, top: float, bottom: float, ymax: float) -> float:
    return bottom - value / ymax * (bottom - top)


def svg_text(x: float, y: float, value: str, anchor: str = "middle", rotate: float | None = None) -> str:
    transform = "" if rotate is None else f' transform="rotate({rotate} {x:.2f} {y:.2f})"'
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}"{transform} '
        f'font-family="Arial" font-size="{FONT_PT}pt" fill="black">{html.escape(value)}</text>'
    )


def load_values() -> tuple[list[list[float]], list[float]]:
    host = pd.read_csv(TABLES / "host_chromosome_metrics_each_run.csv")
    temporal = pd.read_csv(TABLES / "temporal_external_metrics.csv")
    host_values = [
        host.loc[host["task"].eq("classification"), "AUPRC"].astype(float).tolist(),
        host.loc[host["task"].str.startswith("regression"), "R2"].astype(float).tolist(),
    ]
    temporal_values = [
        float(temporal.loc[temporal["task"].eq("classification"), "AUPRC"].item()),
        float(temporal.loc[temporal["task"].str.startswith("regression"), "R2"].item()),
    ]
    if any(len(values) != 3 for values in host_values):
        raise ValueError("Host-chromosome S8 panel requires three runs")
    return host_values, temporal_values


def panel_svg(parts: list[str], left: float, right: float, host: list[list[float]] | None, temporal: list[float] | None, ymax: float, caption: str) -> None:
    top, bottom = 3.5, 98.0
    parts += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{INK}" stroke-width="{AXIS_PT}"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{INK}" stroke-width="{AXIS_PT}"/>',
    ]
    for tick in np.arange(0, ymax + 0.001, 0.2):
        y = sy(float(tick), top, bottom, ymax)
        parts += [
            f'<line x1="{left - 2.5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="{INK}" stroke-width="{MARK_PT}"/>',
            svg_text(left - 4.5, y + 2.3, f"{tick:.1f}", "end"),
        ]
    parts.append(svg_text(left - 18, (top + bottom) / 2, "Performance", rotate=-90))
    centers = [left + (right-left) * 0.28, left + (right-left) * 0.72]
    values = host if host is not None else [[temporal[0]], [temporal[1]]]
    for x, runs, color, label in zip(centers, values, (BLUE, CORAL), ("AUPRC", "R²")):
        mean = float(np.mean(runs)); y = sy(mean, top, bottom, ymax)
        parts.append(f'<rect x="{x-8}" y="{y:.2f}" width="16" height="{bottom-y:.2f}" fill="{color}" stroke="{INK}" stroke-width="{MARK_PT}"/>')
        if len(runs) == 3:
            sd = float(np.std(runs, ddof=1)); yh=sy(min(ymax,mean+sd),top,bottom,ymax); yl=sy(max(0,mean-sd),top,bottom,ymax)
            parts += [f'<line x1="{x}" y1="{yh:.2f}" x2="{x}" y2="{yl:.2f}" stroke="{INK}" stroke-width="{MARK_PT}"/>', f'<line x1="{x-3}" y1="{yh:.2f}" x2="{x+3}" y2="{yh:.2f}" stroke="{INK}" stroke-width="{MARK_PT}"/>', f'<line x1="{x-3}" y1="{yl:.2f}" x2="{x+3}" y2="{yl:.2f}" stroke="{INK}" stroke-width="{MARK_PT}"/>']
        offsets = np.linspace(-2.3, 2.3, len(runs))
        for offset, value in zip(offsets, runs):
            parts.append(f'<circle cx="{x+offset:.2f}" cy="{sy(float(value),top,bottom,ymax):.2f}" r="1.35" fill="black"/>')
        parts.append(svg_text(x, bottom + 10, label))
    parts.append(svg_text((left+right)/2, 137, caption))


def write_svg(host: list[list[float]], temporal: list[float]) -> None:
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_PT}pt" height="{HEIGHT_PT}pt" viewBox="0 0 {WIDTH_PT} {HEIGHT_PT}">','<rect width="100%" height="100%" fill="white"/>']
    panel_svg(parts,23,95,host,None,0.8,"Host chromosome")
    panel_svg(parts,151,224,None,temporal,1.0,"External validation")
    parts.append('</svg>'); SVG.write_text('\n'.join(parts)+'\n',encoding='utf-8')


def write_png(host: list[list[float]], temporal: list[float]) -> None:
    scale=DPI/72; image=Image.new('RGB',(round(WIDTH_PT*scale),round(HEIGHT_PT*scale)),'white'); draw=ImageDraw.Draw(image); font=ImageFont.truetype(str(FONT),round(FONT_PT*scale)); q=lambda x:round(x*scale)
    def text(x,y,value,anchor='mm'): draw.text((q(x),q(y)),value,fill='black',font=font,anchor=anchor)
    def rotated(x,y,value):
        layer=Image.new('RGBA',(q(70),q(14)),(255,255,255,0)); ImageDraw.Draw(layer).text((q(35),q(7)),value,fill='black',font=font,anchor='mm'); layer=layer.rotate(90,expand=True,resample=Image.Resampling.BICUBIC); image.paste(layer,(q(x)-layer.width//2,q(y)-layer.height//2),layer)
    def panel(left,right,values,ymax,caption):
        top,bottom=3.5,98.0; draw.line((q(left),q(top),q(left),q(bottom)),fill=INK,width=q(AXIS_PT)); draw.line((q(left),q(bottom),q(right),q(bottom)),fill=INK,width=q(AXIS_PT))
        for tick in np.arange(0,ymax+.001,.2): y=sy(float(tick),top,bottom,ymax); draw.line((q(left-2.5),q(y),q(left),q(y)),fill=INK,width=q(MARK_PT)); text(left-4.5,y,f'{tick:.1f}','rm')
        rotated(left-18,(top+bottom)/2,'Performance'); centers=[left+(right-left)*.28,left+(right-left)*.72]
        for x,runs,color,label in zip(centers,values,(BLUE,CORAL),('AUPRC','R²')):
            mean=float(np.mean(runs)); y=sy(mean,top,bottom,ymax); draw.rectangle((q(x-8),q(y),q(x+8),q(bottom)),fill=color,outline=INK,width=q(MARK_PT))
            if len(runs)==3:
                sd=float(np.std(runs,ddof=1)); yh=sy(min(ymax,mean+sd),top,bottom,ymax); yl=sy(max(0,mean-sd),top,bottom,ymax); draw.line((q(x),q(yh),q(x),q(yl)),fill=INK,width=q(MARK_PT)); draw.line((q(x-3),q(yh),q(x+3),q(yh)),fill=INK,width=q(MARK_PT)); draw.line((q(x-3),q(yl),q(x+3),q(yl)),fill=INK,width=q(MARK_PT))
            for off,value in zip(np.linspace(-2.3,2.3,len(runs)),runs): yy=sy(float(value),top,bottom,ymax); rr=q(1.35); draw.ellipse((q(x+off)-rr,q(yy)-rr,q(x+off)+rr,q(yy)+rr),fill='black')
            text(x,bottom+10,label)
        text((left+right)/2,137,caption)
    panel(23,95,host,.8,'Host chromosome'); panel(151,224,[[temporal[0]],[temporal[1]]],1.0,'External validation'); image.save(PNG,dpi=(DPI,DPI))


def main() -> int:
    if not FONT.is_file(): raise FileNotFoundError("Arial is required")
    host, temporal=load_values(); FIGURES.mkdir(parents=True,exist_ok=True); write_svg(host,temporal); write_png(host,temporal); return 0


if __name__ == "__main__": raise SystemExit(main())
