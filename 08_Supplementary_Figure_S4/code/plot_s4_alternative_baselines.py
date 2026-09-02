"""Plot S4 k-nearest-neighbor and host-chromosome baselines."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = PACKAGE_DIR / "tables"
FIGURE_DIR = ROOT / "GitHub_upload_V5_figures_20260831_figures_only" / PACKAGE_DIR.name
SVG_PATH = FIGURE_DIR / "Supplementary_Figure_S4_source.svg"
PNG_PATH = FIGURE_DIR / "Supplementary_Figure_S4_generated.png"
FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
BOLD_PATH = Path("C:/Windows/Fonts/arialbd.ttf")
WIDTH_PT, HEIGHT_PT = 319.0, 123.0
DPI = 220
FONT_PT, MARK_PT, AXIS_PT = 7.0, 0.7, 1.0
BLUE, CORAL, INK = "#7da8c7", "#d98779", "#202020"


def y_value(value: float, top: float, bottom: float, ymin: float, ymax: float) -> float:
    return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)


def text_svg(x: float, y: float, value: str, anchor: str = "middle", rotate: float | None = None, bold: bool = False) -> str:
    transform = "" if rotate is None else f' transform="rotate({rotate} {x:.2f} {y:.2f})"'
    weight = "bold" if bold else "normal"
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}"{transform} font-family="Arial" font-size="{FONT_PT}pt" font-weight="{weight}" fill="black">{html.escape(value)}</text>'


def panel_svg(parts, left, top, right, bottom, values, labels, colors, ylabel, limits, ticks, panel):
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{INK}" stroke-width="{AXIS_PT}"/>', f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{INK}" stroke-width="{AXIS_PT}"/>']
    for tick in ticks:
        y = y_value(tick, top, bottom, *limits)
        parts += [f'<line x1="{left-2.5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="{INK}" stroke-width="{MARK_PT}"/>', text_svg(left-4.5, y+2.3, f"{tick:.1f}", "end")]
    parts += [text_svg(left-19, (top+bottom)/2, ylabel, rotate=-90), text_svg(left-35, top-5, panel, "start", bold=True)]
    group = (right-left)/len(values); bar_width = group*0.62
    for index, runs in enumerate(values):
        mean=float(np.mean(runs)); sd=float(np.std(runs,ddof=1)); x=left+group*(index+.5)
        y0=y_value(max(mean,limits[0]),top,bottom,*limits); yz=y_value(max(0.0,limits[0]),top,bottom,*limits)
        parts.append(f'<rect x="{x-bar_width/2:.2f}" y="{min(y0,yz):.2f}" width="{bar_width:.2f}" height="{abs(yz-y0):.2f}" fill="{colors[index]}" stroke="{INK}" stroke-width="{MARK_PT}"/>')
        yh=y_value(min(limits[1],mean+sd),top,bottom,*limits); yl=y_value(max(limits[0],mean-sd),top,bottom,*limits)
        parts += [f'<line x1="{x:.2f}" y1="{yh:.2f}" x2="{x:.2f}" y2="{yl:.2f}" stroke="{INK}" stroke-width="{MARK_PT}"/>', f'<line x1="{x-3:.2f}" y1="{yh:.2f}" x2="{x+3:.2f}" y2="{yh:.2f}" stroke="{INK}" stroke-width="{MARK_PT}"/>', f'<line x1="{x-3:.2f}" y1="{yl:.2f}" x2="{x+3:.2f}" y2="{yl:.2f}" stroke="{INK}" stroke-width="{MARK_PT}"/>']
        for offset, run in zip((-3,0,3),runs):
            y=y_value(float(run),top,bottom,*limits); parts.append(f'<circle cx="{x+offset:.2f}" cy="{y:.2f}" r="1.35" fill="black"/>')
        parts.append(text_svg(x,bottom+10,labels[index]))


def write_svg(cls, reg, host):
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_PT}pt" height="{HEIGHT_PT}pt" viewBox="0 0 {WIDTH_PT} {HEIGHT_PT}">','<rect width="100%" height="100%" fill="white"/>']
    panel_svg(parts,37,14,98,108,cls,["1-NN","5-NN"],[BLUE,BLUE],"AUPRC",(0,1.02),[0,.2,.4,.6,.8,1.0],"A")
    panel_svg(parts,143,14,204,108,reg,["1-NN","5-NN"],[CORAL,CORAL],"R²",(0,.5),[0,.1,.2,.3,.4,.5],"B")
    panel_svg(parts,249,14,315,108,host,["AUPRC","R²"],[BLUE,CORAL],"Performance",(-.1,.8),[0,.2,.4,.6,.8],"C")
    parts.append('</svg>'); SVG_PATH.write_text('\n'.join(parts)+'\n',encoding='utf-8')


def write_png(cls, reg, host):
    scale=DPI/72; image=Image.new('RGB',(round(WIDTH_PT*scale),round(HEIGHT_PT*scale)),'white'); draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(str(FONT_PATH),round(FONT_PT*scale)); bold=ImageFont.truetype(str(BOLD_PATH),round(FONT_PT*scale)); q=lambda a:round(a*scale)
    def draw_text(x,y,value,anchor='mm',use_bold=False): draw.text((q(x),q(y)),value,fill='black',font=bold if use_bold else font,anchor=anchor)
    def rotated(x,y,value):
        layer=Image.new('RGBA',(q(80),q(14)),(255,255,255,0)); ImageDraw.Draw(layer).text((q(40),q(7)),value,fill='black',font=font,anchor='mm'); layer=layer.rotate(90,expand=True,resample=Image.Resampling.BICUBIC); image.paste(layer,(q(x)-layer.width//2,q(y)-layer.height//2),layer)
    def panel(left,top,right,bottom,values,labels,colors,ylabel,limits,ticks,letter):
        draw.line((q(left),q(top),q(left),q(bottom)),fill=INK,width=q(AXIS_PT)); draw.line((q(left),q(bottom),q(right),q(bottom)),fill=INK,width=q(AXIS_PT))
        for tick in ticks:
            y=y_value(tick,top,bottom,*limits); draw.line((q(left-2.5),q(y),q(left),q(y)),fill=INK,width=q(MARK_PT)); draw_text(left-4.5,y,f"{tick:.1f}",'rm')
        rotated(left-19,(top+bottom)/2,ylabel); draw_text(left-35,top-5,letter,'la',True); group=(right-left)/len(values); bw=group*.62
        for i,runs in enumerate(values):
            mean=float(np.mean(runs)); sd=float(np.std(runs,ddof=1)); x=left+group*(i+.5); y0=y_value(max(mean,limits[0]),top,bottom,*limits); yz=y_value(max(0.0,limits[0]),top,bottom,*limits)
            draw.rectangle((q(x-bw/2),q(min(y0,yz)),q(x+bw/2),q(max(y0,yz))),fill=colors[i],outline=INK,width=q(MARK_PT)); yh=y_value(min(limits[1],mean+sd),top,bottom,*limits); yl=y_value(max(limits[0],mean-sd),top,bottom,*limits)
            draw.line((q(x),q(yh),q(x),q(yl)),fill=INK,width=q(MARK_PT)); draw.line((q(x-3),q(yh),q(x+3),q(yh)),fill=INK,width=q(MARK_PT)); draw.line((q(x-3),q(yl),q(x+3),q(yl)),fill=INK,width=q(MARK_PT))
            for off,run in zip((-3,0,3),runs):
                y=y_value(float(run),top,bottom,*limits); rr=q(1.35); draw.ellipse((q(x+off)-rr,q(y)-rr,q(x+off)+rr,q(y)+rr),fill='black')
            draw_text(x,bottom+10,labels[i])
    panel(37,14,98,108,cls,["1-NN","5-NN"],[BLUE,BLUE],"AUPRC",(0,1.02),[0,.2,.4,.6,.8,1.0],"A")
    panel(143,14,204,108,reg,["1-NN","5-NN"],[CORAL,CORAL],"R²",(0,.5),[0,.1,.2,.3,.4,.5],"B")
    panel(249,14,315,108,host,["AUPRC","R²"],[BLUE,CORAL],"Performance",(-.1,.8),[0,.2,.4,.6,.8],"C")
    image.save(PNG_PATH,dpi=(DPI,DPI))


def main():
    if not FONT_PATH.is_file() or not BOLD_PATH.is_file(): raise FileNotFoundError('Arial regular and bold fonts are required')
    knn=pd.read_csv(TABLE_DIR/'knn_metrics_each_run.csv'); host_data=pd.read_csv(TABLE_DIR/'host_chromosome_metrics_each_run.csv')
    cls=[knn.loc[knn.task.eq('classification')&knn.model.eq(m),'AUPRC'].astype(float).tolist() for m in ('1-NN','5-NN')]
    reg=[knn.loc[knn.task.str.startswith('regression')&knn.model.eq(m),'R2'].astype(float).tolist() for m in ('1-NN','5-NN')]
    host=[host_data.loc[host_data.task.eq('classification'),'AUPRC'].astype(float).tolist(),host_data.loc[host_data.task.str.startswith('regression'),'R2'].astype(float).tolist()]
    if any(len(v)!=3 for v in cls+reg+host): raise ValueError('S4 requires three runs for every metric')
    FIGURE_DIR.mkdir(parents=True,exist_ok=True); write_svg(cls,reg,host); write_png(cls,reg,host); return 0


if __name__=='__main__': raise SystemExit(main())
