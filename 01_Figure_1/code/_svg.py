"""Small dependency-free SVG primitives for publication panels."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np


FONT = "Arial"
FONT_SIZE = 7
AXIS_W = 1
MARK_W = 0.7


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x, y, value, anchor="middle", rotate=None, weight=None, size=FONT_SIZE):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    bold = f' font-weight="{weight}"' if weight else ""
    lines = str(value).split("\n")
    if len(lines) == 1:
        return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}"{transform}{bold}>{esc(value)}</text>'
    line_height = float(size) * 1.15
    first_y = y - line_height * (len(lines) - 1) / 2
    tspans = [f'<tspan x="{x:.2f}" y="{first_y:.2f}">{esc(lines[0])}</tspan>']
    tspans.extend(f'<tspan x="{x:.2f}" dy="{line_height:.2f}">{esc(item)}</tspan>' for item in lines[1:])
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}"{transform}{bold}>{"".join(tspans)}</text>'


def line(x1, y1, x2, y2, width=AXIS_W, color="#111", dash=None):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{dashed}/>'


def rect(x, y, width, height, fill, stroke="#111", line_width=MARK_W, opacity=1):
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{line_width}" opacity="{opacity}"/>'


def polygon(points, fill, stroke="#111", line_width=MARK_W, opacity=0.75):
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{coordinates}" fill="{fill}" stroke="{stroke}" stroke-width="{line_width}" opacity="{opacity}"/>'


def write(path: Path, width: float, height: float, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>', *body, '</svg>']
    path.write_text("\n".join(doc), encoding="utf-8")


def _ticks(maximum: float, count: int = 5) -> list[float]:
    maximum = max(float(maximum), 1e-12)
    raw = maximum / count
    magnitude = 10 ** np.floor(np.log10(raw))
    step = min([1, 2, 5, 10], key=lambda value: abs(value * magnitude - raw)) * magnitude
    top = np.ceil(maximum / step) * step
    return [float(value) for value in np.arange(0, top + step / 2, step)]


def _axes(body, x0, y0, plot_w, plot_h, ymax, ylabel, ticks=None):
    if ticks is None:
        ticks = _ticks(ymax)
        top = ticks[-1]
    else:
        ticks = [float(value) for value in ticks]
        top = float(ymax)
    if top <= 0 or max(ticks) > top:
        raise ValueError("Axis maximum must be positive and include every tick")
    body += [line(x0, y0, x0 + plot_w, y0), line(x0, y0, x0, y0 - plot_h)]
    for tick in ticks:
        y = y0 - plot_h * tick / top
        body += [line(x0 - 3, y, x0, y), text(x0 - 6, y + 2.5, f"{tick:g}", anchor="end")]
    body.append(text(x0 - 38, y0 - plot_h / 2, ylabel, rotate=-90))
    return top


def bracket(body, x1, x2, y, label):
    body += [line(x1, y + 4, x1, y, MARK_W), line(x1, y, x2, y, MARK_W), line(x2, y, x2, y + 4, MARK_W),
             text((x1 + x2) / 2, y - 2, label, weight="bold")]


def bar_chart(path, labels, values, colors, ylabel, comparisons=(), width=190, height=190,
              y_ticks=None, y_max=None, category_span=1.0):
    body=[]; left=72; right=10; top=28; bottom=52; x0=left; y0=height-bottom; pw=width-left-right; ph=height-top-bottom
    ymax=float(y_max) if y_max is not None else max(values) * (1.28 + 0.12 * len(comparisons))
    scale=_axes(body,x0,y0,pw,ph,ymax,ylabel,y_ticks)
    gap=pw/len(values); bar_w=min(25,gap*0.56); centers=[]
    center=(len(values)-1)/2
    for i,(label,value,color) in enumerate(zip(labels,values,colors)):
        cx=x0+pw/2+(i-center)*gap*category_span; centers.append(cx); h=ph*value/scale
        body += [rect(cx-bar_w/2,y0-h,bar_w,h,color), text(cx,y0+13,label)]
    for level,(first,second,label) in enumerate(comparisons):
        bracket(body,centers[first],centers[second],top+level*10,label)
    write(path,width,height,body)


def grouped_bar(path, group_labels, series_labels, matrix, colors, ylabel, comparisons=(), width=215, height=190,
                y_ticks=None, y_max=None, series_comparisons=(), group_span=1.0, xlabel=None):
    body=[]; left=72; right=10; top=30; bottom=60; x0=left; y0=height-bottom; pw=width-left-right; ph=height-top-bottom
    ymax=float(y_max) if y_max is not None else float(np.nanmax(matrix))*1.28
    scale=_axes(body,x0,y0,pw,ph,ymax,ylabel,y_ticks)
    group_w=pw/len(group_labels); bar_w=min(18,group_w/(len(series_labels)+1)); centers=[]; series_centers=[]
    group_center=(len(group_labels)-1)/2
    for gi,group in enumerate(group_labels):
        cx=x0+pw/2+(gi-group_center)*group_w*group_span; centers.append(cx); current=[]
        for si,(label,color) in enumerate(zip(series_labels,colors)):
            value=float(matrix[gi][si]); bx=cx+(si-(len(series_labels)-1)/2)*(bar_w+2); h=ph*value/scale
            current.append(bx); body.append(rect(bx-bar_w/2,y0-h,bar_w,h,color))
        series_centers.append(current)
        body.append(text(cx,y0+13,group))
    for level,(group_index,label) in enumerate(comparisons):
        span=(bar_w+2)*(len(series_labels)-1)
        bracket(body,centers[group_index]-span/2,centers[group_index]+span/2,top+level*9,label)
    levels={}
    for group_index,first,second,label in series_comparisons:
        level=levels.get(group_index,0); levels[group_index]=level+1
        bracket(body,series_centers[group_index][first],series_centers[group_index][second],top+level*9,label)
    if xlabel:
        body.append(text(x0+pw/2,y0+30,xlabel))
    lx=x0; ly=height-8-(len(series_labels)-1)*9
    for i,(label,color) in enumerate(zip(series_labels,colors)):
        row_y=ly+i*9
        body += [rect(lx,row_y-6,7,7,color),text(lx+10,row_y,label,anchor="start")]
    write(path,width,height,body)


def box_stats(values):
    values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]
    q1,median,q3=np.percentile(values,[25,50,75]); iqr=q3-q1
    low=values[values>=q1-1.5*iqr].min(); high=values[values<=q3+1.5*iqr].max()
    return low,q1,median,q3,high


def box_chart(path, labels, arrays, colors, ylabel, comparisons=(), width=190, height=190,
              y_ticks=None, y_min=None, y_max=None, box_width=None):
    stats=[box_stats(values) for values in arrays]; ymax=max(s[-1] for s in stats); ymin=min(s[0] for s in stats)
    extra=(ymax-ymin)*(.22+.08*len(comparisons)); body=[]; left=72; right=10; top=28; bottom=52
    x0=left; y0=height-bottom; pw=width-left-right; ph=height-top-bottom
    scale_max=float(y_max) if y_max is not None else ymax+extra
    scale_min=float(y_min) if y_min is not None else min(0,ymin)
    ticks=np.linspace(scale_min,scale_max,5) if y_ticks is None else [float(value) for value in y_ticks]
    if scale_max <= scale_min or min(ticks) < scale_min or max(ticks) > scale_max:
        raise ValueError("Invalid box-chart axis limits")
    body += [line(x0,y0,x0+pw,y0),line(x0,y0,x0,y0-ph)]
    for tick in ticks:
        y=y0-ph*(tick-scale_min)/(scale_max-scale_min); body += [line(x0-3,y,x0,y),text(x0-6,y+2.5,f"{tick:g}",anchor="end")]
    body.append(text(30,y0-ph/2,ylabel,rotate=-90)); gap=pw/len(arrays); centers=[]
    bw=min(float(box_width),gap*.72) if box_width is not None else min(28,gap*.52)
    def sy(value): return y0-ph*(value-scale_min)/(scale_max-scale_min)
    for i,(label,stat,color) in enumerate(zip(labels,stats,colors)):
        cx=x0+gap*(i+.5); centers.append(cx); low,q1,median,q3,high=stat
        body += [line(cx,sy(low),cx,sy(q1)),line(cx,sy(q3),cx,sy(high)),line(cx-bw*.3,sy(low),cx+bw*.3,sy(low)),
                 line(cx-bw*.3,sy(high),cx+bw*.3,sy(high)),rect(cx-bw/2,sy(q3),bw,sy(q1)-sy(q3),color,opacity=.75),
                 line(cx-bw/2,sy(median),cx+bw/2,sy(median)),text(cx,y0+13,label)]
    for level,(first,second,label) in enumerate(comparisons): bracket(body,centers[first],centers[second],top+level*10,label)
    write(path,width,height,body)


def violin_chart(path, labels, arrays, colors, ylabel, comparisons=(), width=190, height=190,
                 y_ticks=None, y_min=0, y_max=None, violin_width=40, bins=80):
    values = [np.asarray(array, dtype=float) for array in arrays]
    if y_max is None:
        y_max = max(float(np.nanmax(array)) for array in values)
    scale_min = float(y_min); scale_max = float(y_max)
    if scale_max <= scale_min:
        raise ValueError("Invalid violin-chart axis limits")
    ticks = [float(value) for value in (y_ticks if y_ticks is not None else _ticks(scale_max))]
    if min(ticks) < scale_min or max(ticks) > scale_max:
        raise ValueError("Invalid violin-chart ticks")
    body=[]; left=72; right=10; top=28; bottom=52
    x0=left; y0=height-bottom; pw=width-left-right; ph=height-top-bottom
    body += [line(x0,y0,x0+pw,y0),line(x0,y0,x0,y0-ph)]
    for tick in ticks:
        y=y0-ph*(tick-scale_min)/(scale_max-scale_min)
        body += [line(x0-3,y,x0,y),text(x0-6,y+2.5,f"{tick:g}",anchor="end")]
    body.append(text(30,y0-ph/2,ylabel,rotate=-90))
    gap=pw/len(values)
    def sy(value): return y0-ph*(value-scale_min)/(scale_max-scale_min)
    edges=np.linspace(scale_min,scale_max,bins+1)
    for i,(label,array,color) in enumerate(zip(labels,values,colors)):
        finite=array[np.isfinite(array)]
        clipped=finite[(finite >= scale_min) & (finite <= scale_max)]
        counts, _ = np.histogram(clipped, bins=edges)
        centers_y=(edges[:-1]+edges[1:])/2
        density=counts.astype(float)
        if density.max() == 0:
            raise ValueError("Violin chart has no values within axis limits")
        half_width=violin_width/2* density/density.max()
        cx=x0+gap*(i+.5)
        points=[(cx,sy(scale_min))]
        points.extend((cx+float(w),sy(float(y))) for w,y in zip(half_width,centers_y))
        points.append((cx,sy(scale_max)))
        points.extend((cx-float(w),sy(float(y))) for w,y in zip(half_width[::-1],centers_y[::-1]))
        points.append((cx,sy(scale_min)))
        body.append(polygon(points,color))
        median=float(np.percentile(finite,50))
        if scale_min <= median <= scale_max:
            body.append(line(cx-violin_width*.32,sy(median),cx+violin_width*.32,sy(median),MARK_W))
        body.append(text(cx,y0+13,label))
    centers=[x0+gap*(i+.5) for i in range(len(values))]
    for level,(first,second,label) in enumerate(comparisons):
        bracket(body,centers[first],centers[second],top+level*10,label)
    write(path,width,height,body)


def grouped_box_chart(path, group_labels, series_labels, arrays, colors, ylabel, comparisons=(),
                      width=230, height=190, y_ticks=None, y_min=0, y_max=None):
    expected=len(group_labels)*len(series_labels)
    if len(arrays) != expected or len(colors) != expected:
        raise ValueError("Grouped box chart requires one array and color per group-series combination")
    stats=[box_stats(values) for values in arrays]
    data_max=max(item[-1] for item in stats)
    scale_min=float(y_min)
    scale_max=float(y_max) if y_max is not None else data_max*1.25
    ticks=_ticks(scale_max) if y_ticks is None else [float(value) for value in y_ticks]
    if scale_max <= scale_min or min(ticks) < scale_min or max(ticks) > scale_max:
        raise ValueError("Invalid grouped-box axis limits")
    body=[]; left=72; right=10; top=30; bottom=60
    x0=left; y0=height-bottom; pw=width-left-right; ph=height-top-bottom
    body += [line(x0,y0,x0+pw,y0),line(x0,y0,x0,y0-ph)]
    for tick in ticks:
        y=y0-ph*(tick-scale_min)/(scale_max-scale_min)
        body += [line(x0-3,y,x0,y),text(x0-6,y+2.5,f"{tick:g}",anchor="end")]
    body.append(text(30,y0-ph/2,ylabel,rotate=-90))
    group_w=pw/len(group_labels); bw=min(18,group_w/(len(series_labels)+1.2)); centers=[]
    def sy(value): return y0-ph*(value-scale_min)/(scale_max-scale_min)
    for gi,group in enumerate(group_labels):
        group_center=x0+group_w*(gi+.5); current=[]
        for si in range(len(series_labels)):
            index=gi*len(series_labels)+si
            cx=group_center+(si-(len(series_labels)-1)/2)*(bw+3); current.append(cx)
            low,q1,median,q3,high=stats[index]
            body += [line(cx,sy(low),cx,sy(q1)),line(cx,sy(q3),cx,sy(high)),
                     line(cx-bw*.3,sy(low),cx+bw*.3,sy(low)),line(cx-bw*.3,sy(high),cx+bw*.3,sy(high)),
                     rect(cx-bw/2,sy(q3),bw,sy(q1)-sy(q3),colors[index],opacity=.75),
                     line(cx-bw/2,sy(median),cx+bw/2,sy(median))]
        centers.append(current); body.append(text(group_center,y0+13,group))
    levels={}
    for group_index,first,second,label in comparisons:
        level=levels.get(group_index,0); levels[group_index]=level+1
        bracket(body,centers[group_index][first],centers[group_index][second],top+level*9,label)
    lx=x0; ly=height-8-(len(series_labels)-1)*9
    for i,(label,color) in enumerate(zip(series_labels,colors[:len(series_labels)])):
        row_y=ly+i*9
        body += [rect(lx,row_y-6,7,7,color),text(lx+10,row_y,label,anchor="start")]
    write(path,width,height,body)
