#!/usr/bin/env python
"""Generate Figure 1B from the exact 61,961-plasmid tRNA table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _svg import _ticks, line, rect, text, write


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plasmid-table", required=True, type=Path)
    parser.add_argument("--table-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    args = parser.parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True); args.figure_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.plasmid_table)
    if len(data) != 61_961 or data["Replicon_Acc"].nunique() != 61_961:
        raise ValueError("Expected 61,961 unique plasmids")
    data["log10_size"] = np.log10(data["Length"])
    data["has_tRNA"] = data["Total_tRNA"].gt(0)
    edges = np.linspace(3, 7, 30)
    data["plot_eligible"] = data["log10_size"].between(edges[0], edges[-1], inclusive="both")
    data["size_bin"] = pd.cut(data["log10_size"], edges, include_lowest=True)
    data[["Replicon_Acc", "Length", "Total_tRNA", "log10_size", "has_tRNA", "plot_eligible", "size_bin"]].to_csv(
        args.table_dir / "Fig1B_plasmid_level_raw.csv", index=False
    )
    plotted = data[data["plot_eligible"]].copy()
    if len(plotted) != 61_874:
        raise ValueError(f"Expected 61,874 plasmids in the fixed 1-kb to 10-Mb display range, found {len(plotted)}")
    summary = plotted.groupby("size_bin", observed=False).agg(
        plasmid_n=("Replicon_Acc", "size"), tRNA_positive_n=("has_tRNA", "sum"),
        probability=("has_tRNA", "mean"),
    ).reset_index(drop=True)
    summary["bin_left"] = edges[:-1]
    summary["bin_right"] = edges[1:]
    summary["bin_midpoint"] = (summary["bin_left"] + summary["bin_right"]) / 2
    summary["probability_se"] = np.sqrt(
        summary["probability"] * (1 - summary["probability"]) / summary["plasmid_n"]
    )
    summary.to_csv(args.table_dir / "Fig1B_size_trna_probability.csv", index=False)
    width, height = 368, 252; left, right, top, bottom = 60, 58, 20, 46
    x0, y0 = left, height-bottom; pw, ph = width-left-right, height-top-bottom
    xmin, xmax = 2.85, 7.2
    probability_min, probability_max = -0.05, 1.05
    count_ticks = list(range(0, 7001, 1000))
    count_max = 7600
    sx=lambda value: x0+pw*(value-xmin)/(xmax-xmin)
    sy=lambda value: y0-ph*(value-probability_min)/(probability_max-probability_min)
    sy_count=lambda value: y0-ph*value/count_max
    probability_color="#E85C54"; count_color="#7FABD1"
    body=[line(x0,y0,x0+pw,y0),line(x0,y0,x0,y0-ph,color=probability_color),
          line(x0+pw,y0,x0+pw,y0-ph,color=count_color)]
    for tick in range(3,8):
        body += [line(sx(tick),y0,sx(tick),y0+3),text(sx(tick),y0+13,f"10^{tick}")]
    for tick in np.linspace(0,1,6):
        body += [line(x0-3,sy(tick),x0,sy(tick),color=probability_color),
                 text(x0-6,sy(tick)+2.5,f"{tick:.1f}",anchor="end")]
    for tick in count_ticks:
        y = sy_count(tick)
        body += [line(x0+pw,y,x0+pw+3,y,color=count_color),text(x0+pw+6,y+2.5,f"{tick:d}",anchor="start")]
    bar_width=(sx(edges[1])-sx(edges[0]))*.95
    for row in summary.itertuples(index=False):
        h=ph*row.plasmid_n/count_max
        body.append(rect(sx(row.bin_midpoint)-bar_width/2,y0-h,bar_width,h,count_color,stroke="white",line_width=.7,opacity=.32))
    points=[]
    for row in summary.itertuples(index=False):
        if not np.isfinite(row.probability):
            continue
        x,y=sx(row.bin_midpoint),sy(row.probability); points.append((x,y))
        lower=max(0,row.probability-row.probability_se); upper=min(1,row.probability+row.probability_se)
        body += [line(x,sy(lower),x,sy(upper),.7,probability_color),
                 line(x-2,sy(lower),x+2,sy(lower),.7,probability_color),
                 line(x-2,sy(upper),x+2,sy(upper),.7,probability_color),
                 f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{probability_color}"/>']
    for first,second in zip(points,points[1:]): body.append(line(first[0],first[1],second[0],second[1],1,probability_color))
    threshold=sx(np.log10(100_000)); body += [line(threshold,y0,threshold,y0-ph,.7,"#7F8C8D","4 3"),
        text(threshold+6,top+25,"Megaplasmid\n(>=100 kb)",anchor="start"),text(x0+pw/2,height-9,"Plasmid size (bp, log10 scale)"),
        text(20,y0-ph/2,"Probability of carrying tRNA",rotate=-90),text(width-20,y0-ph/2,"Number of plasmids",rotate=90)]
    write(args.figure_dir / "Fig1B_size_trna_probability.svg",width,height,body)
    return 0


if __name__ == "__main__": raise SystemExit(main())
