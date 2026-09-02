"""Plot the 14-ring genus-level central-dogma landscape for S15."""

from __future__ import annotations

import html
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
INPUT = PACKAGE / "tables" / "Plasmid_CDE_module_subcategory_landscape_per_plasmid.csv"
PLOT_DATA = PACKAGE / "tables" / "Supplementary_Figure_S15_genus_prevalence.csv"
FIGURE_DIR = ROOT / "GitHub_upload_V5_figures_20260831_figures_only" / PACKAGE.name
PNG = FIGURE_DIR / "Supplementary_Figure_S15_composite.png"
SVG = FIGURE_DIR / "Supplementary_Figure_S15_source.svg"

FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
WIDTH, HEIGHT, DPI, SCALE = 1190, 875, 220, 2
FONT_PT, MARK_PT = 7.0, 0.7
CX, CY = 435.0, 437.0
TAXONOMY = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]
RINGS = [
    ("Replication initiator", "replication_Replication_Initiator_count", "replication"),
    ("Maintenance / partition", "replication_Maintenance_Partition_count", "replication"),
    ("DNA repair and recombination", "replication_DNA_repair_and_recombination_count", "replication"),
    ("Machinery / elongation", "replication_Machinery_Elongation_count", "replication"),
    ("Transcription regulation", "transcription_Transcription_regulation_count", "transcription"),
    ("Initiation and promoter recognition", "transcription_Initiation_and_promoter_recognition_count", "transcription"),
    ("Termination and anti-termination", "transcription_Termination_and_anti_termination_count", "transcription"),
    ("Core transcription machinery", "transcription_Core_transcription_machinery_count", "transcription"),
    ("tRNA", "translation_tRNA_count", "translation"),
    ("Regulation and control", "translation_Regulation_and_Control_count", "translation"),
    ("Aminoacyl-tRNA synthetases", "translation_aaRS_count", "translation"),
    ("Ribosomal RNA", "translation_Ribosomal_RNA_count", "translation"),
    ("Ribosomal proteins", "translation_Ribosomal_proteins_count", "translation"),
    ("Ribosome assembly and maturation", "translation_Ribosome_assembly_and_maturation_count", "translation"),
]
MODULE_COLORS = {"replication": "#8dd3c8", "transcription": "#70a4d5", "translation": "#e26261"}
PHYLUM_COLORS = {
    "Actinomycetota": "#d73080", "Bacillota": "#756bb1", "Bacteroidota": "#4d9221",
    "Cyanobacteriota": "#d99a0b", "Methanobacteriota": "#006d4f", "Other": "#7a7a7a",
    "Pseudomonadota": "#1b9e77", "Spirochaetota": "#a6761d",
}


def hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def blend_white(high: str, value: float) -> str:
    fraction = min(max(float(value) / 100.0, 0.0), 1.0)
    rgb = hex_rgb(high)
    mixed = tuple(round(255 + fraction * (channel - 255)) for channel in rgb)
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def load_genus() -> pd.DataFrame:
    data = pd.read_csv(INPUT, keep_default_na=False)
    required = set(TAXONOMY + [column for _, column, _ in RINGS])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing S15 columns: {sorted(missing)}")
    for column in TAXONOMY:
        data[column] = data[column].fillna("").astype(str).str.strip().replace("", "Unknown")
    for _, column, _ in RINGS:
        data[column] = pd.to_numeric(data[column], errors="raise").ge(1).astype(float)
    named = {"Total_Plasmids": ("Genus", "size")}
    for _, column, _ in RINGS:
        named[column.replace("_count", "_prevalence_pct")] = (column, "mean")
    genus = data.groupby(TAXONOMY, as_index=False).agg(**named)
    genus = genus.loc[genus["Total_Plasmids"].gt(10) & ~genus["Genus"].isin(["", "Unknown", "uncultured"])].copy()
    prevalence = [column for column in genus.columns if column.endswith("_prevalence_pct")]
    genus[prevalence] *= 100.0
    genus = genus.sort_values(TAXONOMY, kind="stable").reset_index(drop=True)
    if genus["Genus"].duplicated().any():
        raise ValueError("S15 requires unique genus labels across taxonomy paths")
    if len(genus) < 2:
        raise ValueError("Too few genera remain after the >10-plasmid filter")
    return genus


def xy(radius: float, angle: float) -> tuple[float, float]:
    return CX + radius * math.cos(angle), CY + radius * math.sin(angle)


def tree_nodes(genus: pd.DataFrame, angles: list[float]):
    descendants = defaultdict(list)
    for index, row in genus.iterrows():
        path = tuple(str(row[column]) for column in TAXONOMY)
        for depth in range(len(path) + 1):
            descendants[path[:depth]].append(index)
    node_angles = {node: sum(angles[i] for i in indexes) / len(indexes) for node, indexes in descendants.items()}
    return descendants, node_angles


def polygon(radius_inner: float, radius_outer: float, angle: float, half_step: float) -> list[tuple[float, float]]:
    return [xy(radius_inner, angle-half_step), xy(radius_outer, angle-half_step), xy(radius_outer, angle+half_step), xy(radius_inner, angle+half_step)]


def svg_poly(points, fill, stroke="white") -> str:
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" stroke-width="{MARK_PT}"/>'


def svg_text(x, y, value, anchor="start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-family="Arial" font-size="{FONT_PT}pt" fill="black">{html.escape(str(value))}</text>'


def build_svg(genus: pd.DataFrame, angles: list[float], step: float) -> None:
    descendants, node_angles = tree_nodes(genus, angles)
    radii = [35, 65, 100, 135, 170, 200, 208]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">', '<rect width="100%" height="100%" fill="white"/>']
    for node in sorted(descendants, key=lambda item: (len(item), item)):
        depth = len(node)
        if depth >= len(TAXONOMY):
            continue
        children = [candidate for candidate in descendants if len(candidate) == depth + 1 and candidate[:-1] == node]
        children.sort(key=lambda item: min(descendants[item]))
        if not children:
            continue
        radius = radii[depth]
        child_angles = [node_angles[child] for child in children]
        if len(child_angles) > 1:
            arc = [xy(radius, child_angles[0] + (child_angles[-1]-child_angles[0])*index/40) for index in range(41)]
            parts.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in arc)}" fill="none" stroke="#7a7a7a" stroke-width="{MARK_PT}"/>')
        for child in children:
            p1=xy(radius,node_angles[child]); p2=xy(radii[len(child)],node_angles[child])
            parts.append(f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" stroke="#7a7a7a" stroke-width="{MARK_PT}"/>')
    display = genus["Phylum"].where(genus["Phylum"].isin(PHYLUM_COLORS), "Other")
    for angle, phylum in zip(angles, display):
        x,y=xy(214,angle); parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.3" fill="{PHYLUM_COLORS[str(phylum)]}"/>')
    inner, ring_width, gap = 222.0, 13.0, 10.0
    for index, (_, count_column, module) in enumerate(RINGS):
        if index in (4,8): inner += gap
        outer=inner+ring_width; column=count_column.replace("_count","_prevalence_pct")
        for angle,value in zip(angles,genus[column]): parts.append(svg_poly(polygon(inner,outer,angle,step*.49),blend_white(MODULE_COLORS[module],value)))
        inner=outer
    add_svg_legends(parts, genus)
    parts.append('</svg>'); SVG.write_text('\n'.join(parts)+'\n',encoding='utf-8')


def add_svg_legends(parts: list[str], genus: pd.DataFrame) -> None:
    x0=888; parts.append(svg_text(x0,237,'Phylum'))
    present=set(genus["Phylum"].where(genus["Phylum"].isin(PHYLUM_COLORS),"Other")); order=[name for name in PHYLUM_COLORS if name in present]
    for index,name in enumerate(order):
        y=262+index*24; parts += [f'<circle cx="{x0+9}" cy="{y}" r="6" fill="{PHYLUM_COLORS[name]}"/>',svg_text(x0+20,y+3,name)]
    base=262+len(order)*24+30
    titles={"replication":"Replication subcategories (%)","transcription":"Transcription subcategories (%)","translation":"Translation subcategories (%)"}
    for idx,module in enumerate(("replication","transcription","translation")):
        y=base+idx*72; parts.append(svg_text(x0,y,titles[module]))
        for i in range(101): parts.append(f'<rect x="{x0+i*.82:.2f}" y="{y+14}" width=".85" height="14" fill="{blend_white(MODULE_COLORS[module],i)}" stroke="none"/>')
        parts.append(f'<rect x="{x0}" y="{y+14}" width="82" height="14" fill="none" stroke="#777" stroke-width="{MARK_PT}"/>')
        parts += [svg_text(x0,y+43,'0','middle'),svg_text(x0+41,y+43,'50','middle'),svg_text(x0+82,y+43,'100','middle')]


def build_png(genus: pd.DataFrame, angles: list[float], step: float) -> None:
    image=Image.new('RGB',(WIDTH*SCALE,HEIGHT*SCALE),'white'); draw=ImageDraw.Draw(image); q=lambda a:round(a*SCALE)
    descendants,node_angles=tree_nodes(genus,angles); radii=[35,65,100,135,170,200,208]; line_width=max(1,q(MARK_PT*DPI/72))
    for node in sorted(descendants,key=lambda item:(len(item),item)):
        depth=len(node)
        if depth>=len(TAXONOMY): continue
        children=[candidate for candidate in descendants if len(candidate)==depth+1 and candidate[:-1]==node]; children.sort(key=lambda item:min(descendants[item]))
        if not children: continue
        radius=radii[depth]; child_angles=[node_angles[child] for child in children]
        if len(child_angles)>1:
            points=[xy(radius,child_angles[0]+(child_angles[-1]-child_angles[0])*i/40) for i in range(41)]; draw.line([(q(x),q(y)) for x,y in points],fill='#7a7a7a',width=line_width)
        for child in children:
            p1=xy(radius,node_angles[child]); p2=xy(radii[len(child)],node_angles[child]); draw.line((q(p1[0]),q(p1[1]),q(p2[0]),q(p2[1])),fill='#7a7a7a',width=line_width)
    display=genus["Phylum"].where(genus["Phylum"].isin(PHYLUM_COLORS),"Other")
    for angle,phylum in zip(angles,display):
        x,y=xy(214,angle); r=q(2.3); draw.ellipse((q(x)-r,q(y)-r,q(x)+r,q(y)+r),fill=PHYLUM_COLORS[str(phylum)])
    inner,ring_width,gap=222.0,13.0,10.0
    for index,(_,count_column,module) in enumerate(RINGS):
        if index in (4,8): inner+=gap
        outer=inner+ring_width; column=count_column.replace('_count','_prevalence_pct')
        for angle,value in zip(angles,genus[column]): draw.polygon([(q(x),q(y)) for x,y in polygon(inner,outer,angle,step*.49)],fill=blend_white(MODULE_COLORS[module],value),outline='white')
        inner=outer
    font=ImageFont.truetype(str(FONT_PATH),round(FONT_PT*DPI/72*SCALE)); x0=888; draw.text((q(x0),q(237)),'Phylum',fill='black',font=font,anchor='ls')
    present=set(display); order=[name for name in PHYLUM_COLORS if name in present]
    for index,name in enumerate(order):
        y=262+index*24; r=q(6); draw.ellipse((q(x0+9)-r,q(y)-r,q(x0+9)+r,q(y)+r),fill=PHYLUM_COLORS[name]); draw.text((q(x0+20),q(y)),name,fill='black',font=font,anchor='lm')
    base=262+len(order)*24+30; titles={"replication":"Replication subcategories (%)","transcription":"Transcription subcategories (%)","translation":"Translation subcategories (%)"}
    for idx,module in enumerate(('replication','transcription','translation')):
        y=base+idx*72; draw.text((q(x0),q(y)),titles[module],fill='black',font=font,anchor='ls')
        for i in range(101): draw.rectangle((q(x0+i*.82),q(y+14),q(x0+(i+1)*.82),q(y+28)),fill=blend_white(MODULE_COLORS[module],i))
        draw.rectangle((q(x0),q(y+14),q(x0+82),q(y+28)),outline='#777',width=line_width)
        for value,x in zip(('0','50','100'),(x0,x0+41,x0+82)): draw.text((q(x),q(y+43)),value,fill='black',font=font,anchor='mm')
    image.resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS).save(PNG,dpi=(DPI,DPI))


def main() -> int:
    if not FONT_PATH.is_file(): raise FileNotFoundError('Arial is required')
    genus=load_genus(); genus.to_csv(PLOT_DATA,index=False); FIGURE_DIR.mkdir(parents=True,exist_ok=True)
    step=2*math.pi/len(genus); angles=[-math.pi/2+index*step for index in range(len(genus))]
    build_svg(genus,angles,step); build_png(genus,angles,step); print(f'Retained genera: {len(genus)}'); return 0


if __name__=='__main__': raise SystemExit(main())
