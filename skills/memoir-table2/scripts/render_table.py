#!/usr/bin/env python3
"""Render a Memoir Table II-style Seen/Unseen table from explicitly supplied results."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


METRICS = ("TL ↓", "NE ↓", "nDTW ↑", "SR ↑", "SPL ↑", "t-nDTW ↑")
SPLITS = ("seen", "unseen")


@dataclass
class Cell:
    value: str
    std: str = ""
    bold: bool = False

    def text(self):
        return self.value + (" ± " + self.std if self.std else "")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def format_value(value, decimals, label):
    if value is None:
        return "—"
    if isinstance(value, bool):
        raise ValueError(f"{label} cannot be a boolean")
    if isinstance(value, (float, int)):
        require(math.isfinite(value), f"{label} must be finite")
        return f"{value:.{decimals}f}"
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a number, text, or null")
    text = value.strip()
    require(text.lower() not in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"},
            f"{label} must be finite")
    return text


def parse_cell(value, decimals, label):
    bold = False
    std = ""
    if isinstance(value, dict):
        require("value" in value and not (set(value) - {"value", "std", "bold"}),
                f"{label} accepts value, std, and bold")
        bold = value.get("bold", False)
        require(isinstance(bold, bool), f"{label}.bold must be boolean")
        if value.get("std") is not None:
            require(value["value"] is not None, f"{label}: uncertainty needs a value")
            std = format_value(value["std"], decimals, label + ".std")
            try:
                require(float(std) >= 0 and math.isfinite(float(std)), f"{label}.std must be finite and nonnegative")
            except ValueError as exc:
                raise ValueError(f"{label}.std must be finite and nonnegative") from exc
        value = value["value"]
    text = format_value(value, decimals, label)
    if "±" in text:
        require(not std and text.count("±") == 1, f"{label} has ambiguous uncertainty")
        text, std = (part.strip() for part in text.split("±"))
        require(bool(text) and bool(std), f"{label} has an incomplete uncertainty")
        format_value(text, decimals, label)
        try:
            require(float(std) >= 0 and math.isfinite(float(std)), f"{label}.std must be finite and nonnegative")
        except ValueError as exc:
            raise ValueError(f"{label}.std must be finite and nonnegative") from exc
    return Cell(text, std, bold)


def load_table(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "Input must be a JSON object")
    decimals = data.get("decimals", 2)
    require(type(decimals) is int and 0 <= decimals <= 8, "decimals must be an integer between 0 and 8")
    require(isinstance(data.get("caption", ""), str), "caption must be text")
    notes = data.get("notes", [])
    require(isinstance(notes, list) and all(isinstance(note, str) for note in notes), "notes must be a list of strings")
    require(isinstance(data.get("rows"), list) and data["rows"], "rows must be a nonempty list")
    rows = []
    for index, raw in enumerate(data["rows"]):
        label = f"rows[{index}]"
        require(isinstance(raw, dict), f"{label} must be an object")
        method = raw.get("method")
        require(isinstance(method, str) and bool(method.strip()), f"{label}.method must be nonempty text")
        row = {"method": method.expandtabs(2).rstrip()}
        for flag in ("highlight", "group_before"):
            row[flag] = raw.get(flag, False)
            require(isinstance(row[flag], bool), f"{label}.{flag} must be boolean")
        for split in SPLITS:
            require(isinstance(raw.get(split), list) and len(raw[split]) == 6,
                    f"{label}.{split} must contain exactly six values in TL, NE, nDTW, SR, SPL, t-nDTW order")
            row[split] = [parse_cell(value, decimals, f"{label}.{split}[{i}]")
                          for i, value in enumerate(raw[split])]
        rows.append(row)
    return {"caption": data.get("caption", ""), "notes": notes, "rows": rows}


def markdown(table):
    def escape(text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|").replace("\n", "<br>")

    lines = [table["caption"], ""] if table["caption"] else []
    headers = ["Methods"] + [f"Val {split.title()} {metric}" for split in SPLITS for metric in METRICS]
    lines += ["| " + " | ".join(headers) + " |", "| --- | " + " | ".join(["---:"] * 12) + " |"]
    for row in table["rows"]:
        indent = len(row["method"]) - len(row["method"].lstrip(" "))
        method = "&nbsp;" * indent + escape(row["method"].lstrip(" "))
        values = [f"**{method}**" if row["highlight"] else method]
        for split in SPLITS:
            for cell in row[split]:
                text = escape(cell.text())
                values.append(f"**{text}**" if row["highlight"] or cell.bold else text)
        lines.append("| " + " | ".join(values) + " |")
    for note in table["notes"]:
        lines.extend(["", note])
    return "\n".join(lines) + "\n"


def draw(table, stem, formats, font_path, font_size):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager, pyplot as plt
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise SystemExit("Image and PDF rendering requires matplotlib. Install it in the Python environment used for this command.") from exc

    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        families = [FontProperties(fname=str(font_path)).get_name()]
    else:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        families = [name for name in ("Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif")
                    if name in installed]
        all_text = table["caption"] + "".join(table["notes"]) + "".join(row["method"] for row in table["rows"])
        if re.search(r"[\u2e80-\u9fff\uf900-\ufaff]", all_text):
            families.extend(name for name in ("Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", "SimSun", "PingFang SC")
                            if name in installed)
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False})
    figure = plt.figure(figsize=(12, 3), dpi=300, facecolor="white")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    def font(size, bold=False):
        return FontProperties(family=families, size=size, weight="bold" if bold else "normal")

    def width(text, size=font_size, bold=False):
        return max((renderer.get_text_width_height_descent(line, font(size, bold), False)[0]
                    for line in text.split("\n")), default=0) * 72 / figure.dpi

    def wrapped(text, available, size):
        lines = []
        for paragraph in text.split("\n"):
            line = ""
            # Break on spaces for Latin text and on characters for CJK text.
            tokens = re.findall(r"[\u2e80-\u9fff\uf900-\ufaff]|[^\s\u2e80-\u9fff\uf900-\ufaff]+|[ \t]+", paragraph)
            for token in tokens:
                if width(line + token, size) <= available:
                    line += token
                    continue
                if line.strip():
                    lines.append(line.rstrip())
                    line = ""
                token = token.lstrip()
                for character in token:
                    if line and width(line + character, size) > available:
                        lines.append(line)
                        line = ""
                    line += character
            lines.append(line.rstrip())
        return lines

    gap = font_size * 0.9
    method_width = max(width("Methods"), *(width(r["method"], bold=r["highlight"]) for r in table["rows"])) + gap * 2
    cell_widths = []
    for i, metric in enumerate(METRICS):
        sizes = [width(metric, bold=True)]
        for row in table["rows"]:
            for split in SPLITS:
                cell = row[split][i]
                bold = row["highlight"] or cell.bold
                size = width(cell.value, bold=bold)
                if cell.std:
                    size += font_size * .2 + width("±" + cell.std, font_size * .76, bold)
                sizes.append(size)
        cell_widths.append(max(sizes) + gap)
    split_width = sum(cell_widths)
    total_width = method_width + split_width * 2 + gap
    note_size = font_size * .88
    caption_lines = wrapped(table["caption"], total_width, font_size) if table["caption"] else []
    note_lines = [wrapped(note, total_width - 2 * gap, note_size) for note in table["notes"]]
    row_heights = [font_size * (1.6 + 1.15 * (row["method"].count("\n"))) for row in table["rows"]]
    caption_height = len(caption_lines) * font_size * 1.25 + (font_size * .8 if caption_lines else 0)
    header_height = font_size * 3.0
    notes_height = sum(len(lines) * note_size * 1.25 + note_size * .35 for lines in note_lines)
    margin = font_size * .75
    table_top = margin + caption_height
    body_top = table_top + header_height
    bottom = body_top + sum(row_heights)
    height = bottom + (font_size * .6 if note_lines else 0) + notes_height + margin
    figure.set_size_inches((total_width + 2 * margin) / 72, height / 72)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(-margin, total_width + margin)
    axes.set_ylim(height, 0)
    axes.axis("off")

    def text(x, y, content, size=font_size, bold=False, ha="left"):
        return axes.text(x, y, content, fontproperties=font(size, bold), ha=ha, va="center",
                         linespacing=1.15, parse_math=False)

    def rule(y, start=0, end=total_width, weight=.55):
        axes.plot([start, end], [y, y], color="black", linewidth=weight, solid_capstyle="butt")

    y = margin + font_size * .6
    for line in caption_lines:
        text(total_width / 2, y, line, ha="center")
        y += font_size * 1.25
    rule(table_top, weight=.85)
    text(gap * .3, table_top + header_height * .72, "Methods")
    split_starts = [method_width, method_width + split_width + gap]
    for label, start in zip(("Val Seen", "Val Unseen"), split_starts):
        text(start + split_width / 2, table_top + font_size * .72, label, bold=True, ha="center")
        rule(table_top + font_size * 1.37, start + gap * .15, start + split_width - gap * .15)
        cursor = start
        for metric, col_width in zip(METRICS, cell_widths):
            text(cursor + col_width / 2, table_top + font_size * 2.13, metric, bold=True, ha="center")
            cursor += col_width
    rule(body_top - font_size * .15, weight=.45)
    rule(body_top + font_size * .05, weight=.45)
    y = body_top
    for index, (row, row_height) in enumerate(zip(table["rows"], row_heights)):
        if row["highlight"]:
            axes.add_patch(Rectangle((0, y + font_size * .15), total_width, row_height - font_size * .15,
                                     facecolor="#eeeeee", edgecolor="none", zorder=0))
        if index and row["group_before"]:
            rule(y, weight=.55)
        center = y + row_height / 2 + font_size * .05
        text(gap * .3, center, row["method"], bold=row["highlight"])
        for split, start in zip(SPLITS, split_starts):
            cursor = start
            for cell, col_width in zip(row[split], cell_widths):
                bold = row["highlight"] or cell.bold
                if cell.std:
                    primary = width(cell.value, bold=bold)
                    secondary = width("±" + cell.std, font_size * .76, bold)
                    x = cursor + (col_width - primary - secondary - font_size * .2) / 2
                    text(x, center, cell.value, bold=bold)
                    text(x + primary + font_size * .2, center + font_size * .04,
                         "±" + cell.std, size=font_size * .76, bold=bold)
                else:
                    text(cursor + col_width / 2, center, cell.value, bold=bold, ha="center")
                cursor += col_width
        y += row_height
    rule(bottom + font_size * .1, weight=1.0)
    y = bottom + font_size * .6 + note_size * .6
    for lines in note_lines:
        for line in lines:
            text(gap * .3, y, line, size=note_size)
            y += note_size * 1.25
        y += note_size * .35
    for extension in formats:
        figure.savefig(stem.with_suffix("." + extension), dpi=300, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output filename stem")
    parser.add_argument("--formats", default="png,pdf,md", help="Comma-separated selection from png,pdf,md")
    parser.add_argument("--font", type=Path, help="Optional font file, including a CJK font when needed")
    parser.add_argument("--font-size", type=float, default=10, help="Body type size in points")
    args = parser.parse_args()
    formats = list(dict.fromkeys(value.strip() for value in args.formats.split(",")))
    require(formats and set(formats) <= {"png", "pdf", "md"}, "formats must select png, pdf, or md")
    require(math.isfinite(args.font_size) and args.font_size > 0, "font-size must be positive")
    require(args.font is None or args.font.is_file(), "font file does not exist")
    table = load_table(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graphics = [kind for kind in formats if kind != "md"]
    if graphics:
        draw(table, args.output, graphics, args.font, args.font_size)
    if "md" in formats:
        args.output.with_suffix(".md").write_text(markdown(table), encoding="utf-8")
    for kind in formats:
        print(args.output.with_suffix("." + kind).resolve())


if __name__ == "__main__":
    main()
