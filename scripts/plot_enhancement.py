#!/usr/bin/env python3
"""Render aligned waveform, spectrogram and pause views from level-matched WAVs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import numpy as np
import soundfile as sf
from scipy import signal

COLORS = ("#61768d", "#8973bd", "#c56b22", "#008975", "#9a4276", "#274d9a", "#9b3431")
INK = "#132638"
MUTED = "#4c6277"
GRID = "#d8e3ec"
PAPER = "#f4f8fb"
DB_MIN, DB_MAX = -85, -25
SR = 16000


def read(path):
    x, sr = sf.read(path, dtype="float32")
    if sr != SR or x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"{path}: expected a finite 16 kHz mono WAV")
    return x


def title(fig, tag, main, subtitle):
    fig.text(.028, .96, f"{tag}  ·  {main}", fontsize=24, weight="bold", color=INK, va="top")
    fig.text(.029, .96 - .44/fig.get_figheight(), subtitle, fontsize=11, color=MUTED, va="top")


def tracks_plot(tag, tracks, output, mode, pause_at):
    spectrogram = mode == "spectrogram"
    fig, axes = plt.subplots(len(tracks), 1, figsize=(14, 2.3+1.10*len(tracks)), sharex=True,
                             squeeze=False, facecolor=PAPER)
    axes = axes[:, 0]
    fig.subplots_adjust(left=.17, right=.875 if spectrogram else .962, bottom=.10, top=.845,
                        hspace=.17)
    description = ("固定响度 · 同时间轴 · 统一色标  |  频率 100 Hz–8 kHz"
                   if spectrogram else "固定响度 · 同时间轴 · 共用振幅刻度")
    title(fig, tag, "增强后级频谱" if spectrogram else "增强后级波形", description)
    length = len(tracks[0][1])
    duration = length / SR
    max_amp = min(1.0, max(float(np.abs(x).max()) for _, x in tracks) * 1.055)
    image = None
    for index, (ax, (label, x)) in enumerate(zip(axes, tracks)):
        ax.text(-.04, .5, label, transform=ax.transAxes, ha="right", va="center",
                fontsize=11, weight="bold", color=INK, clip_on=False)
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        if spectrogram:
            hz, times, z = signal.stft(x, SR, nperseg=512, noverlap=384, scaling="spectrum")
            db = 20 * np.log10(np.maximum(2 * np.abs(z[1:]), 1e-6))
            image = ax.pcolormesh(times, hz[1:], db, cmap="magma", vmin=DB_MIN,
                                  vmax=DB_MAX, shading="auto", rasterized=True)
            ax.set_ylim(100, SR/2)
            ax.set_yscale("log")
            ax.set_yticks([125, 500, 2000, 8000], ["125", "500", "2k", "8k"])
        else:
            width = max(1, int(np.ceil(length / 1550)))
            start = np.arange(0, length, width)
            center = (start + (np.minimum(start + width, length) - start) / 2) / SR
            lower, upper = np.minimum.reduceat(x, start), np.maximum.reduceat(x, start)
            ax.fill_between(center, lower, upper, step="mid", color=COLORS[index % len(COLORS)],
                            alpha=.9, linewidth=0)
            ax.axhline(0, color="#8496a7", lw=.5)
            ax.set_ylim(-max_amp, max_amp)
            ax.set_yticks([0])
            ax.grid(axis="x", color=GRID, lw=.65)
        for t in pause_at:
            ax.axvspan(t+.048, t+.208, color="#7ad4da", alpha=.57 if spectrogram else .26, lw=0)
        ax.tick_params(axis="both", labelsize=9, colors=MUTED, length=0, pad=5)
    if image is not None:
        cax = fig.add_axes([.909, .255, .013, .43])
        colorbar = fig.colorbar(image, cax=cax, ticks=[-85, -65, -45, -25])
        colorbar.ax.tick_params(labelsize=9, colors=MUTED, length=0)
        colorbar.outline.set_edgecolor(GRID)
        colorbar.ax.set_xlabel("dBFS", color=MUTED, fontsize=9, labelpad=8)
    axes[-1].set_xlim(0, duration)
    axes[-1].xaxis.set_major_locator(MultipleLocator(5 if duration < 50 else 10))
    axes[-1].set_xlabel("时间（秒）", color=MUTED, labelpad=9)
    fig.savefig(output, dpi=165, facecolor=fig.get_facecolor())
    plt.close(fig)


def pause_plot(tag, tracks, output, times):
    fig, axes = plt.subplots(1, len(times), figsize=(14, 4.7), sharey=True,
                             squeeze=False, facecolor=PAPER)
    fig.subplots_adjust(left=.075, right=.97, bottom=.17, top=.68, wspace=.19)
    title(fig, tag, "句尾停顿放大", "300–3400 Hz 短时能量  |  与原波形使用相同的响度匹配")
    handles = []
    for ax, t in zip(axes[0], times):
        ax.axvspan(t+.048, t+.208, color="#b2e6e9", alpha=.67, lw=0)
        for index, (name, x) in enumerate(tracks):
            a = max(0, round((t-.38)*SR))
            b = min(len(x), round((t+.57)*SR))
            bp = signal.butter(4, [300, 3400], fs=SR, btype="bandpass", output="sos")
            y = signal.sosfiltfilt(bp, x[a:b])
            hop = 128
            n = len(y)//hop
            db = 10*np.log10(np.maximum((y[:n*hop].reshape(n, hop)**2).mean(axis=1), 1e-9))
            time = (a + (np.arange(n)+.5)*hop)/SR
            line, = ax.plot(time, db, color=COLORS[index % len(COLORS)], label=name, lw=1.40)
            if ax is axes[0][0]:
                handles.append(line)
        ax.set_title(f"{t:.2f} 秒附近", fontsize=12, color=INK, pad=9)
        ax.set_facecolor("white")
        ax.set_xlim(t-.35, t+.53)
        ax.set_ylim(-85, -15)
        ax.grid(axis="y", color=GRID, lw=.65)
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.xaxis.set_major_locator(MultipleLocator(.2))
        ax.set_xlabel("时间（秒）", fontsize=9, color=MUTED)
    axes[0][0].set_ylabel("短时电平（dBFS）", fontsize=10, color=MUTED)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.024, .815), ncol=len(handles),
               frameon=False, fontsize=10, handlelength=2.0, columnspacing=1.6)
    fig.savefig(output, dpi=165, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON mapping clip IDs to labeled, aligned WAVs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, action="append", default=[],
                        help="Register a CJK font file for this render")
    args = parser.parse_args()
    for font in args.font:
        font_manager.fontManager.addfont(font)
    available = {font.name for font in font_manager.fontManager.ttflist}
    families = ("Noto Sans CJK SC", "Microsoft YaHei", "Noto Sans CJK JP", "WenQuanYi Zen Hei")
    cjk = next((name for name in families if name in available), None)
    if cjk is None:
        parser.error("A CJK font is required. Pass its font file via --font.")
    plt.rcParams.update({"font.family": [cjk, "DejaVu Sans"], "axes.unicode_minus": False})
    clips = json.loads(args.manifest.read_text())
    args.output.mkdir(exist_ok=True, parents=True)
    for tag, record in clips.items():
        tracks = [(label, read(args.manifest.parent / source)) for label, source in record["tracks"].items()]
        if not tracks or len({len(x) for _, x in tracks}) != 1:
            raise ValueError(f"{tag}: all tracks must have the same sample count")
        pauses = record.get("pauses", [])
        for style in ("waveform", "spectrogram"):
            tracks_plot(tag, tracks, args.output / f"{tag}-{style}.png", style, pauses)
        if pauses:
            pause_plot(tag, tracks, args.output / f"{tag}-pauses.png", pauses)


if __name__ == "__main__":
    main()
