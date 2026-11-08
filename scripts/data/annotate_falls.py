#!/usr/bin/env python3
"""
Annoteringsverktøy for fallstart/-slutt, lagrer i samme struktur som Le2i-fasiten.

For en video i
    FallDataset/<Rom>/<Rom>/Videos/video (1).avi
skrives fasiten til
    FallDataset/<Rom>/<Rom>/Annotation_files/video (1).txt
(Ligger videoen direkte i rommappen, havner Annotation_files i samme mappe.)

Filformat (som de to første linjene i originalfilene):
    <fallstart>
    <fallslutt>
Ikke-fall lagres som 0 og 0. Ingen per-frame bbox-linjer skrives.

Eksisterende .txt-filer blir ALDRI overskrevet; videoer som allerede har fasit hoppes over.
Alle egne annoteringer logges i own_annotations_log.csv, så dere alltid kan skille
egne labels fra originale.

Bruk (test først med --dry-run for å se hvor filene havner):
    python annotate_falls.py "C:\\Users\\Bruker\\IKT213\\FallDataset" --dry-run
    python annotate_falls.py "C:\\Users\\Bruker\\IKT213\\FallDataset" --annotator ola

Taster:
    SPACE      spill av / pause
    d / a      +1 / -1 frame
    l / j      +25 / -25 frames (1 sekund ved 25 fps)
    b          sett FALLSTART på nåværende frame
    e          sett FALLSLUTT på nåværende frame
    r          nullstill markeringer
    ENTER      lagre (krever start og slutt) og gå til neste video
    n          lagre som IKKE-FALL (0 / 0) og gå til neste
    x          hopp over video uten å lagre
    q / ESC    avslutt

Frame-numre er 1-indeksert, slik som i Le2i-fasiten.
"""
import argparse
import csv
import datetime
import sys
from pathlib import Path

import cv2

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv"}
LOG_FIELDS = ["video", "annotation_file", "fall_start", "fall_end", "n_frames",
              "annotator", "timestamp"]


def collect_videos(inputs):
    vids = []
    for p in map(Path, inputs):
        if p.is_dir():
            vids += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXT)
        elif p.suffix.lower() in VIDEO_EXT:
            vids.append(p)
    return vids


def annotation_path(video):
    folder = video.parent
    base = folder.parent if folder.name.lower() == "videos" else folder
    return base / "Annotation_files" / (video.stem + ".txt")


def load_frames(path):
    # Hele videoen leses inn i minnet: 320x240-videoer er små, og presis
    # frame-hopping blir pålitelig (seeking i AVI kan ellers være unøyaktig).
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return frames, fps


def write_annotation(txt_path, start, end):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "x", encoding="utf-8") as f:  # "x" = feiler hvis filen finnes
        f.write(f"{start}\n{end}\n")


def append_log(log_path, row):
    new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def render(frame, idx, n, start, end, playing, scale, name):
    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    fnum = idx + 1
    in_fall = start and end and start <= fnum <= end

    lines = [
        name,
        f"frame {fnum}/{n}   {'PLAY' if playing else 'PAUSE'}",
        f"start: {start or '-'}   slutt: {end or '-'}",
    ]
    for i, t in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Tidslinje nederst: grå = video, rød = markert fall, hvit strek = nåværende frame
    bar_y, bar_h = h - 14, 8
    cv2.rectangle(img, (0, bar_y), (w, bar_y + bar_h), (80, 80, 80), -1)
    if start:
        stop = end or fnum
        x0 = int((start - 1) / n * w)
        x1 = int((stop - 1) / n * w) if stop >= start else x0
        cv2.rectangle(img, (x0, bar_y), (max(x1, x0 + 2), bar_y + bar_h), (0, 0, 220), -1)
    cx = int(idx / max(n - 1, 1) * (w - 1))
    cv2.line(img, (cx, bar_y - 4), (cx, bar_y + bar_h + 2), (255, 255, 255), 2)

    if in_fall:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 220), 3)
    return img


def annotate(path, name, scale, win):
    frames, fps = load_frames(path)
    n = len(frames)
    if n == 0:
        print(f"  Kunne ikke lese {path}, hopper over.")
        return ("skip",)

    idx, start, end, playing = 0, None, None, False
    delay = max(1, int(1000 / fps))

    while True:
        cv2.imshow(win, render(frames[idx], idx, n, start, end, playing, scale, name))
        key = cv2.waitKey(delay if playing else 0) & 0xFF

        if playing and key == 255:  # ingen tast under avspilling
            idx = min(idx + 1, n - 1)
            if idx == n - 1:
                playing = False
            continue

        if key == ord(" "):
            playing = not playing
        elif key == ord("d"):
            idx = min(idx + 1, n - 1)
        elif key == ord("a"):
            idx = max(idx - 1, 0)
        elif key == ord("l"):
            idx = min(idx + 25, n - 1)
        elif key == ord("j"):
            idx = max(idx - 25, 0)
        elif key == ord("b"):
            start, playing = idx + 1, False
        elif key == ord("e"):
            end, playing = idx + 1, False
        elif key == ord("r"):
            start = end = None
        elif key in (13, 10):  # ENTER
            if start and end and end >= start:
                return ("save", start, end, n)
            print("  Sett både start (b) og slutt (e), med slutt >= start.")
        elif key == ord("n"):
            return ("save", 0, 0, n)
        elif key == ord("x"):
            return ("skip",)
        elif key in (ord("q"), 27):
            return ("quit",)


def main():
    ap = argparse.ArgumentParser(description="Annoter fallstart/-slutt i Le2i-struktur.")
    ap.add_argument("inputs", nargs="+", help="Videofiler eller mapper (søkes rekursivt)")
    ap.add_argument("--annotator", default="unknown", help="Navn/initialer på den som annoterer")
    ap.add_argument("--log", default="own_annotations_log.csv",
                    help="Logg over egenproduserte annoteringer")
    ap.add_argument("--scale", type=float, default=2.0, help="Forstørrelse av visningen")
    ap.add_argument("--dry-run", action="store_true",
                    help="Vis hvilke videoer som mangler fasit og hvor .txt havner, uten å annotere")
    args = ap.parse_args()

    videos = collect_videos(args.inputs)
    if not videos:
        sys.exit("Fant ingen videoer.")

    todo = [(v, annotation_path(v)) for v in videos if not annotation_path(v).exists()]
    print(f"{len(videos)} videoer funnet, {len(videos) - len(todo)} har fasit, "
          f"{len(todo)} mangler.")

    if args.dry_run:
        for v, txt in todo:
            print(f"  {v}\n    -> {txt}")
        return

    log = Path(args.log)
    win = "annotate_falls"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    for i, (v, txt) in enumerate(todo, 1):
        name = f"{v.parent.parent.name}/{v.name}"
        print(f"[{i}/{len(todo)}] {v}")
        res = annotate(v, name, args.scale, win)
        if res[0] == "quit":
            break
        if res[0] == "save":
            _, s, e, n = res
            try:
                write_annotation(txt, s, e)
            except FileExistsError:
                print(f"  {txt} finnes allerede, lagrer ikke.")
                continue
            append_log(log, {
                "video": str(v), "annotation_file": str(txt),
                "fall_start": s, "fall_end": e, "n_frames": n,
                "annotator": args.annotator,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            })
            print(f"  Lagret {txt.name}: start={s}, slutt={e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()