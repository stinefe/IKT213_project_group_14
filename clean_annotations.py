"""Recreate cleaned Home annotations; preserve original files and metadata."""
import cv2
from config import DATASET_ROOT, require_dataset_root


def clean_lines(lines, frame_count):
    seen = set()
    cleaned = []
    frame_numbers = []
    fall_values = []
    removed = 0
    for line in lines:
        columns = tuple(value.strip() for value in line.strip().split(','))
        if not line.strip():
            cleaned.append(line)
            continue
        if len(columns) == 6:
            frame_number = int(columns[0])
            if columns in seen:
                removed += 1
                continue
            seen.add(columns)
            frame_numbers.append(frame_number)
        elif len(columns) == 1:
            fall_values.append(int(columns[0]))
        else:
            raise ValueError('Unexpected annotation row')
        cleaned.append(line)
    if frame_count <= 0 or frame_numbers != list(range(1, frame_count + 1)):
        raise ValueError('Frame count/order still differs after exact duplicate removal')
    if len(fall_values) != 2:
        raise ValueError('Expected two fall values')
    start, end = fall_values
    if not ((start == end == 0) or (1 <= start <= end <= frame_count)):
        raise ValueError('Invalid fall interval')
    return cleaned, removed


def main():
    require_dataset_root()
    pending = []
    # Validate both subsets completely before writing any cleaned files.
    for name in ('Home_01', 'Home_02'):
        folder = DATASET_ROOT / name / name
        source = folder / 'Annotation_files'
        videos = folder / 'Videos'
        labels = sorted(source.glob('*.txt'))
        video_files = sorted(videos.glob('*.avi'))
        if len(labels) != 30 or len(video_files) != 30:
            raise ValueError(f'{name}: expected 30 original labels and 30 videos')
        if {p.stem for p in labels} != {p.stem for p in video_files}:
            raise ValueError(f'{name}: video/annotation names do not match')
        for label in labels:
            cap = cv2.VideoCapture(str(videos / (label.stem + '.avi')))
            opened = cap.isOpened()
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if not opened or count <= 0:
                raise ValueError(f'Cannot read video for {label}')
            try:
                cleaned, removed = clean_lines(
                    label.read_text(encoding='utf-8-sig').splitlines(keepends=True), count
                )
            except ValueError as error:
                raise ValueError(f'{label}: {error}') from error
            output = folder / 'Annotation_files_cleaned' / label.name
            pending.append((name, output, ''.join(cleaned), removed))
    for name, output, contents, removed in pending:
        output.parent.mkdir(exist_ok=True)
        if output.exists() and output.read_text(encoding='utf-8-sig') == contents:
            print('UNCHANGED:', name, output.name)
        else:
            output.write_text(contents, encoding='utf-8-sig')
            print('SAVED:', name, output.name, '| Duplicate rows removed:', removed)


if __name__ == '__main__':
    main()
