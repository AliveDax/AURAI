"""Command-line entry point: recommend artworks for a room.

    .\\venv\\Scripts\\python.exe main.py --room data/rooms/a.jpg [--room ...] --mood "calming forest" [--color "#2a6f97"] --top 5
    .\\venv\\Scripts\\python.exe main.py --emotion-only --mood calm --top 20      (emotion score only, no room)
"""
import argparse
from pathlib import Path

from artrec.config import load_config, setup_logging
from artrec.recommend import UserInput, recommend, recommend_emotion


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommend artworks that suit a room and a mood.")
    ap.add_argument("--room", type=Path, action="append", help="room photo (repeat for several)")
    ap.add_argument("--mood", default=None, help='mood and/or subject, e.g. "calming blue forest"')
    ap.add_argument("--color", default=None, help='preferred colour as hex, e.g. "#2a6f97"')
    ap.add_argument("--top", type=int, default=None, help="number of results (default: config top_n)")
    ap.add_argument("--emotion-only", action="store_true",
                    help="rank by the emotion in --mood alone: no room photo, style, colour or description")
    args = ap.parse_args()
    setup_logging()

    if args.emotion_only:
        if not args.mood:
            ap.error("--emotion-only needs --mood, e.g. --mood calm")
        if args.room or args.color:
            ap.error("--emotion-only uses the emotion alone; leave out --room and --color")
        recs = recommend_emotion(args.mood, top_n=args.top)
        print(f"\nEmotion used: {recs[0].parsed_text.emotion_text!r}")
        for r in recs:
            print(f"{r.rank:>3}. cosine {r.components['emotion']['raw']:.4f}  EmoArt: {r.dominant_emotion:<12} "
                  f"{r.title or '(untitled)'}  [{r.style}]  {r.image_path}")
        return

    if not args.room:
        ap.error("--room is required (or use --emotion-only)")
    for p in args.room:
        if not p.exists():
            ap.error(f"room photo not found: {p}")
    cfg = load_config()
    recs = recommend(UserInput(args.room, args.mood, args.color), top_n=args.top, cfg=cfg)

    parsed = recs[0].parsed_text
    print(f"\nText parsed as: emotion={parsed.emotion_text!r}  description={parsed.description_text!r}  "
          f"colours={parsed.colors}  (text_mode={cfg.text['text_mode']})")
    for r in recs:
        shares = ", ".join(f"{k} {v['share']:.0%}" for k, v in sorted(r.components.items(), key=lambda kv: -kv[1]["share"]))
        print(f"\n{r.rank}. {r.title or '(untitled)'} - {r.artist or 'unknown artist'}  [{r.style}]")
        print(f"   final {r.final_score:.3f}  |  {shares}  |  EmoArt label: {r.dominant_emotion}")
        notes = []
        if r.subject_match is not None:
            notes.append(f"shows '{parsed.description_text}': {'yes' if r.subject_match else 'no'}")
        if r.color_match is not None:
            exact = r.color_match >= cfg.color["color_match_min_share"]
            notes.append(f"your colour: {r.color_match:.0%} of the artwork"
                         + (f" (picked for your colour{'' if exact else ', closest shade'})" if r.picked_for_color else ""))
        if notes:
            print("   " + "  |  ".join(notes))
        for k, v in r.components.items():
            print(f"   {k:<12} raw {v['raw']:.4f}  norm {v['norm']:.3f}  weight {v['weight']:.2f}  share {v['share']:.0%}")
        print(f"   {r.image_path}")


if __name__ == "__main__":
    main()
