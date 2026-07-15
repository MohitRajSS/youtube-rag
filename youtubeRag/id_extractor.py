from urllib.parse import urlparse, parse_qs


def get_video_id(url: str):
    parsed = urlparse(url)

    # youtu.be/<id>
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]

    # youtube.com/watch?v=<id>
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]

        elif parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]

        elif parsed.path.startswith("/embed/"):
            return parsed.path.split("/")[2]

    return None