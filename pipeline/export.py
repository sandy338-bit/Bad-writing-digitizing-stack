from .schemas import Transcription


def render_text(result: Transcription, mark_review: bool = True) -> str:
    lines = []
    for region in result.regions:
        if region.words:
            values = []
            for word in region.words:
                value = word.text
                if word.status == "crossed_out":
                    value = "[crossed out]"
                elif mark_review and word.status == "review":
                    value = f"[[review: {value}]]"
                values.append(value)
            value = " ".join(values)
            lines.append(value)
            continue
        value = region.text
        if region.status == "crossed_out":
            value = "[crossed out]"
        elif mark_review and region.status == "review":
            value = f"[[review: {value}]]"
        lines.append(value)
    body = "\n".join(lines)
    if mark_review and result.page_status == "review":
        body = "[[page requires verification]]\n" + body
    return body
