import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

XML_PATH = ROOT / "data" / "knowledge_base" / "medlineplus" / "mplus_topics_2026-10-24.xml"
OUTPUT_PATH = ROOT / "data" / "knowledge_base" / "medlineplus" / "medical_knowledge.json"


def clean_text(value):
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(value.split())


def load_medlineplus_xml(xml_path=XML_PATH):
    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    topics = []

    for _, element in ET.iterparse(xml_path, events=("end",)):
        if element.tag != "health-topic":
            continue

        if element.attrib.get("language", "").lower() != "english":
            element.clear()
            continue

        title = element.attrib.get("title", "").strip()
        summary_element = element.find("full-summary")

        summary = (
            clean_text("".join(summary_element.itertext()))
            if summary_element is not None
            else ""
        )

        if title:
            topics.append({
                "id": element.attrib.get("id", ""),
                "title": title,
                "summary": summary,
                "source": "MedlinePlus",
                "source_url": element.attrib.get("url", ""),
                "language": "English",
            })

        element.clear()

    return topics


def build_medlineplus_knowledge_base(
    xml_path=XML_PATH,
    output_path=OUTPUT_PATH,
):
    topics = load_medlineplus_xml(xml_path)

    if not topics:
        raise ValueError("No English medical topics were extracted.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(topics, file, ensure_ascii=False, indent=2)

    return {
        "status": "SUCCESS",
        "topics_extracted": len(topics),
        "output_path": str(output_path),
        "source": "MedlinePlus",
    }


if __name__ == "__main__":
    print(build_medlineplus_knowledge_base())
