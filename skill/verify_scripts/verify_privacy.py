"""检查发布文件中的认证材料、个人主目录及 Office 作者元数据，不回显命中内容。"""
from pathlib import Path
import io
import json
import re
import subprocess
import sys
import zipfile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
RULES = {
    "private_key": re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "ssh_public_key": re.compile(rb"(?:ssh-(?:ed25519|rsa)|ecdsa-sha2-[^\s]+)\s+[A-Za-z0-9+/]{32,}={0,3}"),
    "access_token": re.compile(rb"(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{25,}|sk-proj-[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16})"),
    "personal_home": re.compile(rb"[A-Za-z]:[\\/](?:Users|CodexData)[\\/][^\s<>\"\x00]+", re.I),
    "hardcoded_credential": re.compile(rb"(?i)\b(?:password|passwd|api_key|api_token|access_token)\s*[:=]\s*[\"'](?![<{$])[^\"'\r\n]{6,}[\"']"),
}
OFFICE = {".docx", ".pptx", ".xlsx"}
ARCHIVES = OFFICE | {".zip", ".npz", ".pt"}
issues = []


def check(name, data, depth=0):
    for kind, pattern in RULES.items():
        if pattern.search(data):
            issues.append({"file": name, "kind": kind})
    suffix = Path(name).suffix.lower()
    if suffix in ARCHIVES and depth < 3 and zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                value = archive.read(item)
                child_name = name + "!" + item.filename
                check(child_name, value, depth + 1)
                if suffix in OFFICE and item.filename == "docProps/core.xml":
                    root = ET.fromstring(value)
                    for element in root:
                        if element.tag.split("}")[-1] in {"creator", "lastModifiedBy"} and (element.text or "").strip():
                            issues.append({"file": child_name, "kind": "office_author_metadata"})
                if suffix in OFFICE and item.filename == "docProps/custom.xml":
                    issues.append({"file": child_name, "kind": "office_custom_metadata"})


def main():
    # 只输出类别和文件；审计日志本身不保存凭据或原文片段。
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    count = 0
    for name in files:
        path = ROOT / name
        if name and path.is_file():
            check(name, path.read_bytes())
            count += 1
    trace = ROOT / "report/llm_trace"
    if trace.exists() and any(p.name != "README.md" for p in trace.iterdir()):
        issues.append({"file": "report/llm_trace", "kind": "raw_conversation_export"})
    print(json.dumps({"passed": not issues, "files_checked": count, "issues": issues}, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
