#!/usr/bin/env python3
"""
Onshape Assembly → URDF Converter

Connects to the Onshape REST API, fetches assembly structure and mate features,
downloads STL meshes for each part, and generates a URDF file with proper
links, joints, and transforms.

Usage:
    python my_onshape-to-robot.py [ONSHAPE_ASSEMBLY_URL]

    If no URL is given, reads from my-robot/config.json.
    API credentials are read from .env (ONSHAPE_ACCESS_KEY, ONSHAPE_SECRET_KEY).

Output:
    urdf_output/robot.urdf      - the generated URDF
    urdf_output/meshes/*.stl    - one STL mesh per part
    urdf_output/assembly_raw.json - raw API response (for debugging)

Requirements:
    pip install requests python-dotenv numpy
"""
from __future__ import annotations

import os
import sys
import re
import hmac
import hashlib
import base64
import uuid
import json
import math
import logging
import shutil
from collections import deque
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse
from xml.etree import ElementTree as ET
from xml.dom import minidom

import numpy as np
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Onshape REST client  (API-key + HMAC-SHA256 auth)
# ═══════════════════════════════════════════════════════════════════════════

class OnshapeClient:
    """Authenticated client for the Onshape REST API."""

    def __init__(self, base_url: str, access_key: str, secret_key: str):
        self.base_url = base_url.rstrip("/")
        self.access_key = access_key
        self.secret_key = secret_key

    def _headers(self, method: str, path: str, query: str = "",
                 ctype: str = "application/json") -> dict:
        """Build HMAC-SHA256 auth headers required by the Onshape API."""
        nonce = uuid.uuid4().hex[:25]
        date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

        # Signature = HMAC-SHA256 of:  method\nnonce\ndate\nctype\npath\nquery\n
        # The entire string is lowercased before signing.
        raw = f"{method}\n{nonce}\n{date}\n{ctype}\n{path}\n{query}\n"
        sig = base64.b64encode(
            hmac.new(
                self.secret_key.encode("utf-8"),
                raw.lower().encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        return {
            "Date": date,
            "On-Nonce": nonce,
            "Authorization": f"On {self.access_key}:HmacSHA256:{sig}",
            "Content-Type": ctype,
            "Accept": "application/json",
        }

    def get(self, path: str, params: dict | None = None,
            accept: str = "application/json") -> requests.Response:
        """Authenticated GET.  Follows redirects with re-auth for Onshape domains."""
        qs = urlencode(params) if params else ""
        hdrs = self._headers("GET", path, qs)
        hdrs["Accept"] = accept

        url = f"{self.base_url}{path}" + (f"?{qs}" if qs else "")
        resp = requests.get(url, headers=hdrs, allow_redirects=False, timeout=120)

        # Follow redirects (up to 5 hops)
        for _ in range(5):
            if resp.status_code not in (302, 303, 307):
                break
            redirect_url = resp.headers["Location"]
            parsed = urlparse(redirect_url)

            if parsed.hostname and "onshape.com" in parsed.hostname:
                # Onshape cluster redirect → re-authenticate with new path/query
                redir_hdrs = self._headers("GET", parsed.path, parsed.query or "")
                redir_hdrs["Accept"] = accept
                resp = requests.get(redirect_url, headers=redir_hdrs,
                                    allow_redirects=False, timeout=120)
            else:
                # External redirect (S3 pre-signed, etc.) → no auth needed
                resp = requests.get(redirect_url, timeout=120)
                break

        resp.raise_for_status()
        return resp


# ═══════════════════════════════════════════════════════════════════════════
#  URL parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_onshape_url(url: str) -> tuple[str, str, str, str]:
    """Extract (document_id, wvm_type, wvm_id, element_id) from an Onshape URL.

    Accepts URLs like:
        https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}
        https://cad.onshape.com/documents/{did}/v/{vid}/e/{eid}
    """
    m = re.search(r"/documents/([a-f0-9]+)/(w|v|m)/([a-f0-9]+)/e/([a-f0-9]+)", url)
    if not m:
        raise ValueError(f"Cannot parse Onshape URL: {url}")
    return m.group(1), m.group(2), m.group(3), m.group(4)


# ═══════════════════════════════════════════════════════════════════════════
#  Linear-algebra helpers
# ═══════════════════════════════════════════════════════════════════════════

def flat16_to_T(flat: list[float]) -> np.ndarray:
    """Onshape's row-major 16-element list → 4x4 homogeneous matrix."""
    return np.array(flat, dtype=float).reshape(4, 4)


def matedCS_to_T(cs: dict) -> np.ndarray:
    """Onshape matedCS {origin, xAxis, yAxis, zAxis} → 4x4 transform."""
    T = np.eye(4)
    T[:3, 0] = cs["xAxis"]
    T[:3, 1] = cs["yAxis"]
    T[:3, 2] = cs["zAxis"]
    T[:3, 3] = cs["origin"]
    return T


def T_to_xyz_rpy(T: np.ndarray) -> tuple[tuple, tuple]:
    """4x4 matrix → (x,y,z), (roll,pitch,yaw).

    Uses the URDF fixed-axis ZYX convention (intrinsic X-Y-Z = extrinsic Z-Y-X).
    """
    xyz = tuple(T[:3, 3].tolist())
    R = T[:3, :3]
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        roll  = math.atan2( R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2( R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
    return xyz, (roll, pitch, yaw)


def _fmt(vals, prec: int = 8) -> str:
    """Format a number or sequence of numbers for XML attributes."""
    if isinstance(vals, (float, int, np.floating)):
        vals = (vals,)
    # Clamp near-zero values to avoid floating-point noise like 2.35e-28
    return " ".join(f"{(0.0 if abs(float(v)) < 1e-10 else float(v)):.{prec}g}" for v in vals)


# ═══════════════════════════════════════════════════════════════════════════
#  Onshape mate-type → URDF joint-type mapping
# ═══════════════════════════════════════════════════════════════════════════

MATE_TO_URDF = {
    "FASTENED":    "fixed",
    "REVOLUTE":    "revolute",
    "SLIDER":      "prismatic",
    "CYLINDRICAL": "revolute",      # approximation (ignore linear DOF)
    "PLANAR":      "planar",        # URDF supports planar
    "BALL":        "fixed",         # URDF has no ball joint → lock it
    "PIN_SLOT":    "revolute",      # approximation
    "PARALLEL":    "fixed",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Main converter
# ═══════════════════════════════════════════════════════════════════════════

class OnshapeToURDF:
    """Fetches an Onshape assembly and converts it to URDF."""

    def __init__(self, client: OnshapeClient, url: str, out_dir: Path):
        self.client = client
        self.did, self.wvm, self.wvmid, self.eid = parse_onshape_url(url)
        self.out = Path(out_dir)
        self.meshes = self.out / "meshes"

        # Populated during parsing
        self.parts: dict[str, dict] = {}      # instance_id → part info
        self.mates: list[dict] = []           # mate features
        self.link_frames: dict[str, np.ndarray] = {}  # instance_id → global 4x4

    # ─── API calls ────────────────────────────────────────────────────────

    def _fetch_assembly(self) -> dict:
        """GET assembly definition with mate features and connectors."""
        log.info("Fetching assembly definition ...")
        path = (f"/api/assemblies/d/{self.did}"
                f"/{self.wvm}/{self.wvmid}/e/{self.eid}")
        return self.client.get(path, {
            "includeMateFeatures": "true",
            "includeMateConnectors": "true",
        }).json()

    def _download_stl(self, part: dict) -> Path:
        """Download the STL mesh for a single part from its part studio."""
        stl_path = self.meshes / f"{part['link']}.stl"
        if stl_path.exists():
            return stl_path

        did = part.get("documentId", self.did)
        eid = part["elementId"]
        log.info("  Downloading %s ...", stl_path.name)

        path = f"/api/partstudios/d/{did}/{self.wvm}/{self.wvmid}/e/{eid}/stl"
        params = {
            "partIds": part["partId"],
            "mode": "binary",
            "units": "meter",
        }
        if part.get("configuration") and part["configuration"] != "default":
            params["configuration"] = part["configuration"]

        resp = self.client.get(path, params, accept="application/octet-stream")

        if len(resp.content) < 100:
            log.warning("  STL for %s seems too small (%d bytes) - may be empty",
                        part["link"], len(resp.content))

        stl_path.write_bytes(resp.content)
        return stl_path

    # ─── Assembly parsing ─────────────────────────────────────────────────

    def _sanitize_name(self, raw: str) -> str:
        """Convert an Onshape name to a valid URDF identifier."""
        return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()

    def _parse(self, data: dict):
        """Parse the raw assembly JSON into self.parts and self.mates."""
        root = data["rootAssembly"]

        # 1) Occurrence transforms  (keyed by path-tuple)
        occ_T: dict[tuple, np.ndarray] = {}
        occ_fixed: dict[tuple, bool] = {}
        for occ in root.get("occurrences", []):
            key = tuple(occ["path"])
            occ_T[key] = flat16_to_T(occ["transform"])
            occ_fixed[key] = occ.get("fixed", False)

        # 2) Instances → self.parts
        used_names: set[str] = set()
        all_instances = list(root.get("instances", []))

        # Also collect instances from sub-assemblies
        for sub in data.get("subAssemblies", []):
            all_instances.extend(sub.get("instances", []))

        for inst in all_instances:
            iid = inst["id"]
            if iid in self.parts:
                continue

            # Find the occurrence transform for this instance
            T = np.eye(4)
            for path_key, transform in occ_T.items():
                if path_key[-1] == iid:
                    T = transform
                    break

            fixed = False
            for path_key, is_fixed in occ_fixed.items():
                if path_key[-1] == iid:
                    fixed = is_fixed
                    break

            # Make a unique, safe link name
            name = self._sanitize_name(inst.get("name", iid))
            base = name
            n = 1
            while name in used_names:
                n += 1
                name = f"{base}_{n}"
            used_names.add(name)

            self.parts[iid] = {
                "id": iid,
                "link": name,
                "partId": inst.get("partId", ""),
                "elementId": inst.get("elementId", ""),
                "documentId": inst.get("documentId", self.did),
                "configuration": inst.get("configuration", "default"),
                "T": T,
                "fixed": fixed,
            }

        # 3) Mate features → self.mates
        for feat in root.get("features", []):
            if feat.get("featureType") != "mate":
                continue
            if feat.get("suppressed", False):
                continue

            fd = feat.get("featureData", {})
            entities = fd.get("matedEntities", [])
            if len(entities) != 2:
                continue

            pid = entities[0]["matedOccurrence"][-1]
            cid = entities[1]["matedOccurrence"][-1]

            if pid not in self.parts or cid not in self.parts:
                log.warning("Mate '%s' references unknown occurrence — skipping",
                            fd.get("name"))
                continue

            self.mates.append({
                "name": self._sanitize_name(fd.get("name", feat["id"])),
                "type": fd.get("mateType", "FASTENED"),
                "pid": pid,
                "cid": cid,
                "pCS": entities[0].get("matedCS", {}),
                "cCS": entities[1].get("matedCS", {}),
            })

        log.info("Found %d parts, %d mates", len(self.parts), len(self.mates))

    # ─── Kinematic tree ───────────────────────────────────────────────────

    def _find_root(self) -> str:
        """Choose the root link: prefer the fixed part, else a parent-only part."""
        # Prefer an explicitly fixed part
        for pid, p in self.parts.items():
            if p["fixed"]:
                return pid

        # Otherwise pick a part that only appears as a parent in mates
        parents = {m["pid"] for m in self.mates}
        children = {m["cid"] for m in self.mates}
        roots = parents - children
        if roots:
            return next(iter(roots))

        # Fallback: first part
        return next(iter(self.parts))

    def _build_tree(self, root_id: str) -> list[tuple[str, str, dict]]:
        """BFS spanning tree over the assembly graph.

        Returns [(parent_id, child_id, mate_dict), ...] in BFS order.
        Mates are treated as undirected edges; parent/child CS are swapped
        when traversing against the mate's original direction.
        """
        # Build undirected adjacency:  node → [(neighbour, mate, flipped), ...]
        adj: dict[str, list] = {pid: [] for pid in self.parts}
        for m in self.mates:
            adj[m["pid"]].append((m["cid"], m, False))
            adj[m["cid"]].append((m["pid"], m, True))

        visited = {root_id}
        queue = deque([root_id])
        edges: list[tuple[str, str, dict]] = []

        while queue:
            cur = queue.popleft()
            for nbr, mate, flipped in adj.get(cur, []):
                if nbr in visited:
                    continue
                visited.add(nbr)
                queue.append(nbr)

                if flipped:
                    # Swap parent/child coordinate systems
                    edges.append((cur, nbr, {
                        **mate,
                        "pid": cur, "cid": nbr,
                        "pCS": mate["cCS"], "cCS": mate["pCS"],
                    }))
                else:
                    edges.append((cur, nbr, mate))

        # Orphan parts (no mates) → attach fixed to root
        for pid in self.parts:
            if pid not in visited:
                log.warning("Part '%s' has no mate — attaching fixed to root",
                            self.parts[pid]["link"])
                identity_cs = {
                    "origin": [0, 0, 0],
                    "xAxis": [1, 0, 0], "yAxis": [0, 1, 0], "zAxis": [0, 0, 1],
                }
                edges.append((root_id, pid, {
                    "name": f"fixed_{self.parts[pid]['link']}",
                    "type": "FASTENED",
                    "pid": root_id, "cid": pid,
                    "pCS": identity_cs, "cCS": identity_cs,
                }))
                visited.add(pid)

        return edges

    # ─── URDF XML builders ────────────────────────────────────────────────

    @staticmethod
    def _xml_link(name: str, stl_filename: str, mesh_origin: np.ndarray) -> ET.Element:
        """Create a <link> with visual, collision, and placeholder inertial."""
        link = ET.Element("link", name=name)
        xyz, rpy = T_to_xyz_rpy(mesh_origin)

        # Inertial (placeholder values — replace with real mass properties)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia",
                      ixx="0.001", ixy="0", ixz="0",
                      iyy="0.001", iyz="0", izz="0.001")

        # Visual
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
        geom_v = ET.SubElement(visual, "geometry")
        ET.SubElement(geom_v, "mesh", filename=f"meshes/{stl_filename}")

        # Collision (same mesh)
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
        geom_c = ET.SubElement(collision, "geometry")
        ET.SubElement(geom_c, "mesh", filename=f"meshes/{stl_filename}")

        return link

    @staticmethod
    def _xml_joint(name: str, jtype: str,
                   parent_link: str, child_link: str,
                   origin_T: np.ndarray) -> ET.Element:
        """Create a <joint> element with axis and limits."""
        joint = ET.Element("joint", name=name, type=jtype)
        xyz, rpy = T_to_xyz_rpy(origin_T)

        ET.SubElement(joint, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=child_link)

        if jtype == "revolute":
            ET.SubElement(joint, "axis", xyz="0 0 1")
            ET.SubElement(joint, "limit",
                          effort="100", velocity="3.14159",
                          lower="-3.14159", upper="3.14159")
        elif jtype == "prismatic":
            ET.SubElement(joint, "axis", xyz="0 0 1")
            ET.SubElement(joint, "limit",
                          effort="100", velocity="1.0",
                          lower="-1.0", upper="1.0")
        elif jtype == "continuous":
            ET.SubElement(joint, "axis", xyz="0 0 1")
        # 'fixed' joints need no axis or limits

        return joint

    # ─── Main pipeline ────────────────────────────────────────────────────

    def run(self):
        """Execute the full conversion: fetch → parse → download → URDF."""
        self.meshes.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Fetch assembly definition from Onshape ────────────────
        data = self._fetch_assembly()
        debug_path = self.out / "assembly_raw.json"
        debug_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Raw API response saved to %s", debug_path)

        # ── Step 2: Parse into parts and mates ────────────────────────────
        self._parse(data)
        if not self.parts:
            sys.exit("ERROR: No parts found in the assembly.")

        # ── Step 3: Build kinematic tree (BFS from root) ──────────────────
        root_id = self._find_root()
        edges = self._build_tree(root_id)
        log.info("Root link: '%s'", self.parts[root_id]["link"])

        # ── Step 4: Compute link frames ───────────────────────────────────
        #
        # Base link frame = identity (assembly origin).
        # Each child link frame = mate connector on the parent side (in global).
        # This means the Z-axis of the mate connector becomes the joint axis.
        #
        self.link_frames[root_id] = np.eye(4)

        for parent_id, child_id, mate in edges:
            parent_occ_T = self.parts[parent_id]["T"]
            mate_conn_local = matedCS_to_T(mate["pCS"])
            mate_conn_global = parent_occ_T @ mate_conn_local
            self.link_frames[child_id] = mate_conn_global

        # ── Step 5: Download STL meshes ───────────────────────────────────
        log.info("Downloading STL meshes ...")
        stl_cache: dict[tuple, Path] = {}  # (did, eid, partId) → first path

        for part in self.parts.values():
            key = (part["documentId"], part["elementId"], part["partId"])
            target = self.meshes / f"{part['link']}.stl"

            if key in stl_cache and stl_cache[key] != target:
                # Same geometry, different instance — just copy
                shutil.copy2(stl_cache[key], target)
                log.info("  Copied %s (same part)", target.name)
            else:
                self._download_stl(part)
                stl_cache[key] = target

        # ── Step 6: Build URDF XML ────────────────────────────────────────
        robot = ET.Element("robot", name="robot")

        # Root link
        root_part = self.parts[root_id]
        root_mesh_T = np.linalg.inv(self.link_frames[root_id]) @ root_part["T"]
        robot.append(self._xml_link(
            root_part["link"], f"{root_part['link']}.stl", root_mesh_T,
        ))

        # Joints + child links (in BFS order)
        for parent_id, child_id, mate in edges:
            parent_part = self.parts[parent_id]
            child_part = self.parts[child_id]

            # Joint type
            jtype = MATE_TO_URDF.get(mate["type"], "fixed")

            # Joint origin = transform from parent link frame → child link frame
            joint_T = np.linalg.inv(self.link_frames[parent_id]) @ self.link_frames[child_id]

            # Child mesh origin = transform from child link frame → occurrence frame
            child_mesh_T = np.linalg.inv(self.link_frames[child_id]) @ child_part["T"]

            robot.append(self._xml_joint(
                mate["name"], jtype,
                parent_part["link"], child_part["link"],
                joint_T,
            ))
            robot.append(self._xml_link(
                child_part["link"], f"{child_part['link']}.stl", child_mesh_T,
            ))

        # ── Step 7: Write URDF file ──────────────────────────────────────
        urdf_path = self.out / "robot.urdf"
        raw_xml = ET.tostring(robot, encoding="unicode")
        pretty = minidom.parseString(raw_xml).toprettyxml(indent="  ")
        # minidom adds its own <?xml?> — normalise it
        lines = pretty.splitlines(keepends=True)
        if lines and lines[0].startswith("<?xml"):
            lines[0] = '<?xml version="1.0" ?>\n'
        urdf_path.write_text("".join(lines), encoding="utf-8")

        log.info("URDF written to %s", urdf_path)
        log.info("Meshes in      %s/", self.meshes)
        log.info("Done! %d links, %d joints",
                 len(self.parts), len(edges))


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    here = Path(__file__).resolve().parent
    load_dotenv(here / ".env")

    access_key = os.getenv("ONSHAPE_ACCESS_KEY", "")
    secret_key = os.getenv("ONSHAPE_SECRET_KEY", "")
    if not access_key or not secret_key:
        sys.exit("ERROR: Set ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY in .env")

    base_url = os.getenv("ONSHAPE_API", "https://cad.onshape.com")

    # URL from CLI argument or config.json fallback
    url = None
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        cfg_path = here / "my-robot" / "config.json"
        if cfg_path.exists():
            # Strip // comments (only full-line comments, to avoid breaking URLs)
            raw_lines = cfg_path.read_text(encoding="utf-8").splitlines()
            cleaned = [ln for ln in raw_lines if not re.match(r"^\s*//", ln)]
            cfg = json.loads("\n".join(cleaned))
            url = cfg.get("url")

    if not url:
        sys.exit("Usage: python my_onshape-to-robot.py <ONSHAPE_ASSEMBLY_URL>")

    log.info("Document URL: %s", url)
    out_dir = here / "urdf_output"

    client = OnshapeClient(base_url, access_key, secret_key)
    converter = OnshapeToURDF(client, url, out_dir)
    converter.run()


if __name__ == "__main__":
    main()
