import io
import os
import re
import json
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import trimesh
from scipy.spatial import cKDTree

# 可选：Quadric decimation
try:
    import pyfqmr
    HAS_PYFQMR = True
except Exception:
    HAS_PYFQMR = False

# ========== 基本设定 ==========
st.set_page_config(page_title="Lung 3DBody Pro – Streamlit", layout="wide")

DEFAULT_OPACITY = 1.0
DEFAULT_ROUGHNESS_HINT = 0.55  # Plotly 里只做“视觉近似”
CAMERA_PRESETS = {
    "ISO":      dict(eye=dict(x=1.6, y=1.6, z=1.2)),
    "Axial":    dict(eye=dict(x=0.0, y=0.0, z=2.2)),
    "Coronal":  dict(eye=dict(x=0.0, y=2.2, z=0.0)),
    "Sagittal": dict(eye=dict(x=2.2, y=0.0, z=0.0)),
}

COLOR_RULES = [
    (r'artery|pulmonary[_\s-]*artery|pa', (220, 30, 30)),     # 动脉 红
    (r'vein|pulmonary[_\s-]*vein|pv',     (30, 80, 220)),     # 静脉 蓝
    (r'bronch|airway|trachea',            (240, 200, 40)),    # 支气管 黄
    (r'lung|lobe',                        (180, 220, 220)),   # 肺整体 淡青
]

SEGMENT_PALETTE = [
    (116, 196, 118), (35, 139, 69), (65, 182, 196), (44, 127, 184),
    (251, 128, 114), (227, 26, 28), (141, 160, 203), (106, 61, 154),
    (255, 237, 111), (254, 178, 76), (240, 59, 32), (166, 86, 40)
]

# 左右肺段调色：左右独立色盘，支持 L/R/Left/Right + S1..S10
LEFT_SEG_COLORS = {
    1:(255, 179, 186), 2:(255, 223, 186), 3:(255, 255, 186), 4:(186, 255, 201), 5:(186, 225, 255),
    6:(204, 204, 255), 7:(255, 204, 229), 8:(255, 204, 153), 9:(204, 255, 229), 10:(229, 204, 255)
}
RIGHT_SEG_COLORS = {
    1:(255, 153, 153), 2:(255, 204, 153), 3:(255, 255, 153), 4:(153, 255, 153), 5:(153, 204, 255),
    6:(173, 173, 255), 7:(255, 173, 214), 8:(255, 204, 102), 9:(153, 255, 204), 10:(214, 173, 255)
}

# ========== 工具函数 ==========
def nice_name(filename: str) -> str:
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]

def detect_group(name: str) -> str:
    n = name.lower()
    if re.search(r'artery|pulmonary[_\s-]*artery|pa', n): return 'Arteries'
    if re.search(r'vein|pulmonary[_\s-]*vein|pv', n):     return 'Veins'
    if re.search(r'bronch|airway|trachea', n):            return 'Bronchi'
    # 肺段判定包括 S1~S10 命名
    if re.search(r'\b(l|r|left|right)[-_ ]?s(10|[1-9])\b', n) or re.search(r'segment|seg', n):
        return 'Segments'
    return 'Others'

def suggest_color_by_rules(name: str, group: str, seg_idx: int) -> Tuple[int,int,int]:
    # 先按左右+S号分配
    m = re.search(r'\b(l|r|left|right)[-_ ]?s(10|[1-9])\b', name.lower())
    if m:
        side = m.group(1)
        sidx = int(m.group(2))
        if side in ('l','left'):
            return LEFT_SEG_COLORS.get(sidx, SEGMENT_PALETTE[(seg_idx)%len(SEGMENT_PALETTE)])
        else:
            return RIGHT_SEG_COLORS.get(sidx, SEGMENT_PALETTE[(seg_idx)%len(SEGMENT_PALETTE)])

    # 再按通用规则
    for pat, col in COLOR_RULES:
        if re.search(pat, name.lower()):
            return col
    # 肺段回退调色盘
    if group == 'Segments':
        return SEGMENT_PALETTE[seg_idx % len(SEGMENT_PALETTE)]
    # 其他默认
    return (200, 200, 220)

def rgb_hex(rgb: Tuple[int,int,int]) -> str:
    return '#%02x%02x%02x' % rgb

# ========== 数据结构 ==========
@dataclass
class Part:
    name: str
    group: str
    mesh: trimesh.Trimesh
    color: Tuple[int,int,int]
    visible: bool = True
    opacity: float = DEFAULT_OPACITY
    base_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0,3)))
    base_faces: np.ndarray = field(default_factory=lambda: np.zeros((0,3), dtype=np.int32))
    # 简化参数
    simplify_ratio: float = 1.0   # 1.0 = 不简化
    simplify_target_faces: Optional[int] = None

    def ensure_backup(self):
        if self.base_vertices.size == 0:
            self.base_vertices = self.mesh.vertices.copy()
        if self.base_faces.size == 0:
            self.base_faces = self.mesh.faces.copy()

# ========== 文件加载 ==========
def load_parts_from_upload(files) -> List[Part]:
    parts: List[Part] = []
    seg_idx = 0

    def read_one(byts: bytes, fname: str):
        nonlocal seg_idx
        m = trimesh.load(io.BytesIO(byts), file_type='stl')
        if not isinstance(m, trimesh.Trimesh):
            m = trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
        m.remove_unreferenced_vertices()
        m.remove_degenerate_faces()
        m.merge_vertices()
        name = nice_name(fname)
        group = detect_group(name)
        color = suggest_color_by_rules(name, group, seg_idx)
        if group == 'Segments':
            seg_idx += 1
        p = Part(name=name, group=group, mesh=m, color=color)
        p.ensure_backup()
        parts.append(p)

    for up in files:
        fn = up.name
        if fn.lower().endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(up.read()))
            for n in z.namelist():
                if n.lower().endswith(".stl"):
                    read_one(z.read(n), n)
        elif fn.lower().endswith(".stl"):
            read_one(up.read(), fn)
    return parts

# ========== 网格简化 ==========
def simplify_mesh_trimesh(mesh: trimesh.Trimesh, target_faces: Optional[int]=None, ratio: float=1.0) -> trimesh.Trimesh:
    if not HAS_PYFQMR:
        return mesh  # 无库则原样返回
    # 目标面数：优先 target_faces，否则按比例
    faces = mesh.faces.shape[0]
    if target_faces is None:
        target_faces = max(10, int(faces * ratio))
    target_faces = min(faces, max(10, target_faces))
    Q = pyfqmr.Simplify()
    Q.setMesh(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int64))
    Q.simplify_mesh(target_faces, 7, True)  # 迭代 7，保拓扑
    v, f = Q.getMesh()
    out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return out

# ========== 裁剪/爆炸（这里主打测距和标签，裁剪保持简单） ==========
def clip_by_plane_inplace(part: Part, normal: np.ndarray, d: float):
    n = normal / (np.linalg.norm(normal) + 1e-9)
    V = part.mesh.vertices
    mask_v = (V @ n - d) >= 0.0
    F = part.mesh.faces
    keep = mask_v[F].all(axis=1)
    if keep.sum() < 1:
        part.mesh.vertices = np.zeros((0, 3))
        part.mesh.faces = np.zeros((0, 3), dtype=int)
        return
    new_faces = F[keep]
    used = np.unique(new_faces)
    idx_map = -np.ones(len(V), dtype=int)
    idx_map[used] = np.arange(len(used))
    part.mesh.vertices = V[used]
    part.mesh.faces = idx_map[new_faces]

def apply_explode(parts: List[Part], strength: float):
    if not parts: return
    bounds = np.array([p.mesh.bounds for p in parts])
    mins = bounds[:, 0::2].min(axis=0)
    maxs = bounds[:, 1::2].max(axis=0)
    center = (mins + maxs) / 2.0
    diag = np.linalg.norm(maxs - mins)
    scale = 0.15 * diag * strength
    for p in parts:
        V0 = p.base_vertices
        gcenter = p.mesh.center_mass if p.mesh.is_volume else p.mesh.centroid
        v = gcenter - center
        dirv = v / (np.linalg.norm(v) + 1e-9)
        p.mesh.vertices = V0 + dirv * scale

def restore_geometry(parts: List[Part]):
    for p in parts:
        p.mesh.vertices = p.base_vertices.copy()
        p.mesh.faces = p.base_faces.copy()

# ========== Plotly 组装 ==========
def mesh_trace(part: Part):
    if part.mesh.vertices.size == 0 or part.mesh.faces.size == 0 or not part.visible:
        return None
    x, y, z = part.mesh.vertices.T
    i, j, k = part.mesh.faces.T
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        name=part.name,
        opacity=part.opacity,
        color=f"rgb({part.color[0]},{part.color[1]},{part.color[2]})",
        flatshading=False,
        lighting=dict(ambient=0.35, diffuse=0.7, specular=0.22, roughness=DEFAULT_ROUGHNESS_HINT),
        lightposition=dict(x=2000, y=2000, z=3000),
        hovertemplate=f"{part.name}<extra></extra>",
        showscale=False
    )

def label_trace(points: np.ndarray, texts: List[str]):
    if points.size == 0: return None
    return go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode="markers+text",
        marker=dict(size=6),
        text=texts,
        textposition="top center",
        hoverinfo="text"
    )

def build_figure(parts: List[Part], annotations: List[dict], camera_key: str):
    data = []
    for p in parts:
        tr = mesh_trace(p)
        if tr: data.append(tr)
    # 注释点
    if annotations:
        pts = np.array([a["point"] for a in annotations], dtype=float)
        txt = [a["text"] for a in annotations]
        tr_lab = label_trace(pts, txt)
        if tr_lab: data.append(tr_lab)

    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   aspectmode='data'),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    if camera_key in CAMERA_PRESETS:
        fig.update_scenes(camera=CAMERA_PRESETS[camera_key])
    fig.update_layout(transition_duration=300)
    return fig

# ========== 侧边栏：上传与全局控件 ==========
with st.sidebar:
    st.header("1) 上传 STL / ZIP")
    uploads = st.file_uploader("拖入多个 STL 或一个 ZIP（包含 STL）", type=["stl","zip"], accept_multiple_files=True)

    st.header("2) 相机与裁剪")
    camera_key = st.radio("相机预设", list(CAMERA_PRESETS.keys()), index=0)
    clip_on = st.checkbox("启用平面裁剪", value=False)
    nx = st.slider("nx", -1.0, 1.0, 0.0, 0.01)
    ny = st.slider("ny", -1.0, 1.0, 0.0, 0.01)
    nz = st.slider("nz", -1.0, 1.0, 1.0, 0.01)
    d_clip = st.slider("d（位置）", -200.0, 200.0, 0.0, 0.5)

    st.header("3) 爆炸视图")
    explode = st.slider("强度", 0.0, 1.0, 0.0, 0.01)

# ========== 状态 ==========
if "parts" not in st.session_state:
    st.session_state.parts: List[Part] = []
if "annotations" not in st.session_state:
    st.session_state.annotations: List[dict] = []  # {text, point:[x,y,z]}
if "pick_buffer" not in st.session_state:
    st.session_state.pick_buffer: List[Tuple[int, int]] = []  # [(trace_idx, vertex_idx)]
if "last_fig_tracemap" not in st.session_state:
    st.session_state.last_fig_tracemap: List[str] = []  # trace顺序 -> part.name（最后一个为labels）

# 载入
if uploads:
    st.session_state.parts = load_parts_from_upload(uploads)
    st.session_state.annotations = []
    st.session_state.pick_buffer = []

parts = st.session_state.parts
annotations = st.session_state.annotations

colL, colR = st.columns([0.32, 0.68])

# ========== 左：对象面板 ==========
with colL:
    st.subheader("对象与外观 | 预设调色已应用（左右 + S1~S10）")
    if not parts:
        st.info("请先上传 STL / ZIP")
    else:
        groups = ["Segments","Arteries","Veins","Bronchi","Others"]
        for g in groups:
            gp = [p for p in parts if p.group == g]
            if not gp: continue
            with st.expander(f"{g}（{len(gp)}）", expanded=True):
                gcol1, gcol2 = st.columns(2)
                show_g = gcol1.checkbox(f"显示 {g}", value=True, key=f"show_{g}")
                g_opacity = gcol2.slider(f"{g} 不透明度", 0.0, 1.0, 1.0, 0.01, key=f"op_{g}")
                for p in gp:
                    p.visible = show_g
                    p.opacity = g_opacity
                for p in gp:
                    c1, c2 = st.columns([0.55, 0.45])
                    with c1:
                        p.visible = st.checkbox(p.name, value=p.visible, key=f"vis_{p.name}")
                    with c2:
                        p.opacity = st.slider(f"α: {p.name}", 0.0, 1.0, p.opacity, 0.01, key=f"op_{p.name}")
                    new_hex = st.color_picker(f"颜色: {p.name}", rgb_hex(p.color), key=f"col_{p.name}")
                    p.color = tuple(int(new_hex[i:i+2],16) for i in (1,3,5))

        st.markdown("---")
        st.subheader("网格简化（Quadric Decimation）")
        if not HAS_PYFQMR:
            st.caption("未检测到 pyfqmr，简化功能停用（requirements 已包含，部署环境会自动安装）。")
        for p in parts:
            with st.expander(f"Simplify – {p.name}"):
                faces = int(p.mesh.faces.shape[0])
                st.caption(f"当前面数：{faces}")
                r = st.slider(f"保留比例（{p.name}）", 0.05, 1.0, 1.0, 0.05, key=f"ratio_{p.name}")
                p.simplify_ratio = r
                tgt = st.number_input(f"或指定目标面数（{p.name}）", min_value=10, max_value=max(10, faces), value=min(faces, max(10, faces)))
                apply_btn = st.button(f"应用简化：{p.name}", key=f"simp_{p.name}")
                if apply_btn and HAS_PYFQMR:
                    p.mesh = simplify_mesh_trimesh(p.mesh, target_faces=int(tgt), ratio=float(r))
                    p.ensure_backup()  # 更新备份为新的基准
                    st.success(f"已简化 {p.name}")

        st.markdown("---")
        st.subheader("导出")
        # GLB/GLTF 场景导出（保留颜色）
        if st.button("导出 GLB（带颜色，合并场景）"):
            scene = trimesh.Scene()
            for p in parts:
                if not p.visible or p.mesh.vertices.size==0 or p.mesh.faces.size==0:
                    continue
                # 赋单色到每个面
                color_rgba = np.array([[*p.color, 255]], dtype=np.uint8)
                fc = np.repeat(color_rgba, repeats=len(p.mesh.faces), axis=0)
                gm = p.mesh.copy()
                gm.visual.face_colors = fc
                scene.add_geometry(gm, node_name=p.name)
            glb_bytes = scene.export(file_type='glb')
            st.download_button("下载 scene.glb", data=glb_bytes, file_name="scene.glb")

        # 带标签的 HTML（离线可交互）
        if st.button("导出带标签的 HTML"):
            fig = build_figure(parts, annotations, camera_key)
            html = fig.to_html(full_html=True, include_plotlyjs='cdn')
            st.download_button("下载 labeled_scene.html", data=html.encode("utf-8"), file_name="labeled_scene.html")

# ========== 右：3D 视图 + 交互（拾取/测距/标签） ==========
with colR:
    st.subheader("三维视图（可点击拾取顶点）")

    if not parts:
        st.stop()

    # 每次绘制前恢复原始几何（避免多次裁剪叠加）
    restore_geometry(parts)

    # 爆炸
    if explode > 0:
        apply_explode(parts, explode)

    # 裁剪（可选）
    if clip_on:
        normal = np.array([nx, ny, nz], dtype=float)
        for p in parts:
            clip_by_plane_inplace(p, normal, d_clip)

    # 绘图
    fig = build_figure(parts, annotations, camera_key)

    # 建立 trace -> part 映射（注：最后一个可能是 labels）
    tracemap = []
    for p in parts:
        if p.visible and p.mesh.vertices.size>0 and p.mesh.faces.size>0:
            tracemap.append(p.name)
    if annotations:
        tracemap.append("__labels__")
    st.session_state.last_fig_tracemap = tracemap

    # 使用 plotly_events 捕获点击（返回点对应的 trace/vertex）
    clicked = plotly_events(
        fig, click_event=True, hover_event=False, select_event=False,
        override_height=700, override_width="100%"
    )

    # 顶点拾取缓存（两点测距用）
    if clicked:
        ev = clicked[0]
        trace_idx = ev.get("curveNumber", None)
        pt_idx = ev.get("pointNumber", None)
        if trace_idx is not None and pt_idx is not None:
            if trace_idx < len(tracemap) and tracemap[trace_idx] != "__labels__":
                st.session_state.pick_buffer.append((trace_idx, pt_idx))
                st.success(f"已拾取：{tracemap[trace_idx]} 的顶点 #{pt_idx}")

    # --- 专业测量 ---
    st.markdown("### 测量")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.caption("A) 顶点-顶点测距（点击 3D 视图，从可见对象上依次拾取两点）")
        if st.button("计算两点距离"):
            buf = st.session_state.pick_buffer
            if len(buf) < 2:
                st.warning("请先点击拾取两个顶点。")
            else:
                (t1, v1), (t2, v2) = buf[-2], buf[-1]
                name1, name2 = tracemap[t1], tracemap[t2]
                p1 = next(p for p in parts if p.name==name1)
                p2 = next(p for p in parts if p.name==name2)
                A = p1.mesh.vertices[v1]
                B = p2.mesh.vertices[v2]
                dist = float(np.linalg.norm(A-B))
                st.success(f"点-点距离 ≈ **{dist:.2f}**（与 STL 单位一致，通常为 mm）")

    with mcol2:
        st.caption("B) 表面最近点距离（两对象所有顶点间的最近距离，高效 KDTree）")
        names = [p.name for p in parts if p.visible and p.mesh.vertices.size>0]
        if len(names) >= 2:
            a = st.selectbox("对象 A", names, index=0, key="near_a")
            b = st.selectbox("对象 B", names, index=1, key="near_b")
            go_btn = st.button("计算表面最近距离")
            if go_btn:
                pa = next(p for p in parts if p.name==a)
                pb = next(p for p in parts if p.name==b)
                VA = pa.mesh.vertices
                VB = pb.mesh.vertices
                # 若顶点很多，可采样加速
                MAX_SAMP = 150000
                if len(VA) > MAX_SAMP:
                    VA = VA[np.random.choice(len(VA), MAX_SAMP, replace=False)]
                if len(VB) > MAX_SAMP:
                    VB = VB[np.random.choice(len(VB), MAX_SAMP, replace=False)]
                tree = cKDTree(VB)
                dists, idxs = tree.query(VA, k=1)
                mind = float(dists.min())
                st.success(f"表面最近点距离 ≈ **{mind:.2f}**（单位同 STL）")
        else:
            st.caption("至少两个可见对象才能计算。")

    # --- 标签 / 注释 ---
    st.markdown("### 标签 / 注释")
    st.caption("在 3D 视图中点击一个顶点后，在下面输入标签文本并添加。")
    label_text = st.text_input("标签文本", value="")
    add_lab = st.button("添加标签（取最近一次拾取的顶点）")
    if add_lab:
        buf = st.session_state.pick_buffer
        if not buf:
            st.warning("请先在 3D 视图点击一个顶点。")
        else:
            t, v = buf[-1]
            nm = tracemap[t]
            if nm == "__labels__":
                st.warning("请在模型上拾取，而不是标签图层。")
            else:
                part = next(p for p in parts if p.name==nm)
                pt = part.mesh.vertices[v].tolist()
                txt = label_text if label_text.strip() else nm
                annotations.append({"text": txt, "point": pt})
                st.success(f"已添加标签：{txt}")

    if st.button("清空所有标签"):
        annotations.clear()
        st.info("已清空标签。")

    # --- 截图（PNG） ---
    if st.button("导出当前视图 PNG"):
        try:
            # 重新生成一次图，防止对象变化
            fig2 = build_figure(parts, annotations, camera_key)
            png = fig2.to_image(format="png", width=1600, height=1200, scale=2)
            st.download_button("下载 screenshot.png", data=png, file_name="screenshot.png")
        except Exception as e:
            st.warning(f"导出失败：{e}（也可用图右上角菜单保存）")
