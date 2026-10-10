"""Construction-site terrain and lighting for an externally supplied MuJoCo model.

This module adds only environment geometry. Robot bodies, joints, actuators, tool
geometry and mesh assets must be supplied separately by the caller.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np

from .terrain import (
    ELEV, PAD, Z_BASE, GridSpec, SiteLayout, initial_terrain, outer_terrain,
    soil_texture, save_png,
)


def aligned_surround_half(grid: GridSpec, approx=200.0):
    """Surround half-size whose texture tiles align with the fine soil grid."""
    tile = 2 * grid.half
    n = round((approx + grid.xmin) / tile)
    half = n * tile - grid.xmin
    if abs(grid.xmin - grid.ymin) > 1e-9:
        raise ValueError('grid must be centred on the diagonal for a square aligned surround')
    return float(half)


def aligned_surround(grid: GridSpec, approx=200.0, max_cell=3.2):
    """Aligned surround (half-size, vertex count), in metres and grid vertices."""
    half = aligned_surround_half(grid, approx)
    tile = 2 * grid.half
    for k in range(1, 200):
        cell = tile / k
        cells = 2 * half / cell
        if cell <= max_cell and abs(cells - round(cells)) < 1e-9:
            return half, int(round(cells)) + 1
    raise ValueError('no aligned surround resolution found')


def surround_mesh_arrays(grid: GridSpec, half, n, outer):
    """Far terrain mesh with a square opening over the deformable soil grid."""
    outer = np.asarray(outer, float)
    if outer.shape != (n, n):
        raise ValueError('outer must have shape (n, n)')
    x = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x, x)
    tile = 2 * grid.half
    vertices = np.c_[X.ravel(), Y.ravel(), outer.ravel()]
    x0, x1 = grid.xmin, grid.xmin + tile
    y0, y1 = grid.ymin, grid.ymin + tile
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            if x[j] >= x0 - 1e-6 and x[j + 1] <= x1 + 1e-6 and x[i] >= y0 - 1e-6 and x[i + 1] <= y1 + 1e-6:
                continue
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j
            faces += [(a, b, c), (a, c, d)]
    # Heightfield texture rows run towards -y; continue the same convention.
    uv = np.c_[(X.ravel() - x0) / tile, (y1 - Y.ravel()) / tile]
    return vertices, np.asarray(faces, np.int64), uv


def build_environment_assets(asset_dir, grid: GridSpec, layout: SiteLayout, *,
                             tex_size=2048, surround_n=129, surround_half=200.0):
    """Generate soil textures and initial terrain arrays; no robot assets are created.

    Returns ``h, loose, ground, outer``. Runtime files are written to ``asset_dir``.
    """
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    h, loose, ground = initial_terrain(grid, layout)
    outer = outer_terrain(surround_n, surround_half, inner_half=grid.half, inner_center=grid.center)
    np.savez_compressed(asset_dir / 'terrain_initial.npz', h=h, loose=loose,
                        ground=ground, outer=outer, grid=json.dumps(asdict(grid)))
    save_png(soil_texture(tex_size, fresh=False, meters=2 * grid.half), asset_dir / 'soil_base.png')
    save_png(soil_texture(tex_size, fresh=True, meters=2 * grid.half), asset_dir / 'soil_fresh.png')
    return h, loose, ground, outer


def _element(root, tag):
    child = root.find(tag)
    return ET.SubElement(root, tag) if child is None else child


def _numbers(values):
    return ' '.join(format(float(v), '.8g') for v in np.ravel(values))


def _preserve_model_resources(root, source_dir):
    """Retain external resource resolution when writing MJCF to another folder."""
    compiler = _element(root, 'compiler')
    assetdir = compiler.get('assetdir')
    for attr in ('meshdir', 'texturedir'):
        resource_dir = Path(compiler.get(attr, assetdir if assetdir is not None else '.'))
        if not resource_dir.is_absolute():
            compiler.set(attr, str((source_dir / resource_dir).resolve()))
    for element in root.iter():
        filename = element.get('file')
        if filename and element.tag not in ('mesh', 'texture', 'hfield', 'skin') and not Path(filename).is_absolute():
            element.set('file', str((source_dir / filename).resolve()))


def add_environment(base_xml, out_xml, asset_dir, grid: GridSpec, *,
                    soil_base=Z_BASE, elevation_range=ELEV,
                    working_pad=PAD, pad_contype=1, pad_conaffinity=1,
                    remove_ground_names=('ground',), surround_half=200.0,
                    surround_n=129, surround_outer=None):
    """Add terrain, a working pad, sky and lighting to external base MJCF.

    The soil heightfield is visual geometry: apply ``SoilModel`` wrenches to the
    caller's tool. The optional working pad provides ordinary MuJoCo contact;
    set its contact bitmasks for the supplied model or pass ``working_pad=None``.
    Existing robot bodies, joints, actuators and visuals are preserved.

    Model resource paths remain relative to ``base_xml`` unless its compiler
    already specifies ``meshdir``/``texturedir``. Ground planes named by
    ``remove_ground_names`` are replaced. Environment names must be unused.
    """
    base_xml, out_xml = Path(base_xml).resolve(), Path(out_xml).resolve()
    asset_dir = Path(asset_dir).resolve()
    if not np.isfinite(soil_base) or not np.isfinite(elevation_range) or elevation_range <= 0:
        raise ValueError('soil_base must be finite and elevation_range positive')
    tree = ET.parse(base_xml)
    root = tree.getroot()
    if root.tag != 'mujoco':
        raise ValueError('base_xml must be an MJCF mujoco document')
    asset, world = _element(root, 'asset'), _element(root, 'worldbody')
    _preserve_model_resources(root, base_xml.parent)
    for geom in list(world.findall('geom')):
        if geom.get('name') in remove_ground_names:
            world.remove(geom)
    for child in list(asset):
        if child.tag == 'texture' and child.get('type') == 'skybox':
            asset.remove(child)
    names = {'soil_tex', 'soil_mat', 'fresh_tex', 'soil_content_mat', 'soil',
             'surround', 'surround_mat', 'environment_sun', 'environment_fill'}
    if working_pad is not None:
        names.add('working_pad')
    if surround_outer is not None:
        names.add('surround_mesh')
    conflicts = sorted({e.get('name') for e in root.iter()} & names)
    if conflicts:
        raise ValueError('environment names already exist: ' + ', '.join(conflicts))
    ET.SubElement(asset, 'texture', type='skybox', builtin='gradient', rgb1='.42 .58 .80', rgb2='.86 .88 .90', width='512', height='512')
    for name, filename in (('soil_tex', 'soil_base.png'), ('fresh_tex', 'soil_fresh.png')):
        file = asset_dir / filename
        if not file.is_file():
            raise FileNotFoundError(file)
        ET.SubElement(asset, 'texture', name=name, type='2d', file=str(file))
    ET.SubElement(asset, 'material', name='soil_mat', texture='soil_tex', texrepeat='1 1', texuniform='false', specular='.05', shininess='.05')
    ET.SubElement(asset, 'material', name='soil_content_mat', texture='fresh_tex', texrepeat='1 1', texuniform='false', specular='.02', shininess='.02')
    ET.SubElement(asset, 'material', name='surround_mat', texture='soil_tex', texrepeat='1 1', texuniform='false', specular='.05', shininess='.05')
    ET.SubElement(asset, 'hfield', name='soil', nrow=str(grid.n), ncol=str(grid.n), size=_numbers([grid.half, grid.half, elevation_range, .4]))
    ET.SubElement(world, 'geom', name='soil', type='hfield', hfield='soil',
                  pos=_numbers([*grid.center, soil_base]), material='soil_mat', contype='0', conaffinity='0', group='0')
    if working_pad is not None:
        x0, x1, half_y = working_pad
        if x1 <= x0 or half_y <= 0:
            raise ValueError('working_pad requires x1 > x0 and a positive y half-width')
        ET.SubElement(world, 'geom', name='working_pad', type='box',
                      pos=_numbers([(x0 + x1) / 2, 0, -.5]), size=_numbers([(x1 - x0) / 2, half_y, .5]),
                      contype=str(pad_contype), conaffinity=str(pad_conaffinity), condim='3',
                      friction='1.0 .003 .008', group='3', rgba='.5 .5 .5 1')
    if surround_outer is None:
        ET.SubElement(asset, 'hfield', name='surround', nrow=str(surround_n), ncol=str(surround_n), size=_numbers([surround_half, surround_half, 14, .5]))
        ET.SubElement(world, 'geom', name='surround', type='hfield', hfield='surround', pos='0 0 -1.5', material='surround_mat', contype='0', conaffinity='0', group='0')
    else:
        vertices, faces, uv = surround_mesh_arrays(grid, surround_half, surround_n, surround_outer)
        ET.SubElement(asset, 'mesh', name='surround_mesh', inertia='shell', vertex=_numbers(vertices),
                      texcoord=_numbers(uv), face=' '.join(str(int(i)) for i in faces.ravel()))
        ET.SubElement(world, 'geom', name='surround', type='mesh', mesh='surround_mesh', material='surround_mat', contype='0', conaffinity='0', mass='0', group='0')
    ET.SubElement(world, 'light', name='environment_sun', directional='true', pos='-20 -30 40', dir='.42 .55 -.72', diffuse='.95 .90 .80', specular='.35 .33 .30', castshadow='true')
    ET.SubElement(world, 'light', name='environment_fill', directional='true', pos='20 30 20', dir='-.5 -.6 -.6', diffuse='.20 .23 .28', specular='0 0 0', castshadow='false')
    visual = _element(root, 'visual')
    _element(visual, 'global').set('offwidth', '1920')
    _element(visual, 'global').set('offheight', '1080')
    _element(visual, 'quality').set('shadowsize', '8192')
    _element(visual, 'quality').set('offsamples', '8')
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space='  ')
    tree.write(out_xml, encoding='utf-8', xml_declaration=True)
    return out_xml


def add_carried_soil_mesh(xml_path, out_xml, body_name, tool_geometry):
    """Add a render mesh to a caller-selected body using caller-supplied geometry.

    Requires ``soil_content_mat`` added by ``add_environment``. Mesh vertices can
    be updated at runtime using ``content_mesh.vertices_world``.
    """
    from . import content_mesh
    xml_path = Path(xml_path).resolve()
    out_xml = Path(out_xml).resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    _preserve_model_resources(root, xml_path.parent)
    bodies = [b for b in root.iter('body') if b.get('name') == body_name]
    if len(bodies) != 1:
        raise ValueError('body_name must identify exactly one external model body')
    asset = _element(root, 'asset')
    if asset.find("material[@name='soil_content_mat']") is None:
        raise ValueError('soil_content_mat is missing; add the environment first')
    if any(e.get('name') == 'soil_content' for e in root.iter()):
        raise ValueError('soil_content already exists')
    ET.SubElement(asset, 'mesh', name='soil_content', vertex=_numbers(content_mesh.initial_local(tool_geometry)),
                  texcoord=_numbers(content_mesh.texcoords()), face=' '.join(str(int(i)) for i in content_mesh.FACES.ravel()))
    ET.SubElement(bodies[0], 'geom', name='soil_content', type='mesh', mesh='soil_content',
                  material='soil_content_mat', contype='0', conaffinity='0', mass='0', group='1')
    out_xml = Path(out_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space='  ')
    tree.write(out_xml, encoding='utf-8', xml_declaration=True)
    return out_xml


def load_terrain_into_model(model, h, outer=None, *, soil_base=Z_BASE,
                            elevation_range=ELEV, soil_name='soil',
                            surround_name='surround', surround_base=-1.5):
    """Write metre-valued terrain elevations into named MuJoCo heightfields."""
    import mujoco
    h = np.asarray(h, float)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, soil_name)
    if sid < 0:
        raise ValueError('soil heightfield is missing: ' + soil_name)
    adr, nr, nc = model.hfield_adr[sid], model.hfield_nrow[sid], model.hfield_ncol[sid]
    if h.shape != (nr, nc) or not np.isfinite(h).all():
        raise ValueError('h must be finite with the soil heightfield shape')
    if not np.isfinite(elevation_range) or elevation_range <= 0:
        raise ValueError('elevation_range must be finite and positive')
    model.hfield_data[adr:adr + nr * nc] = np.clip((h - soil_base) / elevation_range, 0, 1).ravel()
    if outer is not None:
        oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, surround_name)
        if oid >= 0:
            outer = np.asarray(outer, float)
            adr, nr, nc = model.hfield_adr[oid], model.hfield_nrow[oid], model.hfield_ncol[oid]
            if outer.shape != (nr, nc) or not np.isfinite(outer).all():
                raise ValueError('outer must be finite with the surround heightfield shape')
            model.hfield_data[adr:adr + nr * nc] = np.clip((outer - surround_base) / model.hfield_size[oid, 2], 0, 1).ravel()
    return sid
