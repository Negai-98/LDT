import os
import numpy as np
import torch
from tqdm import tqdm
import mitsuba as mi
import matplotlib.pyplot as plt


def standardize_bbox(inputs):
    pc = inputs
    centroid = np.mean(pc, axis=0, keepdims=True)
    pc = inputs - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(pc ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)
    pc = pc / furthest_distance / 1.3
    z_min = pc[:, 1].min()
    return pc, z_min


def colormap(x, y, z, light=1):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec = vec / norm * light
    return [vec[0], vec[1], vec[2]]


# airplane [0.1, 0.4, 0.5]
def npy2xml(pc, path, norm, color=None, ball_size=0.015):
    if pc.shape[-1] != 3:
        pc = pc.transpose(1, 0)
    if norm:
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()
        pcl, z_min = standardize_bbox(pc)
    else:
        pcl = pc / 1.3
        z_min = pcl[:, 1].min()
    xml_head = \
        """
    <scene version="3.0.0">
        <integrator type="path">
            <integer name="max_depth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="far_clip" value="100"/>
            <float name="near_clip" value="0.1"/>
            <transform name="to_world">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>

            <sampler type="ldsampler">
                <integer name="sample_count" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="800"/>
                <integer name="height" value="600"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.1"/>
            <float name="int_ior" value="1.46"/>
            <rgb name="diffuse_reflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>

    """

    xml_ball_segment = \
        """
        <shape type="sphere">
            <float name="radius" value="%f"/>
            <transform name="to_world">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """ % ball_size

    xml_tail = \
        """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="to_world">
                <scale x="10" y="10" z="1"/>
                <translate x="0" y="0" z="{:}"/>
            </transform>
        </shape>

        <shape type="rectangle">
            <transform name="to_world">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """.format(z_min)

    xml_segments = [xml_head]
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    for i in range(pcl.shape[0]):
        if color is None:
            map = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
        else:
            map = colormap(color[0], color[1], color[2], color[3])
        xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *map))
    xml_segments.append(xml_tail)
    xml_content = str.join('', xml_segments)

    with open(path, 'w') as f:
        f.write(xml_content)


def visualize_3D(points, out_path, path_name, norm, color=None, type='scalar_rgb'):
    xml_path = os.path.join(out_path, 'render.xml')
    img_path = os.path.join(out_path, path_name)
    npy2xml(points, xml_path, norm, color=color)
    mi.set_variant(type)
    scene = mi.load_file(xml_path)
    image = mi.render(scene, spp=64)
    mi.util.write_bitmap(img_path, image)


def render_3D(path, sample, number=None, norm=True, show=False, color=None, mode="scalar_rgb", ball_size=0.015):
    if number is not None:
        sample = sample[:number]
    tbar = tqdm(range(len(sample)))
    for idx in tbar:
        xml_path = os.path.join(path, 'render.xml'.format(idx))
        img_path = os.path.join(path, '{:}.png'.format(idx))
        npy2xml(sample[idx], xml_path, norm, color=color, ball_size=ball_size)
        mi.set_variant(mode)
        scene = mi.load_file(xml_path)
        image = mi.render(scene, spp=64)
        if show:
            plt.imshow(image)
            plt.show()
        mi.util.write_bitmap(img_path, image)
