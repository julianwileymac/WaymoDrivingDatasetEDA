# Dash visualization
import os

from waymo_open_dataset.v2.perception.utils import lidar_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import transform_utils

# from waymo_open_dataset.camera.ops import py_camera_model_ops

import lidar_utils_fix as lf
from typing import Optional
import warnings
from pathlib import Path

import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2
import glob
import plotly.graph_objs as go
import plotly
from plotly.offline import iplot
import dash
from dash import Dash, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_deck
import colorlover as cl
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import patches
import plotly.express as px

from dotenv import load_dotenv
from jupyter_dash import JupyterDash
import pydeck as pdk
from PIL import Image

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")

# Path to the directory with all components
dataset_dir = '/media/julian/Volume F/Shared/Data/Waymo Open Dataset'
data_dir = Path(dataset_dir)

# context_name = '10023947602400723454_1120_000_1140_000'
context_name = '10226164909075980558_180_000_200_000'

# context_name = '10017090168044687777_6380_000_6400_000'

def read(dataset: str = 'perception', split: str='training', tag: str='camera_image', context_name: str='10226164909075980558_180_000_200_000') -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag.
  dataset: Either motion or perception
  split: train, testing, etc.
  tag: camera_image, lidar, etc.
    context_name: The name of the context to read.
    :returns: A Dask DataFrame with the data.
  """
  print(f'{dataset_dir}/{dataset}/{split}/{tag}/{context_name}.parquet')
  paths = tf.io.gfile.glob(f'{dataset_dir}/{dataset}/{split}/{tag}/{context_name}.parquet')
  print(paths)
  return dd.read_parquet(paths)

def load_lidar_df(context_name: str):
    """Loads the LiDAR data from the frame.
    context_name is the name of the context to read.
    :returns: The LiDAR data.
    """

    lidar_df = read('perception', 'training', 'lidar')
    lidar_box_df = read('perception','training','lidar_box')
    lidar_pose_df = read('perception', 'training', 'lidar_pose')
    lidar_calibration_df = read('perception', 'training', 'lidar_calibration')
    lidar_segmentation_df = read('perception', 'training', 'lidar_segmentation')
    lidar_camera_projection_df = read('perception', 'training', 'lidar_camera_projection')
    lidar_camera_synced_box_df = read('perception', 'training', 'lidar_camera_synced_box')
    lidar_hkp_df = read('perception', 'training', 'lidar_hkp')
    projected_lidar_box_df = read('perception', 'training', 'projected_lidar_box')
    stats_df = read('perception', 'training', 'stats')
    vehicle_pose_df = read('perception', 'training', 'vehicle_pose')

    lidar_all_df = v2.merge(lidar_df, lidar_box_df)
    lidar_all_df = v2.merge(lidar_all_df, lidar_pose_df)
    lidar_all_df = v2.merge(lidar_all_df, lidar_calibration_df)
    lidar_all_df = v2.merge(lidar_all_df, lidar_segmentation_df)
    lidar_all_df = v2.merge(lidar_all_df, lidar_camera_projection_df)
    lidar_all_df = v2.merge(lidar_all_df, lidar_camera_synced_box_df)
    lidar_all_df = v2.merge(lidar_all_df, lidar_hkp_df)
    lidar_all_df = v2.merge(lidar_all_df, projected_lidar_box_df)
    vehicle_df = v2.merge(vehicle_pose_df, stats_df)
    lidar_all_df = v2.merge(lidar_all_df, vehicle_df)

    _, lidar_row = next(iter(lidar_df.iterrows()))
    _, lidar_box_row = next(iter(lidar_box_df.iterrows()))
    _, lidar_pose_row = next(iter(lidar_pose_df.iterrows()))
    _, lidar_calibration_row = next(iter(lidar_calibration_df.iterrows()))
    l1 = lidar_calibration_df.loc[lidar_calibration_df['key.laser_name'] == 1]
    _, lidar_calibration_row1 = next(iter(l1.iterrows()))
    _, lidar_camera_projection_row = next(iter(lidar_camera_projection_df.iterrows()))
    _, lidar_camera_synced_box_row = next(iter(lidar_camera_synced_box_df.iterrows()))
    _, lidar_hkp_row = next(iter(lidar_hkp_df.iterrows()))
    _, projected_lidar_box_row = next(iter(projected_lidar_box_df.iterrows()))
    _, stats_row = next(iter(stats_df.iterrows()))
    _, lidar_segmentation_row = next(iter(lidar_segmentation_df.iterrows()))
    _, vehicle_pose_row = next(iter(vehicle_pose_df.iterrows()))

    lidar = v2.LiDARComponent.from_dict(lidar_row)
    lidar_box = v2.LiDARBoxComponent(lidar_box_row)
    lidar_pose = v2.LiDARPoseComponent.from_dict(lidar_pose_row)
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(lidar_calibration_row1)
    lidar_camera_projection = v2.LiDARCameraProjectionComponent.from_dict(lidar_camera_projection_row)
    lidar_camera_synced_box = v2.LiDARCameraSyncedBoxComponent.from_dict(lidar_camera_synced_box_row)
    lidar_hkp = v2.LiDARHkPComponent.from_dict(lidar_hkp_row)
    projected_lidar_box = v2.ProjectedLiDARBoxComponent.from_dict(projected_lidar_box_row)
    stats = v2.StatsComponent.from_dict(stats_row)
    lidar_segmentation = v2.LiDARSegmentationLabelComponent.from_dict(lidar_segmentation_row)
    vehicle_pose = v2.VehiclePoseComponent.from_dict(vehicle_pose_row)

    it = iter(lidar_all_df.iterrows())

    # return lidar, lidar_box, lidar_pose, lidar_calibration, lidar_camera_projection, lidar_camera_synced_box, lidar_hkp, projected_lidar_box, stats, lidar_segmentation, vehicle_pose
    return lidar_all_df, it

def get_lidar_components(row):
    """

    :param row:
    :return:
    """

    lidar = v2.LiDARComponent.from_dict(row)
    lidar_box = v2.LiDARBoxComponent(row)
    lidar_pose = v2.LiDARPoseComponent.from_dict(row)
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(row)
    lidar_camera_projection = v2.LiDARCameraProjectionComponent.from_dict(row)
    lidar_camera_synced_box = v2.LiDARCameraSyncedBoxComponent.from_dict(row)
    lidar_hkp = v2.LiDARHkPComponent.from_dict(row)
    projected_lidar_box = v2.ProjectedLiDARBoxComponent.from_dict(row)
    stats = v2.StatsComponent.from_dict(row)
    lidar_segmentation = v2.LiDARSegmentationLabelComponent.from_dict(row)
    vehicle_pose = v2.VehiclePoseComponent.from_dict(row)

    return lidar, lidar_box, lidar_pose, lidar_calibration, lidar_camera_projection, lidar_camera_synced_box, lidar_hkp, projected_lidar_box, stats, lidar_segmentation, vehicle_pose

def get_point_cloud(lidar: v2.LiDARComponent, lidar_pose: v2.LiDARPoseComponent, lidar_calibration: v2.LiDARCalibrationComponent, vehicle_pose: v2.VehiclePoseComponent):
  """Gets the point cloud for the LiDAR data.
  :param lidar: The LiDAR data.
  :param lidar_pose: The LiDAR pose data.
  :param lidar_calibration: The LiDAR calibration data.
  :param vehicle_pose: The vehicle pose data.
  :returns: The LiDAR point cloud.
  """
  if lidar is None or lidar_pose is None or lidar_calibration is None:
    return None

  points = lf.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_calibration,
                                                 lidar_pose.range_image_return1, vehicle_pose, False)
  return points
def load_camera_df(context_name: str):
    """
    Loads the camera data from the frame.
    :param context_name:
    :return:
    """
    camera_box_df = read('perception', 'training', 'camera_box')
    camera_image_df = read('perception', 'training', 'camera_image')
    camera_hkp_df = read('perception', 'training', 'camera_hkp')
    asssociation_df = read('perception', 'training', 'camera_to_lidar_box_association')
    camera_calibration_df = read('perception', 'training', 'camera_calibration')

    all_camera_df = v2.merge(camera_image_df, camera_box_df, left_group=True, right_group=True)
    all_camera_df = v2.merge(all_camera_df, camera_hkp_df)
    all_camera_df = v2.merge(all_camera_df, camera_calibration_df)
    all_camera_df = v2.merge(all_camera_df, asssociation_df)

    _, camera_box_row = next(iter(camera_box_df.iterrows()))
    _, camera_image_row = next(iter(camera_image_df.iterrows()))
    _, camera_hkp_row = next(iter(camera_hkp_df.iterrows()))
    _, association_row = next(iter(asssociation_df.iterrows()))
    _, camera_calibration_row = next(iter(camera_calibration_df.iterrows()))

    camera_box = v2.CameraBoxComponent.from_dict(camera_box_row)
    camera_image = v2.CameraImageComponent.from_dict(camera_image_row)
    camera_hkp = v2.CameraHkPComponent.from_dict(camera_hkp_row)
    association = v2.CameraToLiDARBoxAssociationComponent.from_dict(association_row)
    camera_calibration = v2.CameraCalibrationComponent.from_dict(camera_calibration_row)

    it = iter(all_camera_df)
    # return camera_box, camera_image, camera_hkp, association, camera_calibration
    return all_camera_df, it
def get_camera_components(row):
    """

    :param row:
    :return:
    """
    camera_box = v2.CameraBoxComponent.from_dict(row)
    camera_image = v2.CameraImageComponent.from_dict(row)
    camera_hkp = v2.CameraHkPComponent.from_dict(row)
    association = v2.CameraToLiDARBoxAssociationComponent.from_dict(row)
    camera_calibration = v2.CameraCalibrationComponent.from_dict(row)

    return camera_box, camera_image, camera_hkp, association, camera_calibration


def get_contexts(dataset_dir='/media/julian/Volume F/Shared/Data/Waymo Open Dataset/perception/training/'):
  """

  :param dataset_dir:
  :return:
  """
  camera_files = glob.glob(dataset_dir + 'camera_image/*.parquet')
  contexts = []

  for file in camera_files:
    context = file[87:-8]
    contexts.append(context)

  return contexts


def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
    vehicle_pose: Vehicle pose transform from vehicle into world coordinate
      system.
    calibration: Camera calibration details (including intrinsics/extrinsics).
    points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
    Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
      tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
      [4, 4])
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant([
      calibration.width,
      calibration.height,
      open_dataset.CameraCalibration.GLOBAL_SHUTTER,
    ],
                         dtype=tf.int32)
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    # return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,camera_image_metadata,world_points).numpy()
    return (extrinsic, intrinsic, metadata,
                                              camera_image_metadata,
                                              world_points)
def show_camera_image(camera_image, layout):
  """Display the given camera image."""
  ax = plt.subplot(*layout)
  plt.imshow(tf.image.decode_jpeg(camera_image.image))
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')
  return ax

def rgba(r):
  """Generates a color based on range.

  Args:
    r: the range value of a given point.
  Returns:
    The color for a given range
  """
  c = plt.get_cmap('jet')((r % 20.0) / 20.0)
  c = list(c)
  c[-1] = 0.5  # alpha
  return c

def draw_2d_box(ax, u, v, color, linewidth=1):
  """Draws 2D bounding boxes as rectangles onto the given axis."""
  rect = patches.Rectangle(
      xy=(u.min(), v.min()),
      width=u.max() - u.min(),
      height=v.max() - v.min(),
      linewidth=linewidth,
      edgecolor=color,
      facecolor=list(color) + [0.1])  # Add alpha for opacity
  ax.add_patch(rect)


def draw_3d_wireframe_box(ax, u, v, color, linewidth=3):
  """Draws 3D wireframe bounding boxes onto the given axis."""
  # List of lines to interconnect. Allows for various forms of connectivity.
  # Four lines each describe bottom face, top face and vertical connectors.
  lines = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
           (0, 4), (1, 5), (2, 6), (3, 7))

  for (point_idx1, point_idx2) in lines:
    line = plt.Line2D(
        xdata=(int(u[point_idx1]), int(u[point_idx2])),
        ydata=(int(v[point_idx1]), int(v[point_idx2])),
        linewidth=linewidth,
        color=list(color) + [0.5])  # Add alpha for opacity
    ax.add_line(line)
def plot_cloud(pointcloud, title):
    plotly.offline.init_notebook_mode()
    trace = go.Scatter3d(
      x=-pointcloud[:, 0],
      y=pointcloud[:, 1],  # <-- Put your data instead
      z=pointcloud[:, 2],  # <-- Put your data instead
      mode='markers',
      marker={
        'size': 1,
        'opacity': 0.8,
      }
    )

    layout = go.Layout(
      margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
      scene=dict(
        xaxis=dict(title="x", range=[-200, 200]),
        yaxis=dict(title="y", range=[-200, 200]),
        zaxis=dict(title="z", range=[-200, 200])
      ),
      title=title
    )
    data = [trace]
    plot_figure = go.Figure(data=data, layout=layout)
    # plotly.offline.iplot(plot_figure)
    return plot_figure

MODES = ['first-person','orbit']
STYLES = ["light", "dark", "satellite"]
DATASETS = ['/media/julian/Volume F/Shared/Data/Waymo Open Dataset/perception/training',
            '/media/julian/Volume F/Shared/Data/Waymo Open Dataset/perception/validation',
            '/media/julian/Volume F/Shared/Data/Waymo Open Dataset/perception/testing']
NAME2COLOR = dict(
    zip(
        ["vehicles", "signs", "cyclists", "pedestrian"],
        cl.to_numeric(cl.scales["4"]["div"]["Spectral"]),
    )
)
CAMERAS = ['Camera 1', 'Camera 2', 'Camera 3', 'Camera 4']
LIDARS = ['Lidar 1', 'Lidar 2', 'Lidar 3']
context_list = get_contexts()
# app = JupyterDash(__name__)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Waymo Autonomous Vehicle Visualization"
server = app.server

def Header(name, app):
  title = html.H1(name, style={"margin-top": 5})
  logo = html.Img(
    src=app.get_asset_url("dash-logo.png"), style={"float": "right", "height": 60}
  )
  link = html.A(logo, href="https://plotly.com/dash/")

  return dbc.Row([dbc.Col(title, md=8), dbc.Col(link, md=4)])

def unsnake(st):
    """BECAUSE_WE_DONT_READ_LIKE_THAT"""
    return st.replace("_", " ").title()

controls = [
    dbc.FormGroup([
    dbc.Label("Map Style"),
    dbc.Select(
      id="select-style",
      options=[{"label":s.capitalize(), "value":s} for s in STYLES],
      value=STYLES[0],
    ),
  ]),
    dbc.FormGroup([
        dbc.Label("Camera Position"),
        dbc.Select(
            id='camera',
            options=[
                    {"label": unsnake(s.replace("CAM_", "")), "value": s}
                    for s in CAMERAS
                ],
                value=CAMERAS[0],
        )
    ]),
    dbc.FormGroup([
            dbc.Label("Image Overlay"),
            dbc.Checklist(
                id="overlay",
                value=[],
                options=[
                    {"label": x.title(), "value": x} for x in ["pointcloud", "boxes"]
                ],
                inline=True,
                switch=True,
            ),
    ]),
    dbc.FormGroup(
        [
            dbc.Label("Frame"),
            html.Br(),
            dbc.Spinner(
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            "Prev", id="prev", n_clicks=0, color="primary", outline=True
                        ),
                        dbc.Button("Next", id="next", n_clicks=0, color="primary"),
                    ],
                    id="button-group",
                    style={"width": "100%"},
                ),
                spinner_style={"margin-top": 0, "margin-bottom": 0},
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Progression"),
            dbc.Spinner(
                dbc.Input(
                    id="progression", type="range", min=0, max=2272, value=0
                ),
                spinner_style={"margin-top": 0, "margin-bottom": 0},
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Lidar Position"),
            dbc.Select(
                id="lidar",
                value=LIDARS[0],
                options=[
                    {"label": unsnake(s.replace("LIDAR_", "")), "value": s}
                    for s in LIDARS
                ],
            ),
        ]
    ),
    dbc.FormGroup(
            [
                dbc.Label("Lidar View Mode"),
                dbc.Select(
                    id="view-mode",
                    value="map",
                    options=[
                        {"label": unsnake(x), "value": x}
                        for x in ["first_person", "orbit", "map"]
                    ],
                ),
            ]
        ),
]


deck_card = dbc.Card(
    dash_deck.DeckGL(id="deck-pointcloud", tooltip={"html": "<b>Label:</b> {name}"}),
    body=True,
    style={"height": "calc(95vh - 215px)"},
)

def compute_pointcloud_for_image(
    cam_row,
    dot_size: int = 2,
    pointsensor_channel: str = "LIDAR_TOP",
    camera_channel: str = "CAM_FRONT",
    out_path: str = None,
):
    """Scatter-plots a point-cloud on top of image.
    Args:
        sample_token: Sample token.
        dot_size: Scatter plot dot size.
        pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
        camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        out_path: Optional path to save the rendered figure to disk.
    Returns:
        tuple containing the points, array of colors and a pillow image
    """
    # sample_record = lv5.get("sample", sample_token)

    cam_image1 = v2.CameraImageComponent.from_dict(cam_row)
    im = tf.image.decode_jpeg(cam_image1.image)
    # plt.imshow(tf.image.decode_jpeg(cam_image1.image))
    # Here we just grab the front camera and the point sensor.
    # pointsensor_token = sample_record["data"][pointsensor_channel]
    # camera_token = sample_record["data"][camera_channel]

    # points, coloring, im = lv5.explorer.map_pointcloud_to_image(
    #    pointsensor_token, camera_token
    # )

    return im
def render_box_in_image(camera_image, sample: str, camera_channel: str):
    """

    :param lv5:
    :param im:
    :param sample:
    :param camera_channel:
    :return:
    """
    # camera_token = sample["data"][camera_channel]
    # data_path, boxes, camera_intrinsic = lv5.get_sample_data(
    #     camera_token, flat_vehicle_coordinates=False
    # )
    """
    im = tf.image.decode_jpeg(camera_image.image)
    arr = np.array(im)

    for j, (object_id, x, y) in enumerate(zip(camera_box.key.camera_object_id, camera_box.box.center.x, camera_box.box.center.y)):
        print(f'\tid: {object_id},  center: ({x:.1f}, {y:.1f}) px')
        if j > 2:
            print('\t...')
            break

    # for box in boxes:
    #     c = NAME2COLOR[box.name]
    #     box.render_cv2(arr, normalize=True, view=camera_intrinsic, colors=(c, c, c))

    new = Image.fromarray(arr)
    """
    # im = tf.image.decode_jpeg(camera_image.image)
    new = camera_image
    return new

def build_figure(cam_row, lidar, camera, overlay):
    im = compute_pointcloud_for_image(
        cam_row=cam_row, pointsensor_channel=lidar, camera_channel=camera
    )

    if "boxes" in overlay:
        im = render_box_in_image(camera_image=im,sample='test', camera_channel=camera)

    fig = px.imshow(im, binary_format="jpeg", binary_compression_level=2)

    if "pointcloud" in overlay:
        pass
        """
        fig.add_trace(
            go.Scattergl(
                x=points[0,],
                y=points[1,],
                mode="markers",
                opacity=0.4,
                marker_color=coloring,
                marker_size=3,
            )
        )
        """

    fig.update_layout(
        margin=dict(l=10, r=10, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode=False,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, range=(0, im.size[0]))
    fig.update_yaxes(showticklabels=False, showgrid=False, range=(im.size[1], 0))

    return fig
def build_deck(mode, pc_df, polygon_data):
    if mode == "first_person":
        view = pdk.View(type="FirstPersonView", controller=True)
        view_state = pdk.ViewState(latitude=0, longitude=0, bearing=-90, pitch=15)
        point_size = 10

    elif mode == "orbit":
        view = pdk.View(type="OrbitView", controller=True)
        view_state = pdk.ViewState(
            target=[0, 0, 1e-5],
            controller=True,
            zoom=23,
            rotation_orbit=-90,
            rotation_x=15,
        )
        point_size = 3

    else:
        view_state = pdk.ViewState(
            latitude=0,
            longitude=0,
            bearing=45,
            pitch=50,
            zoom=20,
            max_zoom=30,
            position=[0, 0, 1e-5],
        )
        view = pdk.View(type="MapView", controller=True)
        point_size = 1

    pc_layer = pdk.Layer(
        "PointCloudLayer",
        data=pc_df,
        get_position=["x", "y", "z"],
        get_color=[255, 255, 255],
        auto_highlight=True,
        pickable=False,
        point_size=point_size,
        coordinate_system=2,
        coordinate_origin=[0, 0],
    )

    box_layer = pdk.Layer(
        "PolygonLayer",
        data=polygon_data,
        stroked=True,
        pickable=True,
        filled=True,
        extruded=True,
        opacity=0.2,
        wireframe=True,
        line_width_min_pixels=1,
        get_polygon="polygon",
        get_fill_color="color",
        get_line_color=[255, 255, 255],
        get_line_width=0,
        coordinate_system=2,
        get_elevation="elevation",
    )

    tooltip = {"html": "<b>Label:</b> {name}"}

    r = pdk.Deck(
        [pc_layer, box_layer],
        initial_view_state=view_state,
        views=[view],
        tooltip=tooltip,
        map_provider=None,
    )

    return r


app.layout = dbc.Container(
    [
        Header("Dash Autonomous Driving Demo", app),
        html.Br(),
        dbc.Card(dbc.Row([dbc.Col(c) for c in controls], form=True), body=True),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Card(dcc.Graph(id="graph-camera"), body=True), md=5),
                dbc.Col(deck_card, md=7),

            ]
        ),
        html.Div(id="temp"),
    ],
    fluid=True,
)

"""
app.layout = html.Div([

    html.H4('Interactive LiDAR Plot'),
    dcc.Dropdown(names ,value='points', id='point-drop'),
    dcc.Graph(id="graph"),
])


@callback(Output('graph', 'figure'), Input('point-drop','value'))
def update_graph(value):
    points_np = points.numpy()
    points_fig = plot_cloud(points_np, title=value)

    return points_fig
"""

@app.callback(
    Output("progression", "value"),
    [Input("prev", "n_clicks"), Input("next", "n_clicks")],
    [State("progression", "value")],
)
def update_current_token(btn_prev, btn_next, curr_progress):
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]

    if "next" in prop_id:
        return min(int(curr_progress) + 1, len(context_list))
    elif "prev" in prop_id:
        return max(0, int(curr_progress) - 1)
    else:
        return dash.no_update

@app.callback(
    [
        Output("graph-camera", "figure"),
        Output("deck-pointcloud", "data"),
        Output("button-group", "children"),
        Output("progression", "type"),
    ],
    [
        Input("progression", "value"),
        Input("camera", "value"),
        Input("lidar", "value"),
        Input("overlay", "value"),
        Input("view-mode", "value"),
    ],
)
def update_graphs(progression, camera, lidar, overlay, view_mode):
    token = context_list[int(progression)]
    # lidar, lidar_box, lidar_pose, lidar_calibration, lidar_camera_projection, lidar_camera_synced_box, lidar_hkp, \
    #    projected_lidar_box, stats, lidar_segmentation, vehicle_pose = load_lidar(token)
    full_lidar_df, lidar_it = load_lidar_df(token)
    _, li_row = next(lidar_it)
    lidar_pc = v2.LiDARComponent.from_dict(li_row)
    lidar_box1 = v2.LiDARBoxComponent.from_dict(li_row)
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(li_row)
    lidar_pose = v2.LiDARPoseComponent.from_dict(li_row)
    vehicle_pose = v2.VehiclePoseComponent.from_dict(li_row)
    pc = lf.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_calibration,
                                               lidar_pose.range_image_return1, vehicle_pose, False)
    pc_df = pd.DataFrame(pc, columns=["x", "y", "z"])

    full_lidar_df = (
        full_lidar_df.groupby(['key.segment_context_name', 'key.frame_timestamp_micros', 'key.laser_name'])
        .agg(list)
        .reset_index()
    )    # camera_box, camera_image, camera_hkp, association, camera_calibration = load_camera(token)
    full_camera_df, camera_it = load_camera_df(progression)
    full_camera_df = (
        full_camera_df.groupby(['key.segment_context_name', 'key.frame_timestamp_micros', 'key.camera_name'])
        .agg(list)
        .reset_index()
    )
    # lidar_box_df = read('perception','training','lidar_box', context_name=token)

    # sample = lv5.get("sample", token)
    # pointsensor_token = sample["data"][lidar]
    # pointsensor = lv5.get("sample_data", pointsensor_token)
    # pc = LidarPointCloud.from_file(lv5.data_path / pointsensor["filename"])
    _, li_row = next(lidar_it)
    _, cam_row = next(camera_it)

    lidar = v2.LiDARComponent.from_dict(li_row)
    lidar_box1 = v2.LiDARBoxComponent.from_dict(li_row)
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(li_row)
    lidar_pose = v2.LiDARPoseComponent.from_dict(li_row)
    vehicle_pose = v2.VehiclePoseComponent.from_dict(li_row)



    # if lidar in ["LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]:
    #     pc_df.z = -pc_df.z + 1

    # _, boxes, camera_intrinsic = lv5.get_sample_data(
    #     pointsensor["token"], flat_vehicle_coordinates=False
    # )

    n = len(lidar_box1.key.laser_object_id)

    box_tensor = tf.constant(
        [lidar_box1.box.center.x, lidar_box1.box.center.y, lidar_box1.box.center.z, lidar_box1.box.size.x,
         lidar_box1.box.size.y, lidar_box1.box.size.z, lidar_box1.box.heading])
    print(box_tensor)
    boxes = tf.transpose(box_tensor)

    center_x, center_y, center_z, length, width, height, heading = tf.unstack(
        boxes, axis=-1)

    # [N, 3, 3]
    rotation = transform_utils.get_yaw_rotation(heading)
    # [N, 3]
    translation = tf.stack([center_x, center_y, center_z], axis=-1)

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    # [N, 8, 3]
    corners = tf.reshape(
        tf.stack([
            l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
            -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2
        ],
            axis=-1), [-1, 8, 3])
    # [N, 8, 3]
    corners = tf.einsum('nij,nkj->nki', rotation, corners) + tf.expand_dims(
        translation, axis=-2)
    box_list = []
    for j, (object_id, x, y, z, size_x, size_y, size_z, heading) in enumerate(
            zip(lidar_box1.key.laser_object_id, lidar_box1.box.center.x, lidar_box1.box.center.y,
                lidar_box1.box.center.z, lidar_box1.box.size.x, lidar_box1.box.size.y, lidar_box1.box.size.z,
                lidar_box1.box.heading)):
            box_list.append([x,y,z,size_x,size_y,size_z,heading])

    np_box_list = np.asarray(box_list)

    polygon_data = [
        {
            "name": box,
            "polygon": corners[i,:4,:],
            "width": box_list[i, 3],
            "length": box_list[i, 4],
            "elevation": box_list[i,5],
            "color": NAME2COLOR[box],
            "token": token,
            "distance": np.sqrt(np.square(np_box_list[i,:3]).sum()),
        }

        for i, box in enumerate(lidar_box1.key.laser_object_id)
    ]
        # for box in enumerate(corners)
    # Build figure and pydeck object
    fig = build_figure(cam_row=cam_row, lidar=LIDARS[0], camera=CAMERAS[0], overlay='boxes')
    r = build_deck(view_mode, pc_df, polygon_data)

    return fig, r.to_json(), dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True, port=18070)