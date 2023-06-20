# Dash visualization

from waymo_open_dataset.v2.perception.utils import lidar_utils
import lidar_utils_fix as lf
from typing import Optional
import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path

import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2

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

def load_lidar(context_name: str):
  """Loads the LiDAR data from the frame.
  context_name is the name of the context to read.
  :returns: The LiDAR data.
  """

  lidar_df = read('perception', 'training', 'lidar')
  lidar_pose_df = read('perception', 'training', 'lidar_pose')
  lidar_calibration_df = read('perception', 'training', 'lidar_calibration')
  lidar_segmentation_df = read('perception', 'training', 'lidar_segmentation')
  lidar_camera_projection_df = read('perception', 'training', 'lidar_camera_projection')
  lidar_camera_synced_box_df = read('perception', 'training', 'lidar_camera_synced_box')
  lidar_hkp_df = read('perception', 'training', 'lidar_hkp')
  projected_lidar_box_df = read('perception', 'training', 'projected_lidar_box')
  stats_df = read('perception', 'training', 'stats')
  vehicle_pose_df = read('perception', 'training', 'vehicle_pose')

  _, lidar_row = next(iter(lidar_df.iterrows()))
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
  lidar_pose = v2.LiDARPoseComponent.from_dict(lidar_pose_row)
  lidar_calibration = v2.LiDARCalibrationComponent.from_dict(lidar_calibration_row1)
  lidar_camera_projection = v2.LiDARCameraProjectionComponent.from_dict(lidar_camera_projection_row)
  lidar_camera_synced_box = v2.LiDARCameraSyncedBoxComponent.from_dict(lidar_camera_synced_box_row)
  lidar_hkp = v2.LiDARHkPComponent.from_dict(lidar_hkp_row)
  projected_lidar_box = v2.ProjectedLiDARBoxComponent.from_dict(projected_lidar_box_row)
  stats = v2.StatsComponent.from_dict(stats_row)
  lidar_segmentation = v2.LiDARSegmentationLabelComponent.from_dict(lidar_segmentation_row)
  vehicle_pose = v2.VehiclePoseComponent.from_dict(vehicle_pose_row)

  return lidar, lidar_pose, lidar_calibration, lidar_camera_projection, lidar_camera_synced_box, lidar_hkp, \
    projected_lidar_box, stats, lidar_segmentation, vehicle_pose

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
def load_camera(context_name: str):
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

  return camera_box, camera_image, camera_hkp, association, camera_calibration

