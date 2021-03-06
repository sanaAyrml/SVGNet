"""This module is used for computing map features for motion forecasting baselines."""

from typing import Any, Dict, List, Tuple
from .transform import *
import math 
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import cascaded_union
import cv2

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import (
    get_nt_distance,
    remove_overlapping_lane_seq,
)
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from .baseline_config import (
    _MANHATTAN_THRESHOLD,
    _DFS_THRESHOLD_FRONT_SCALE,
    _DFS_THRESHOLD_BACK_SCALE,
    _MAX_SEARCH_RADIUS_CENTERLINES,
    _MAX_CENTERLINE_CANDIDATES_TEST,
)
import operator



class MapFeaturesUtils:
    """Utils for computation of map-based features."""
    def __init__(self):
        """Initialize class."""
        self._MANHATTAN_THRESHOLD = _MANHATTAN_THRESHOLD
        self._DFS_THRESHOLD_FRONT_SCALE = _DFS_THRESHOLD_FRONT_SCALE
        self._DFS_THRESHOLD_BACK_SCALE = _DFS_THRESHOLD_BACK_SCALE
        self._MAX_SEARCH_RADIUS_CENTERLINES = _MAX_SEARCH_RADIUS_CENTERLINES
        self._MAX_CENTERLINE_CANDIDATES_TEST = _MAX_CENTERLINE_CANDIDATES_TEST

    def get_point_in_polygon_score(self, lane_seq: List[int],
                                   xy_seq: np.ndarray, city_name: str,
                                   avm: ArgoverseMap) -> int:
        """Get the number of coordinates that lie insde the lane seq polygon.

        Args:
            lane_seq: Sequence of lane ids
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
        Returns:
            point_in_polygon_score: Number of coordinates in the trajectory that lie within the lane sequence

        """
        lane_seq_polygon = cascaded_union([
            Polygon(avm.get_lane_segment_polygon(lane, city_name)).buffer(0)
            for lane in lane_seq
        ])
        point_in_polygon_score = 0
        for xy in xy_seq:
            point_in_polygon_score += lane_seq_polygon.contains(Point(xy))
        return point_in_polygon_score

    def sort_lanes_based_on_point_in_polygon_score(
            self,
            lane_seqs: List[List[int]],
            xy_seq: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
    ) -> List[List[int]]:
        """Filter lane_seqs based on the number of coordinates inside the bounding polygon of lanes.

        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
        Returns:
            sorted_lane_seqs: Sequences of lane sequences sorted based on the point_in_polygon score

        """
        point_in_polygon_scores = []
        for lane_seq in lane_seqs:
            point_in_polygon_scores.append(
                self.get_point_in_polygon_score(lane_seq, xy_seq, city_name,
                                                avm))
        randomized_tiebreaker = np.random.random(len(point_in_polygon_scores))
        sorted_point_in_polygon_scores_idx = np.lexsort(
            (randomized_tiebreaker, np.array(point_in_polygon_scores)))[::-1]
        sorted_lane_seqs = [
            lane_seqs[i] for i in sorted_point_in_polygon_scores_idx
        ]
        sorted_scores = [
            point_in_polygon_scores[i]
            for i in sorted_point_in_polygon_scores_idx
        ]
        return sorted_lane_seqs, sorted_scores

    def get_heuristic_centerlines_for_test_set(
            self,
            lane_seqs: List[List[int]],
            xy_seq: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
            max_candidates: int,
            scores: List[int],
    ) -> List[np.ndarray]:
        """Sort based on distance along centerline and return the centerlines.
        
        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
            max_candidates: Maximum number of centerlines to return
        Return:
            sorted_candidate_centerlines: Centerlines in the order of their score 

        """
        aligned_centerlines = []
        diverse_centerlines = []
        diverse_scores = []
        num_candidates = 0

        # Get first half as aligned centerlines
        aligned_cl_count = 0
        for i in range(len(lane_seqs)):
            lane_seq = lane_seqs[i]
            score = scores[i]
            diverse = True
            centerline = avm.get_cl_from_lane_seq([lane_seq], city_name)[0]
            if aligned_cl_count < int(max_candidates / 2):
                start_dist = LineString(centerline).project(Point(xy_seq[0]))
                end_dist = LineString(centerline).project(Point(xy_seq[-1]))
                if end_dist > start_dist:
                    aligned_cl_count += 1
                    aligned_centerlines.append(centerline)
                    diverse = False
            if diverse:
                diverse_centerlines.append(centerline)
                diverse_scores.append(score)

        num_diverse_centerlines = min(len(diverse_centerlines),
                                      max_candidates - aligned_cl_count)
        test_centerlines = aligned_centerlines
        if num_diverse_centerlines > 0:
            probabilities = ([
                float(score + 1) / (sum(diverse_scores) + len(diverse_scores))
                for score in diverse_scores
            ] if sum(diverse_scores) > 0 else [1.0 / len(diverse_scores)] *
                             len(diverse_scores))
            diverse_centerlines_idx = np.random.choice(
                range(len(probabilities)),
                num_diverse_centerlines,
                replace=False,
                p=probabilities,
            )
            diverse_centerlines = [
                diverse_centerlines[i] for i in diverse_centerlines_idx
            ]
            test_centerlines += diverse_centerlines

        return test_centerlines

    def get_candidate_centerlines_for_trajectory(
            self,
            xy: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
            yaw_deg:int,
            viz: bool = False,
            max_search_radius: float = 100.0,
            seq_len: int = 50,
            max_candidates: int = 10,
            mode: str = "test",
    ) -> List[np.ndarray]:


        centroid=xy[19]
#         print("centroid is  ",centroid)
#         print("yaw_deg is ",yaw_deg)
        
        raster_size=(224,224)
        ego_yaw=-math.pi*yaw_deg/180
#         print("ego_yaw is ",ego_yaw)
        
        world_to_image_space = world_to_image_pixels_matrix(
        raster_size,
        (0.6,0.6),
        ego_translation_m=centroid,
        ego_yaw_rad=ego_yaw,
        ego_center_in_image_ratio=np.array([0.125,0.5]),

        )
        
                
        
        return world_to_image_space

    def get_candidate_centerlines_for_raster(
            self,
            xy: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
            yaw_deg:int,
            viz: bool = False,
            max_search_radius: float = 100.0,
            world_to_image_space=None
    ) -> List[np.ndarray]:

        # Get all lane candidates within a bubble
        curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
            xy[-1, 0], xy[-1, 1], city_name, 100)

        candidate_centerlines=avm.find_local_lane_centerlines(xy[-1, 0], xy[-1, 1], city_name, max_search_radius)
        lane_distance=[]
        centroid=xy[19]
        for lane in candidate_centerlines:
            lane_distance.append([lane,np.linalg.norm(centroid-lane[0][:2])])
        lane_distance.sort(key=operator.itemgetter(1))
        
        raster_size=(224,224)
        candidate_centerlines_normalized=[]

        if viz:
            img = 255 * np.ones(shape=(raster_size[0], raster_size[1], 3), dtype=np.uint8)

#             plt.figure(0, figsize=(8, 7))
            for centerline_coords,distance in lane_distance:
                cnt_line = transform_points(centerline_coords, world_to_image_space)
                cv2.polylines(img, [cv2_subpixel(cnt_line)], False, (0, 0, 0), lineType=cv2.LINE_AA,shift=CV2_SHIFT )
                cropped_vector=crop_tensor(cnt_line, raster_size)
                if len(cropped_vector)>1:
                    candidate_centerlines_normalized.append(cropped_vector)
#                     print("cropped cntr-line",cropped_vector)
#                     break
                 
                    
                
            img=img.astype(np.float32) / 255
            
        
        
        return img,candidate_centerlines_normalized

    
    
    def compute_map_features(
            self,
            agent_track: np.ndarray,
            obs_len: int,
            seq_len: int,
            raw_data_format: Dict[str, int],
            mode: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute map based features for the given sequence.

        If the mode is test, oracle_nt_dist will be empty, candidate_nt_dist will be populated.
        If the mode is train/val, oracle_nt_dist will be populated, candidate_nt_dist will be empty.

        Args:
            agent_track : Data for the agent track
            obs_len : Length of observed trajectory
            seq_len : Length of the sequence
            raw_data_format : Format of the sequence
            mode: train/val/test mode
            
        Returns:
            oracle_nt_dist (numpy array): normal and tangential distances for oracle centerline
                map_feature_helpers (dict): Dictionary containing helpers for map features

        """
        # Get observed 2 secs of the agent
        agent_xy = agent_track[:, [raw_data_format["X"], raw_data_format["Y"]
                                   ]].astype("float")
        agent_track_obs = agent_track[:obs_len]
        agent_xy_obs = agent_track_obs[:, [
            raw_data_format["X"], raw_data_format["Y"]
        ]].astype("float")

        # Get API for Argo Dataset map
        avm = ArgoverseMap()

        city_name = agent_track[0, raw_data_format["CITY_NAME"]]

        # Get candidate centerlines using observed trajectory
        if mode == "test":
            oracle_centerline = np.full((seq_len, 2), None)
            oracle_nt_dist = np.full((seq_len, 2), None)
            candidate_centerlines = self.get_candidate_centerlines_for_trajectory(
                agent_xy_obs,
                city_name,
                avm,
                viz=False,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                max_candidates=self._MAX_CENTERLINE_CANDIDATES_TEST,
            )

            # Get nt distance for the entire trajectory using candidate centerlines
            candidate_nt_distances = []
            for candidate_centerline in candidate_centerlines:
                candidate_nt_distance = np.full((seq_len, 2), None)
                candidate_nt_distance[:obs_len] = get_nt_distance(
                    agent_xy_obs, candidate_centerline)
                candidate_nt_distances.append(candidate_nt_distance)

        else:
            oracle_centerline = self.get_candidate_centerlines_for_trajectory(
                agent_xy,
                city_name,
                avm,
                viz=False,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                mode=mode,
            )[0]
            candidate_centerlines = [np.full((seq_len, 2), None)]
            candidate_nt_distances = [np.full((seq_len, 2), None)]

            # Get NT distance for oracle centerline
            oracle_nt_dist = get_nt_distance(agent_xy,
                                             oracle_centerline,
                                             viz=False)

        map_feature_helpers = {
            "ORACLE_CENTERLINE": oracle_centerline,
            "CANDIDATE_CENTERLINES": candidate_centerlines,
            "CANDIDATE_NT_DISTANCES": candidate_nt_distances,
        }

        return oracle_nt_dist, map_feature_helpers
