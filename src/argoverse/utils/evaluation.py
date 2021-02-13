

import torch
import torch.nn as nn




def get_ade(forecasted_trajectory, gt_trajectory) -> float:
    """Compute Average Displacement Error.
    Args:
        forecasted_trajectory: Predicted trajectory with shape (bs,pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (bs,pred_len x 2)
    Returns:
        ade: Average Displacement Error
    """
    pred_len = forecasted_trajectory.shape[1]
    bs = forecasted_trajectory.shape[0]
    

    ade=torch.sqrt(
                (forecasted_trajectory[:,:, 0] - gt_trajectory[:,:, 0]) ** 2
                + (forecasted_trajectory[:,:, 1] - gt_trajectory[:,:, 1]) ** 2
    )
            
    
    return ade.mean()


def get_ade_6(forecasted_trajectory, gt_trajectory) -> float:
    """Compute Average Displacement Error.
    Args:
        forecasted_trajectory: Predicted trajectory with shape (bs,pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (bs,pred_len x 2)
    Returns:
        ade: Average Displacement Error
    """
    pred_len = forecasted_trajectory.shape[1]
    bs = forecasted_trajectory.shape[0]
    print(forecasted_trajectory.permute(1,0,2,3)[:,:,:, 0].shape)
    print(torch.sqrt(
                (forecasted_trajectory.permute(1,0,2,3)[:,:,:, 0] - gt_trajectory.unsqueeze(0)[:,:,:, 0]) ** 2
                + (forecasted_trajectory.permute(1,0,2,3)[:,:,:, 1] - gt_trajectory.unsqueeze(0)[:,:,:, 1]) ** 2
    ).shape)
    

    ade=torch.mean(torch.sqrt(
                (forecasted_trajectory.permute(1,0,2,3)[:,:,:, 0] - gt_trajectory.unsqueeze(0)[:,:,:, 0]) ** 2
                + (forecasted_trajectory.permute(1,0,2,3)[:,:,:, 1] - gt_trajectory.unsqueeze(0)[:,:,:, 1]) ** 2
    ),(1,2))
            
    
    return ade.min()


def get_fde(forecasted_trajectory, gt_trajectory) -> float:
    """Compute Final Displacement Error.
    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)
    Returns:
        fde: Final Displacement Error
    """
    fde = torch.sqrt(
        (forecasted_trajectory[:,-1, 0] - gt_trajectory[:,-1, 0]) ** 2
        + (forecasted_trajectory[:,-1, 1] - gt_trajectory[:,-1, 1]) ** 2
    )
    return fde.mean()



