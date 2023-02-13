import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate import (
    MujocoManipulateEnv,quat_from_angle_and_axis
    
)

from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block import MANIPULATE_BLOCK_XML


class hand(MujocoManipulateEnv, EzPickle):
    def __init__(
        self,
        target_position="fixed",
        target_rotation="fixed",
        reward_type="dense",
        **kwargs,
    ):
        MujocoManipulateEnv.__init__(
            self,
            model_path=MANIPULATE_BLOCK_XML,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)
        self.rot_range=0
        self.pos_range=0


    def _sample_goal(self):
        # Select a goal for the object position.
        
        target_pos = self._utils.get_joint_qpos(
            self.model, self.data, "object:joint"
        )[:3]
        target_pos+=self.np_random.uniform(-self.pos_range, self.pos_range,3)

        # Select a goal for the object rotation.
        
        
        angle = self.np_random.uniform(-self.rot_range, self.rot_range)+3.14
        axis = np.array([0.0, 0.0, 1.0])+self.np_random.uniform(0, self.rot_range,3)
        target_quat = quat_from_angle_and_axis(angle, axis)
        

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        #print(goal)
        return goal

    def compute_reward(self, achieved_goal, goal, info):
            if self.reward_type == "sparse":
                success = self._is_success(achieved_goal, goal).astype(np.float32)
                return success - 1.0
            else:
                d_pos, d_rot = self._goal_distance(achieved_goal, goal)
                # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
                # dominated by `d_rot` (in radians).
                return -(0.1 * d_pos + d_rot)

if __name__ =="__main__":
    env = hand(render_mode="human")
    for q in range(30):
        
        state, _ = env.reset()
        env.rot_range=q/10
        env.pos_range=q/500
        for i in range(20):
            state, reward, terminated, truncated, _ = env.step([-1 for _ in range(20)])
            print(reward)