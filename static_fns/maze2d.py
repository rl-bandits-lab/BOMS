import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        target = -1000 * np.zeros((2,), dtype=np.float32)
        solve_thresh = 0.1
        vel_thresh = 0.1
        waypoint_prev_loc = obs[:,0:2]
        
        location = next_obs[:,0:2]
        velocity = next_obs[:,2:4]

        vel = waypoint_prev_loc - location
        vel_norm = np.linalg.norm(vel, axis=1)
        dist = np.linalg.norm(location - target, axis=1)
        
        not_done = (dist >= solve_thresh) \
                    + (vel_norm >= vel_thresh)
        
        done = ~not_done
        done = done[:,None]
        return done
