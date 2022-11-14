# Via-Point based Stochastic Trajectory Optimization

This repository contains python code for hands-on numerical, gradient-free, time-optimal trajectory optimization.

Features:
- Implement your objective as a function of positions, velocities, accelerations and total duration of the trajectory, **no gradient** needed.
- Define **velocity and acceleration limits**, the solution is internally constrained to those (no extra cost term needed).
- **Avoid local minima** through the exploration characteristics of VP-STO due to CMA-ES as underlying optimization technique.
- **Choose to fix** initial position, final position, initial velocity, final velocity or any combination of those (no extra cost term needed).
- Use a low resolution for fast optimization, and a high resolution for the final solution due to the **time-continuous trajectory representation**.
- **Linear scaling** of the computational complexity with the number of DoFs.

![Sampling Banner](media/sampling_banner.gif)
*The process of approximating time-optimal trajectories in cluttered environments (check out the example code for reproducing the results).*

---
### Dependencies

The optimization algorithm only depends on [numpy](https://numpy.org) and [pycma](https://github.com/CMA-ES/pycma). Those can be installed by

    pip install numpy
    pip install git+https://github.com/CMA-ES/pycma.git@master

Optional: The example notebooks additionally depend on matplotlib.
