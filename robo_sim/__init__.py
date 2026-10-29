"""ROBO-SIM: interactive two-link robotic arm simulation and motion planning.

The computational core is deliberately independent of the web interface, so
every module here can be imported, tested and scripted without Streamlit.

Suggested reading order for the mathematics:
    kinematics.py -> geometry.py -> collision.py -> path_planning.py
"""

__version__ = "1.0.0"
