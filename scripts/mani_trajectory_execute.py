#!/usr/bin/env python

# Original work: official beginner's documentation from Moveit
# 
# Link: https://ros-planning.github.io/moveit_tutorials/doc/motion_planning_pipeline/motion_planning_pipeline_tutorial.html

# Author: Acorn Pooley, Mike Lautman


# Modified by: Xun Tu

# The main file to execute a saved trajectory for the manipulator


# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import time
try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

## END_SUB_TUTORIAL


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class MoveGroupPythonInterfaceSimple(object):
    """A simple MoveGroupPythonInterface"""

    def __init__(self):
        super(MoveGroupPythonInterfaceSimple, self).__init__()

        ## Setup
        #
        # First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("mani_trajectory_exec", anonymous=True)

        # Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        # kinematic model and the robot's current joint states

        # Use the default one
        robot = moveit_commander.RobotCommander()

        # Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        # for getting, setting, and updating the robot's internal understanding of the
        # surrounding world:

        # Use a simple, default scene
        scene = moveit_commander.PlanningSceneInterface()

        # Instantiate two `MoveGroupCommander`_ objects.  The object is an interface
        # to a planning group (group of joints).  

        # In this project there are two groups for the manipulator:
        # One is "arm", the robotic arm
        # The other is "gripper", the gripper at the robot's end-effector
        
        # This interface can be used to plan and execute motions:
        group_name1 = "arm"
        move_group_arm = moveit_commander.MoveGroupCommander(group_name1)

        # Same for the gripper
        group_name2 = "gripper"
        move_group_gripper = moveit_commander.MoveGroupCommander(group_name2)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz: (debugging purpose)
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )



        ## Getting Basic Information (Debugging Purpose; de-activated by fault)

        # We can get the name of the reference frame for this robot:
        # planning_frame = move_group.get_planning_frame()
        # print("============ Planning frame: %s" % planning_frame)

        # # We can also print the name of the end-effector link for this group:
        # eef_link = move_group.get_end_effector_link()
        # print("============ End effector link: %s" % eef_link)

        # # We can get a list of all the groups in the robot:
        # group_names = robot.get_group_names()
        # print("============ Available Planning Groups:", robot.get_group_names())

        # # Sometimes for debugging it is useful to print the entire state of the
        # # robot:
        # print("============ Printing robot state")
        # print(robot.get_current_state())
        # print("")


        ## Misc variables
        self.robot = robot
        self.scene = scene
        self.move_group_arm = move_group_arm
        self.move_group_gripper = move_group_gripper

        self.display_trajectory_publisher = display_trajectory_publisher
 
        self.group_names = [group_name1, group_name2]

    def go_to_joint_state_arm(self, joint_goal):
        # Move the robotic arm to the desired joint state

        # joint_goal = self.move_group_arm.get_current_joint_values()
        
        # joint_goal[0] = 0
        # joint_goal[1] = 0
        # joint_goal[2] = 0
        # joint_goal[3] = 0
  

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        self.move_group_arm.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group_arm.stop()

        # For testing:
        current_joints = self.move_group_arm.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.05)

    def set_gripper_width(self, gripper_width):
        # Move the gripper to the desired width
        #
        # WARNING: when trying to grasp something, but 
        # the width is too small, it might damage the motor 
        # and cause the motor to fail
        # 

        # Similarly, use 'go' to control the gripper
        self.move_group_gripper.go([gripper_width, gripper_width], wait=True)

        # Call 'stop'
        self.move_group_gripper.stop()

        # For testing
        current_gripper_width = self.move_group_gripper.get_current_joint_values()
        return all_close([gripper_width, gripper_width], current_gripper_width, 0.05)

    def execute_trajectory_joint(self):
        # The function to move the manipulator along a pre-recorded trajectory
        # The trajectory is defined by several joint states along the path

        ## TODO: create your own trajectory, so that the robot can grasp
        ## the target object along it
        ## You can either continue using the provided baseline method and finetuning it
        ## Or create more complicated trajectories by yourself

        ## Baseline Method: move the gripper near the target, and close it
        self.set_gripper_width(0.012)
        time.sleep(1)
        ## TODO: fill up the trajectory list below with your own recorded joint states
        
        trajectory_list = [[0, 0*pi/180, 0*pi/180, 0*pi/180]]
        for wp in trajectory_list:
            self.go_to_joint_state_arm(wp)
            # Somehow "wait=True" doesn't prevent the system from pausing for a enough long time...
            time.sleep(10)

        ## TODO: set up the gripper width with your own recorded data
 	    # Stay for a while
        time.sleep(0.5)
        # Close the gripper (don't use 0.0 for your case, since it may damage the motor)
        self.set_gripper_width(0.0)
        
        # Optional move the gripper back to zero pose
        # self.go_to_joint_state_arm([0, 0, 0, 0])
        ## Custom Method (Optional)

    #####################################################
    #### The following functions are OPTIONAL        #### 
    ####                                             ####
    #### You can read and use them for more powerful ####
    #### functionalities, such as debugging or do the ###
    #### the motion planning in a new way. But they   ###
    #### are NOT required for the final challenge     ###
    #####################################################

    def go_to_pose_goal_arm(self, pose_goal): #Optional
        # Move the robotic arm to the desired pose

        # An Example of specifying the goal pose with quaternions
        #
        # pose_goal = geometry_msgs.msg.Pose()
        # pose_goal.orientation.w = 1.0
        # pose_goal.position.x = 0.4
        # pose_goal.position.y = 0.1
        # pose_goal.position.z = 0.4

        self.move_group_arm.set_pose_target(pose_goal)

        # Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = self.move_group_arm.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group_arm.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        self.move_group_arm.clear_pose_targets()


        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group_arm.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.05)

    def plan_cartesian_path_arm(self, scale=1): #Optional
        ##
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through. If executing  interactively in a
        ## Python shell, set scale = 1.0.
        ##
        waypoints = []

        wpose = self.move_group_arm.get_current_pose().pose
        wpose.position.z -= scale * 0.1  # First move up (z)
        wpose.position.y += scale * 0.2  # and sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.y -= scale * 0.1  # Third move sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = self.move_group_arm.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )  # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return plan, fraction

        ## END_SUB_TUTORIAL

    def display_trajectory(self, plan): # Optional
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        ##
        ## Displaying a Trajectory
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
        ## group.plan() method does this automatically so this is not that useful
        ## here (it just displays the same trajectory again):
        ##
        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        display_trajectory_publisher.publish(display_trajectory)

    def execute_plan_arm(self, plan): #Optional
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group_arm

        ## BEGIN_SUB_TUTORIAL execute_plan
        ##
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        move_group.execute(plan, wait=True)

        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail





def main():
    try:
        print("")
        print("----------------------------------------------------------")
        print("Welcome to the Turtlebot3 Manipulator Control")
        print("Source: MoveIt MoveGroup Python Interface Tutorial")
        print("----------------------------------------------------------")
        print("Press Ctrl-D to exit at any time")
        print("")
        input(
            "============ Press `Enter` to start ===================="
        )
        tutlebot3 = MoveGroupPythonInterfaceSimple()

        input(
            "============ Press `Enter` to execute a movement following a trajectory with joint states"
        )
        tutlebot3.execute_trajectory_joint()

        # input("============ Press `Enter` to execute a movement using a pose goal ...")
        # tutorial.go_to_pose_goal()

        # input("============ Press `Enter` to plan and display a Cartesian path ...")
        # cartesian_plan, fraction = tutorial.plan_cartesian_path()

        # input(
        #     "============ Press `Enter` to display a saved trajectory (this will replay the Cartesian path)  ..."
        # )
        # tutorial.display_trajectory(cartesian_plan)

        # input("============ Press `Enter` to execute a saved path ...")
        # tutorial.execute_plan(cartesian_plan)



        print("============ Manipulation complete!")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()

## BEGIN_TUTORIAL
## .. _moveit_commander:
##    http://docs.ros.org/noetic/api/moveit_commander/html/namespacemoveit__commander.html
##
## .. _MoveGroupCommander:
##    http://docs.ros.org/noetic/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html
##
## .. _RobotCommander:
##    http://docs.ros.org/noetic/api/moveit_commander/html/classmoveit__commander_1_1robot_1_1RobotCommander.html
##
## .. _PlanningSceneInterface:
##    http://docs.ros.org/noetic/api/moveit_commander/html/classmoveit__commander_1_1planning__scene__interface_1_1PlanningSceneInterface.html
##
## .. _DisplayTrajectory:
##    http://docs.ros.org/noetic/api/moveit_msgs/html/msg/DisplayTrajectory.html
##
## .. _RobotTrajectory:
##    http://docs.ros.org/noetic/api/moveit_msgs/html/msg/RobotTrajectory.html
##
## .. _rospy:
##    http://docs.ros.org/noetic/api/rospy/html/
## END_TUTORIAL
