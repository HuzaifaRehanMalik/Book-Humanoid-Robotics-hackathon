---
id: ros2-fundamentals
title: ROS 2 Fundamentals for Humanoid Robotics
slug: /ros2-fundamentals
---

# ROS 2 Fundamentals for Humanoid Robotics

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the core concepts and architecture of ROS 2
- Explain the differences between ROS 1 and ROS 2
- Implement basic communication patterns in ROS 2
- Apply ROS 2 concepts specifically to humanoid robotics applications

## Introduction

Robot Operating System 2 (ROS 2) represents a significant evolution from its predecessor, designed specifically to address the challenges of modern robotics applications, particularly in humanoid robotics where safety, real-time performance, and distributed computing are critical. Unlike ROS 1, which was built on a centralized architecture with a single master node, ROS 2 is built on a distributed architecture using the Data Distribution Service (DDS) middleware (Quigley et al., 2023). This architectural shift makes ROS 2 more suitable for complex humanoid robotics applications that require real-time performance, multi-robot systems, and safety-critical operations.

ROS 2 has become the de facto standard for humanoid robotics development due to its improved real-time capabilities, enhanced security features, and better support for commercial deployment. This chapter will explore the fundamental concepts of ROS 2 with a focus on humanoid robotics applications, providing both theoretical understanding and practical implementation guidance.

## ROS 2 Architecture

### DDS Middleware

The core of ROS 2's architecture is the Data Distribution Service (DDS) middleware, which provides several critical capabilities for humanoid robotics:

- **Decentralized communication**: No single point of failure, enhancing system robustness
- **Real-time performance**: Deterministic message delivery with configurable timing guarantees
- **Quality of Service (QoS)**: Configurable reliability and performance parameters tailored to specific application needs
- **Language and platform independence**: Support for multiple programming languages and operating systems

The DDS middleware enables ROS 2 to function in distributed environments where humanoid robots may have multiple computing nodes, sensors, and actuators that need to communicate reliably in real-time (Lalanda & Haudot, 2023).

### Nodes and Processes

In ROS 2, nodes are lightweight processes that communicate with each other through topics, services, and actions. Each node runs in its own process space, which improves robustness compared to ROS 1 by providing process isolation. This is particularly important for humanoid robots where a failure in one component (e.g., vision processing) should not affect critical systems (e.g., balance control).

```python
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.get_logger().info('Humanoid Controller Node Started')

        # Initialize humanoid-specific parameters
        self.declare_parameter('robot_name', 'default_humanoid')
        self.declare_parameter('control_frequency', 100)  # Hz
```

## Communication Patterns

### Topics (Publish/Subscribe)

Topics provide asynchronous, many-to-many communication between nodes. This pattern is ideal for sensor data distribution and state updates in humanoid robots, where multiple systems may need to consume the same information simultaneously.

```python
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Publisher for joint commands
        self.joint_cmd_publisher = self.create_publisher(
            Float64MultiArray,
            'joint_commands',
            10
        )

        # Publisher for joint states
        self.joint_state_publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Timer for publishing at 100Hz
        self.timer = self.create_timer(0.01, self.publish_joint_data)

    def publish_joint_data(self):
        # Create and publish joint command message
        cmd_msg = Float64MultiArray()
        # Fill with joint commands for humanoid robot
        self.joint_cmd_publisher.publish(cmd_msg)

        # Create and publish joint state message
        state_msg = JointState()
        state_msg.name = ['joint1', 'joint2', 'joint3']  # Example joint names
        state_msg.position = [0.0, 0.0, 0.0]  # Example positions
        self.joint_state_publisher.publish(state_msg)
```

<!-- <div class="textbook-figure">

![ROS 2 Topic Communication](/img/ros2-topic-communication.png)

**Figure 1.** ROS 2 topic communication model showing multiple publishers and subscribers exchanging messages through the DDS middleware. In humanoid robotics applications, this enables distributed processing of sensor data and coordinated control of multiple subsystems.

</div> -->

### Services (Request/Response)

Services provide synchronous, one-to-one communication with request/response patterns. This communication pattern is useful for configuration, calibration, and control mode switching in humanoid robots where a definitive response is required.

```python
from example_interfaces.srv import SetBool
from std_srvs.srv import Trigger

class ControllerManager(Node):
    def __init__(self):
        super().__init__('controller_manager')

        # Service for enabling/disabling controllers
        self.enable_srv = self.create_service(
            SetBool,
            'enable_controller',
            self.enable_callback
        )

        # Service for resetting the robot
        self.reset_srv = self.create_service(
            Trigger,
            'reset_robot',
            self.reset_callback
        )

    def enable_callback(self, request, response):
        """Enable or disable controller based on request."""
        if request.data:
            self.get_logger().info('Controller enabled')
            response.success = True
            response.message = 'Controller enabled successfully'
        else:
            self.get_logger().info('Controller disabled')
            response.success = True
            response.message = 'Controller disabled successfully'

        return response

    def reset_callback(self, request, response):
        """Reset humanoid robot to safe state."""
        try:
            # Implement reset logic
            self.get_logger().info('Resetting humanoid robot to safe state')
            response.success = True
            response.message = 'Robot reset successfully'
        except Exception as e:
            response.success = False
            response.message = f'Reset failed: {str(e)}'

        return response
```

### Actions (Goal/Feedback/Result)

Actions provide asynchronous communication with feedback for long-running tasks, making them perfect for humanoid robot behaviors that may take seconds or minutes to complete. Actions are particularly useful for complex movements, navigation tasks, and manipulation sequences.

```python
from rclpy.action import ActionServer, ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class TrajectoryExecutor(Node):
    def __init__(self):
        super().__init__('trajectory_executor')

        # Action server for executing joint trajectories
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'follow_joint_trajectory',
            self.execute_trajectory
        )

        # Joint trajectory publisher for real-time control
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory,
            'joint_trajectory',
            10
        )

    def execute_trajectory(self, goal_handle):
        """Execute joint trajectory for humanoid robot with feedback."""
        self.get_logger().info('Executing joint trajectory...')

        # Get the trajectory from the goal
        trajectory = goal_handle.request.trajectory

        # Initialize feedback
        feedback = FollowJointTrajectory.Feedback()
        result = FollowJointTrajectory.Result()

        # Execute trajectory points
        for i, point in enumerate(trajectory.points):
            # Update feedback
            feedback.actual.positions = point.positions
            feedback.desired = point
            feedback.progress = float(i) / len(trajectory.points)

            # Publish feedback
            goal_handle.publish_feedback(feedback)

            # Simulate execution time
            self.get_clock().sleep_for(Duration(seconds=0.1))

            # Check for cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                return result

        # Complete successfully
        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result
```

<div class="textbook-definition">

**Definition:** Actions in ROS 2 are a communication pattern that extends services to support long-running operations with continuous feedback. They are ideal for humanoid robotics tasks that require monitoring of progress, such as walking, grasping, or complex manipulation sequences.

</div>

## Quality of Service (QoS) in Humanoid Robotics

QoS profiles are crucial for humanoid robotics applications where timing and reliability requirements vary significantly across different types of data. ROS 2 provides configurable QoS settings that allow developers to specify how messages should be handled in terms of reliability, durability, and history.

- **Reliable**: For critical control commands where message loss is unacceptable
- **Best Effort**: For sensor data where some loss is acceptable to maintain real-time performance
- **Keep Last**: For state updates where only the most recent value matters
- **Keep All**: For logging and debugging where all messages must be preserved

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class HumanoidQoSProfiles:
    """QoS profiles optimized for humanoid robotics applications."""

    # For critical safety commands
    SAFETY_COMMANDS = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        durability=DurabilityPolicy.TRANSIENT_LOCAL
    )

    # For sensor data (IMU, encoders, etc.)
    SENSOR_DATA = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST
    )

    # For state updates (joint positions, robot pose)
    STATE_UPDATES = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST
    )

    # For logging and debugging
    LOGGING = QoSProfile(
        depth=100,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_ALL,
        durability=DurabilityPolicy.TRANSIENT_LOCAL
    )

# Example usage
class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers with appropriate QoS
        self.safety_pub = self.create_publisher(
            Bool,
            'safety_override',
            qos_profile=HumanoidQoSProfiles.SAFETY_COMMANDS
        )

        self.sensor_pub = self.create_publisher(
            JointState,
            'joint_states',
            qos_profile=HumanoidQoSProfiles.SENSOR_DATA
        )
```

<div class="textbook-theorem">

**Theorem:** Proper QoS configuration in ROS 2 systems for humanoid robotics ensures that critical safety and control messages are delivered reliably while maintaining real-time performance for sensor data processing. The choice of QoS parameters directly impacts system performance and safety characteristics.

</div>

## Launch Systems and Compositions

ROS 2 provides powerful launch systems for managing complex humanoid robot applications. The launch system allows for coordinated startup, configuration, and management of multiple nodes that comprise a complete humanoid robot system.

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    """Launch description for a basic humanoid robot system."""

    # Declare launch arguments
    robot_name_launch_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_humanoid',
        description='Name of the robot'
    )

    return LaunchDescription([
        robot_name_launch_arg,

        # Composable node container for real-time critical nodes
        ComposableNodeContainer(
            name='humanoid_core_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                ComposableNode(
                    package='humanoid_control',
                    plugin='humanoid_control::JointStatePublisher',
                    name='joint_state_publisher'
                ),
                ComposableNode(
                    package='humanoid_control',
                    plugin='humanoid_control::BalanceController',
                    name='balance_controller'
                ),
            ],
            output='screen',
        ),

        # Standalone nodes for non-real-time processing
        Node(
            package='humanoid_perception',
            executable='vision_node',
            name='vision_node',
            parameters=[
                {'robot_name': LaunchConfiguration('robot_name')}
            ]
        ),

        Node(
            package='humanoid_ui',
            executable='command_interface',
            name='command_interface',
            parameters=[
                {'robot_name': LaunchConfiguration('robot_name')}
            ]
        )
    ])
```

## Package Management and Build Systems

ROS 2 uses colcon as its build system, supporting multiple build types optimized for different development needs:

- **ament_cmake**: For C++ packages with high-performance requirements
- **ament_python**: For Python packages with rapid prototyping needs
- **ament_ccmake**: For mixed C/C++ packages requiring integration of both languages

The build system also supports cross-compilation for different hardware platforms commonly used in humanoid robots, from embedded systems to high-performance computing nodes.

## Security in ROS 2

Security is paramount for humanoid robots operating in human environments. ROS 2 provides comprehensive security features:

- **Authentication**: Verifying node identity through certificates and keys
- **Authorization**: Controlling access to topics, services, and actions
- **Encryption**: Protecting data in transit between nodes

These security features are essential for humanoid robots that may operate in public spaces or handle sensitive information.

## ROS 2 in Humanoid Robotics Applications

### Middleware Integration

ROS 2 serves as the backbone for humanoid robot middleware, connecting perception, planning, and control systems:

- **Perception stack**: Sensor processing, environment understanding, and state estimation
- **Planning stack**: Motion planning, path planning, and task scheduling
- **Control stack**: Low-level actuator control, feedback control, and safety systems

### Multi-Robot Systems

ROS 2's distributed architecture enables coordination of multiple humanoid robots through:

- Shared coordinate frames and transforms
- Distributed perception and mapping
- Collaborative task execution and load balancing
- Inter-robot communication and coordination

## Advanced ROS 2 Concepts for Humanoid Robotics

### Real-Time Systems and Deterministic Behavior

Real-time performance is critical for humanoid robots, especially for control systems that must maintain balance and respond to environmental changes within strict timing constraints.

#### Real-Time Scheduling

```python
import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
from control_msgs.msg import JointTrajectoryControllerState
import time

class RealTimeHumanoidController(Node):
    def __init__(self):
        super().__init__('realtime_humanoid_controller')

        # Set up real-time publisher with high frequency
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectoryControllerState,
            'joint_commands',
            QoSProfile(depth=1, reliability=1, durability=2)  # RELIABLE, TRANSIENT_LOCAL
        )

        # Create timer for real-time control loop (1000Hz for critical control)
        self.control_timer = self.create_timer(
            0.001,  # 1ms = 1000Hz
            self.real_time_control_callback,
            clock=self.get_clock()
        )

        # Track timing performance
        self.last_callback_time = self.get_clock().now()

    def real_time_control_callback(self):
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_callback_time).nanoseconds / 1e9

        # Log timing jitter
        if time_diff > 0.002:  # More than 2ms delay
            self.get_logger().warn(f'Timing jitter detected: {time_diff:.4f}s')

        self.last_callback_time = current_time

        # Perform critical control calculations here
        self.perform_control_step()

    def perform_control_step(self):
        """Implement critical control logic with deterministic timing"""
        # Balance control, joint position updates, etc.
        pass
```

#### Real-Time Optimizations

- **CPU Affinity**: Pin critical nodes to specific CPU cores
- **Memory Locking**: Prevent critical data from being swapped to disk
- **Priority Scheduling**: Use SCHED_FIFO for real-time threads
- **Deterministic Algorithms**: Avoid algorithms with variable execution time

### Advanced Launch Systems

Complex humanoid robots require sophisticated launch configurations that can handle multiple subsystems:

```python
# advanced_humanoid_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='my_humanoid',
        description='Name of the robot'
    )

    enable_vision = DeclareLaunchArgument(
        'enable_vision',
        default_value='true',
        description='Enable vision processing nodes'
    )

    # Composable container for real-time critical nodes
    critical_container = ComposableNodeContainer(
        name='critical_control_container',
        namespace=LaunchConfiguration('robot_name'),
        package='rclcpp_components',
        executable='component_container_mt',  # Multi-threaded container
        composable_node_descriptions=[
            ComposableNode(
                package='humanoid_control',
                plugin='humanoid_control::BalanceController',
                name='balance_controller',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'control_frequency': 1000,  # 1000Hz
                    'kp': 10.0,
                    'kd': 1.0
                }]
            ),
            ComposableNode(
                package='humanoid_control',
                plugin='humanoid_control::JointStatePublisher',
                name='joint_state_publisher',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'publish_frequency': 500  # 500Hz
                }]
            ),
        ],
        output='screen',
    )

    # Separate nodes for non-critical processing
    perception_nodes = [
        Node(
            package='humanoid_perception',
            executable='object_detector',
            name='object_detector',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                os.path.join(get_package_share_directory('humanoid_perception'), 'config', 'object_detection.yaml')
            ],
            condition=IfCondition(LaunchConfiguration('enable_vision')),
            respawn=True,
            respawn_delay=2
        ),
        Node(
            package='humanoid_perception',
            executable='person_tracker',
            name='person_tracker',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            condition=IfCondition(LaunchConfiguration('enable_vision')),
        )
    ]

    # Launch nodes with staggered startup to avoid resource conflicts
    launch_description = LaunchDescription([
        use_sim_time,
        robot_name,
        enable_vision,
        critical_container,
    ])

    # Add perception nodes with delay
    for i, node in enumerate(perception_nodes):
        launch_description.add_action(
            TimerAction(
                period=2.0 * (i + 1),  # Stagger startup by 2 seconds
                actions=[node]
            )
        )

    return launch_description
```

### ROS 2 for Multi-Robot Systems

Humanoid robots often operate in teams or alongside other robots:

```python
# multi_humanoid_coordinator.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from multi_robot_msgs.msg import RobotStatus, CoordinationCommand

class MultiHumanoidCoordinator(Node):
    def __init__(self):
        super().__init__('multi_humanoid_coordinator')

        # Subscribe to status from all robots
        self.robot_statuses = {}
        self.robot_subscribers = []

        # Robot IDs for the team
        self.robot_ids = ['humanoid_01', 'humanoid_02', 'humanoid_03']

        for robot_id in self.robot_ids:
            status_sub = self.create_subscription(
                RobotStatus,
                f'/{robot_id}/status',
                lambda msg, rid=robot_id: self.robot_status_callback(msg, rid),
                10
            )
            self.robot_subscribers.append(status_sub)

        # Publisher for coordination commands
        self.coordination_pub = self.create_publisher(
            CoordinationCommand,
            'coordination_commands',
            10
        )

        # Timer for coordination logic
        self.coordination_timer = self.create_timer(
            0.1,  # 10Hz coordination
            self.coordination_callback
        )

    def robot_status_callback(self, msg, robot_id):
        """Update status of a specific robot"""
        self.robot_statuses[robot_id] = {
            'position': msg.position,
            'task': msg.current_task,
            'battery': msg.battery_level,
            'status': msg.robot_status
        }

    def coordination_callback(self):
        """Implement multi-robot coordination logic"""
        # Example: Task allocation based on robot capabilities and positions
        available_robots = [
            rid for rid, status in self.robot_statuses.items()
            if status['status'] == 'IDLE'
        ]

        # Allocate tasks based on proximity and capability
        if available_robots:
            # Simple round-robin task allocation
            task = self.get_next_task()
            if task:
                target_robot = available_robots[0]  # Simple allocation
                self.assign_task_to_robot(target_robot, task)

    def get_next_task(self):
        """Get the next task from a task queue"""
        # Implementation would depend on task management system
        return None

    def assign_task_to_robot(self, robot_id, task):
        """Send task assignment to specific robot"""
        cmd = CoordinationCommand()
        cmd.target_robot = robot_id
        cmd.task_description = task
        cmd.command_type = 'TASK_ASSIGNMENT'

        self.coordination_pub.publish(cmd)
```

### Performance Monitoring and Diagnostics

Monitoring system performance is crucial for humanoid robots:

```python
# diagnostic_monitor.py
import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Float64

class HumanoidDiagnostics(Node):
    def __init__(self):
        super().__init__('humanoid_diagnostics')

        # Publisher for diagnostic messages
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        # Publishers for specific metrics
        self.cpu_usage_pub = self.create_publisher(Float64, 'system/cpu_usage', 10)
        self.memory_usage_pub = self.create_publisher(Float64, 'system/memory_usage', 10)

        # Timer for diagnostic collection
        self.diag_timer = self.create_timer(1.0, self.collect_diagnostics)

        # Track performance metrics
        self.metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'network_latency': 0.0,
            'control_loop_jitter': 0.0
        }

    def collect_diagnostics(self):
        """Collect system diagnostics and publish"""
        # Collect system metrics (implementation would interface with system monitoring)
        self.metrics['cpu_usage'] = self.get_cpu_usage()
        self.metrics['memory_usage'] = self.get_memory_usage()

        # Create diagnostic message
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # Create status for overall system
        system_status = DiagnosticStatus()
        system_status.name = 'Humanoid Robot System'
        system_status.level = DiagnosticStatus.OK
        system_status.message = 'All systems nominal'

        # Add key-value pairs for metrics
        system_status.values = [
            KeyValue(key='CPU Usage (%)', value=f"{self.metrics['cpu_usage']:.2f}"),
            KeyValue(key='Memory Usage (%)', value=f"{self.metrics['memory_usage']:.2f}"),
            KeyValue(key='Control Loop Jitter (ms)', value=f"{self.metrics['control_loop_jitter']:.2f}")
        ]

        # Check for issues and update status level if needed
        if self.metrics['cpu_usage'] > 90.0:
            system_status.level = DiagnosticStatus.WARN
            system_status.message = 'High CPU usage detected'
        elif self.metrics['memory_usage'] > 95.0:
            system_status.level = DiagnosticStatus.ERROR
            system_status.message = 'Memory usage critical'

        diag_array.status.append(system_status)
        self.diag_pub.publish(diag_array)

        # Publish individual metrics
        self.cpu_usage_pub.publish(Float64(data=self.metrics['cpu_usage']))
        self.memory_usage_pub.publish(Float64(data=self.metrics['memory_usage']))

    def get_cpu_usage(self):
        """Get current CPU usage"""
        # Implementation would use system tools like psutil
        import psutil
        return psutil.cpu_percent(interval=None)

    def get_memory_usage(self):
        """Get current memory usage"""
        import psutil
        return psutil.virtual_memory().percent
```

### Advanced Communication Patterns

#### Lifecycle Nodes for Better Resource Management

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

class LifecycleHumanoidController(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_humanoid_controller')
        self.joint_pub = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the node"""
        self.get_logger().info(f'Configuring {self.get_name()}')

        # Create publisher only when configured
        self.joint_pub = self.create_publisher(
            JointState,
            'joint_states',
            QoSProfile(depth=10)
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate the node"""
        self.get_logger().info(f'Activating {self.get_name()}')

        # Activate publisher
        self.joint_pub.on_activate()

        # Create timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_callback)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivate the node"""
        self.get_logger().info(f'Deactivating {self.get_name()}')

        # Deactivate publisher
        self.joint_pub.on_deactivate()

        # Destroy timer
        self.destroy_timer(self.control_timer)

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Clean up the node"""
        self.get_logger().info(f'Cleaning up {self.get_name()}')

        # Destroy publisher
        self.destroy_publisher(self.joint_pub)
        self.joint_pub = None

        return TransitionCallbackReturn.SUCCESS

    def control_callback(self):
        """Control loop callback"""
        # Publish joint states
        msg = JointState()
        msg.name = ['joint1', 'joint2', 'joint3']
        msg.position = [0.0, 0.0, 0.0]
        self.joint_pub.publish(msg)
```

## Best Practices for Humanoid Robotics

### Performance Considerations

- Use appropriate QoS settings for different data types to balance reliability and performance
- Implement efficient serialization for high-frequency data like sensor readings
- Monitor network performance in distributed systems to identify bottlenecks
- Use composable nodes for real-time critical applications to reduce communication overhead
- Implement real-time scheduling for critical control loops
- Optimize message sizes to reduce network overhead
- Use efficient data structures for real-time processing

### Safety and Reliability

- Implement proper error handling and graceful degradation strategies
- Use latching for critical state topics that must persist across node restarts
- Implement timeouts for blocking operations to prevent system hangs
- Design fail-safe mechanisms that bring the robot to a safe state when errors occur
- Use lifecycle nodes for better resource management
- Implement comprehensive diagnostics and monitoring
- Design for fault tolerance with redundant systems

### Code Organization

- Follow ROS 2 style guides and naming conventions for consistency
- Use composition over multiple executables when real-time performance is critical
- Implement comprehensive logging and diagnostics for debugging and monitoring
- Structure packages around functional components rather than programming languages
- Use parameter files for configuration management
- Implement proper testing strategies (unit, integration, system)
- Document interfaces and data flow clearly

## Exercises and Labs

### Exercise 1: Node Communication
Create a publisher node that publishes joint position commands for a humanoid robot's left arm, and a subscriber node that logs these commands to verify communication. Implement proper QoS settings for real-time performance.

### Exercise 2: Service Implementation
Implement a ROS 2 service that accepts a humanoid robot pose and returns whether it's within safe joint limits. Include proper error handling and validation.

### Lab Activity: ROS 2 Navigation Stack
Configure and test the ROS 2 Navigation Stack for a humanoid robot simulation environment, focusing on the integration between different packages and the use of appropriate QoS settings.

### Exercise 3: QoS Configuration
Experiment with different Quality of Service policies for critical control topics versus sensor data topics in a simulated humanoid robot system, and analyze the impact on performance and reliability.

## Summary

ROS 2 provides the essential infrastructure for developing complex humanoid robotics applications. Its distributed architecture, quality of service features, and security capabilities make it well-suited for the demanding requirements of humanoid robots operating in human environments. Understanding these fundamentals is crucial for building robust and reliable humanoid robot systems that can operate safely and effectively.

The architecture of ROS 2, built on DDS middleware, enables the development of distributed, real-time systems that can scale from single robots to multi-robot systems. The various communication patterns—topics, services, and actions—provide the flexibility needed to implement complex humanoid behaviors while maintaining system reliability and safety.

## Discussion Questions

1. How does the DDS middleware in ROS 2 improve the reliability of humanoid robot systems compared to ROS 1's master-based architecture?
2. What QoS policies would you recommend for critical safety-related topics in a humanoid robot, and why?
3. How would you design a ROS 2 system architecture for coordinating multiple humanoid robots in a shared workspace?

## References

Lalanda, P., & Haudot, V. (2023). ROS 2 for Robotics: A Practical Introduction. Academic Press.

Quigley, M., Gerkey, B., & Smart, W. D. (2023). Programming Robots with ROS: A Practical Introduction to the Robot Operating System. O'Reilly Media.

ROS 2 Documentation Working Group. (2023). ROS 2 User Guide. Open Robotics. https://docs.ros.org

Santos, J. M., & Costa, L. (2022). Middleware for Humanoid Robotics: A Survey of ROS and ROS 2. *Journal of Intelligent & Robotic Systems*, 105(1), 1-25.

Zhang, L., Kumar, A., & Liu, H. (2023). Quality of Service in ROS 2 for Real-time Humanoid Robotics Applications. *IEEE Robotics and Automation Letters*, 8(4), 2345-2352.