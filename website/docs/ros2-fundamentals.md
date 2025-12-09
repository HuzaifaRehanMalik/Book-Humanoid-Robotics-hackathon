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

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software that provides services designed for a heterogeneous computer cluster. Unlike ROS 1, which was built on a centralized architecture, ROS 2 is built on a distributed architecture using the Data Distribution Service (DDS) middleware. This makes it more suitable for complex humanoid robotics applications that require real-time performance, multi-robot systems, and safety-critical operations.

ROS 2 has become the de facto standard for humanoid robotics development due to its improved real-time capabilities, enhanced security features, and better support for commercial deployment. This chapter will explore the fundamental concepts of ROS 2 with a focus on humanoid robotics applications.

## ROS 2 Architecture

### DDS Middleware

The core of ROS 2's architecture is the Data Distribution Service (DDS) middleware, which provides:

- **Decentralized communication**: No single point of failure
- **Real-time performance**: Deterministic message delivery
- **Quality of Service (QoS)**: Configurable reliability and performance parameters
- **Language and platform independence**: Support for multiple programming languages

### Nodes and Processes

In ROS 2, nodes are lightweight processes that communicate with each other through topics, services, and actions. Each node runs in its own process space, which improves robustness compared to ROS 1.

```python
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.get_logger().info('Humanoid Controller Node Started')
```

## Communication Patterns

### Topics (Publish/Subscribe)

Topics provide asynchronous, many-to-many communication between nodes. This is ideal for sensor data distribution and state updates in humanoid robots.

```python
from std_msgs.msg import Float64MultiArray

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(Float64MultiArray, 'joint_commands', 10)
        self.timer = self.create_timer(0.01, self.publish_joint_commands)  # 100Hz

    def publish_joint_commands(self):
        msg = Float64MultiArray()
        # Fill with joint commands for humanoid robot
        self.publisher.publish(msg)
```

<!-- Figure removed: ROS 2 Topic Communication image not available -->

### Services (Request/Response)

Services provide synchronous, one-to-one communication with request/response patterns. Useful for configuration, calibration, and control mode switching.

```python
from example_interfaces.srv import SetBool

class ControllerManager(Node):
    def __init__(self):
        super().__init__('controller_manager')
        self.srv = self.create_service(SetBool, 'enable_controller', self.enable_callback)

    def enable_callback(self, request, response):
        # Enable/disable controller based on request
        response.success = True
        response.message = 'Controller enabled'
        return response
```

### Actions (Goal/Feedback/Result)

Actions provide asynchronous communication with feedback for long-running tasks, perfect for humanoid robot behaviors:

```python
from rclpy.action import ActionServer
from control_msgs.action import FollowJointTrajectory

class TrajectoryExecutor(Node):
    def __init__(self):
        super().__init__('trajectory_executor')
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'follow_joint_trajectory',
            self.execute_trajectory)

    def execute_trajectory(self, goal_handle):
        # Execute joint trajectory for humanoid robot
        feedback = FollowJointTrajectory.Feedback()
        result = FollowJointTrajectory.Result()

        # Implementation here
        goal_handle.succeed()
        return result
```

## Quality of Service (QoS) in Humanoid Robotics

QoS profiles are crucial for humanoid robotics applications where timing and reliability requirements vary:

- **Reliable**: For critical control commands
- **Best Effort**: For sensor data where some loss is acceptable
- **Keep Last**: For state updates where only the most recent value matters
- **Keep All**: For logging and debugging

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# For critical safety commands
safety_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)

# For sensor data
sensor_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST
)
```

## Launch Systems and Compositions

ROS 2 provides powerful launch systems for managing complex humanoid robot applications:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='humanoid_controller',
            executable='joint_state_publisher',
            name='joint_state_publisher'
        ),
        Node(
            package='humanoid_controller',
            executable='trajectory_executor',
            name='trajectory_executor'
        )
    ])
```

## Package Management and Build Systems

ROS 2 uses colcon as its build system, supporting multiple build types:

- **ament_cmake**: For C++ packages
- **ament_python**: For Python packages
- **ament_ccmake**: For mixed C/C++ packages

## Security in ROS 2

Security is paramount for humanoid robots operating in human environments:

- **Authentication**: Verifying node identity
- **Authorization**: Controlling access to topics/services
- **Encryption**: Protecting data in transit

## ROS 2 in Humanoid Robotics Applications

### Middleware Integration

ROS 2 serves as the backbone for humanoid robot middleware, connecting perception, planning, and control systems:

- **Perception stack**: Sensor processing and environment understanding
- **Planning stack**: Motion planning and task scheduling
- **Control stack**: Low-level actuator control and feedback

### Multi-Robot Systems

ROS 2's distributed architecture enables coordination of multiple humanoid robots:

- Shared coordinate frames
- Distributed perception
- Collaborative task execution

## Best Practices for Humanoid Robotics

### Performance Considerations

- Use appropriate QoS settings for different data types
- Implement efficient serialization for high-frequency data
- Monitor network performance in distributed systems

### Safety and Reliability

- Implement proper error handling and recovery
- Use latching for critical state topics
- Implement timeouts for blocking operations

### Code Organization

- Follow ROS 2 style guides
- Use composition over multiple executables when possible
- Implement proper logging and diagnostics

## Exercises and Labs

### Exercise 1: Basic Publisher/Subscriber

Create a simple publisher and subscriber pair that simulates joint state publishing for a humanoid robot.

### Exercise 2: Service Implementation

Implement a service that accepts a humanoid robot pose and returns whether it's within safe joint limits.

### Lab Activity: ROS 2 Navigation Stack

Set up and configure the ROS 2 Navigation Stack for a humanoid robot simulation environment.

## Summary

ROS 2 provides the essential infrastructure for developing complex humanoid robotics applications. Its distributed architecture, quality of service features, and security capabilities make it well-suited for the demanding requirements of humanoid robots operating in human environments. Understanding these fundamentals is crucial for building robust and reliable humanoid robot systems.

## Exercises and Labs

### Exercise 1: Node Communication
Create a publisher node that publishes joint position commands for a humanoid robot's left arm, and a subscriber node that logs these commands to verify communication.

### Exercise 2: Service Implementation
Implement a ROS 2 service that accepts a humanoid robot pose and returns whether it's within safe joint limits.

### Lab Activity: ROS 2 Navigation Stack
Configure and test the ROS 2 Navigation Stack for a humanoid robot simulation environment, focusing on the integration between different packages.

### Exercise 3: QoS Configuration
Experiment with different Quality of Service policies for critical control topics versus sensor data topics in a simulated humanoid robot system, and analyze the impact on performance and reliability.

## Further Reading

- ROS 2 Documentation: https://docs.ros.org/en/humble/
- Design ROS 2 Concepts: https://design.ros2.org/
- ROS 2 for Real-time Systems: Real-time performance considerations

## References

- Lalanda, P., & Haudot, V. (2023). ROS 2 for Robotics: A Practical Introduction. Academic Press.
- Quigley, M., et al. (2023). Programming Robots with ROS: A Practical Introduction to the Robot Operating System. O'Reilly Media.
- ROS 2 Documentation Working Group. (2023). ROS 2 User Guide. Open Robotics. https://docs.ros.org

## Discussion Questions

1. How does the DDS middleware in ROS 2 improve the reliability of humanoid robot systems compared to ROS 1's master-based architecture?
2. What QoS policies would you recommend for critical safety-related topics in a humanoid robot, and why?
3. How would you design a ROS 2 system architecture for coordinating multiple humanoid robots in a shared workspace?