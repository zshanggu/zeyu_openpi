// Moves the Franka Panda to a home joint configuration with a smooth (quintic) joint-space trajectory.
//
// Usage: go_home [robot_ip] [q1 ... q7]
//   robot_ip defaults to 172.16.0.2; the goal defaults to the standard Franka "ready" pose.
// Environment: GO_HOME_MAX_VEL caps joint speed in rad/s (default 0.3).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <franka/exception.h>
#include <franka/robot.h>

namespace {

constexpr std::array<double, 7> kDefaultHome = {0.0, -M_PI_4, 0.0, -3.0 * M_PI_4, 0.0, M_PI_2, M_PI_4};

void PrintJoints(const std::string& label, const std::array<double, 7>& q) {
  std::cout << label;
  for (double v : q) {
    std::cout << " " << v;
  }
  std::cout << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  std::string robot_ip = argc > 1 ? argv[1] : "172.16.0.2";
  std::array<double, 7> q_goal = kDefaultHome;
  if (argc == 9) {
    for (size_t i = 0; i < 7; i++) {
      q_goal[i] = std::stod(argv[i + 2]);
    }
  } else if (argc > 2) {
    std::cerr << "Usage: " << argv[0] << " [robot_ip] [q1 ... q7]" << std::endl;
    return 1;
  }
  const char* max_vel_env = std::getenv("GO_HOME_MAX_VEL");
  const double max_vel = max_vel_env ? std::stod(max_vel_env) : 0.3;

  auto duration_from = [&](const std::array<double, 7>& q_now) {
    double max_delta = 0.0;
    for (size_t i = 0; i < 7; i++) {
      max_delta = std::max(max_delta, std::abs(q_goal[i] - q_now[i]));
    }
    // Quintic profile peaks at 1.875 * delta / T, so pick T to respect max_vel.
    return std::max(2.0, 1.875 * max_delta / max_vel);
  };

  try {
    // Preview with a short-lived connection: the robot drops connections left idle while we wait for Enter.
    {
      franka::Robot preview(robot_ip, franka::RealtimeConfig::kIgnore);
      std::array<double, 7> q_now = preview.readOnce().q;
      PrintJoints("Current joints:", q_now);
      PrintJoints("Goal joints:   ", q_goal);
      std::cout << "Moving over " << duration_from(q_now) << " s (max joint speed " << max_vel << " rad/s)."
                << std::endl
                << "Keep the user stop button in hand. Press Enter to move, Ctrl+C to abort..." << std::endl;
    }
    std::cin.ignore();

    // The host kernel is not PREEMPT_RT, so skip the realtime scheduling requirement.
    franka::Robot robot(robot_ip, franka::RealtimeConfig::kIgnore);
    robot.automaticErrorRecovery();
    robot.setCollisionBehavior({{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
                               {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
                               {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
                               {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
                               {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
                               {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
    // Recompute from the state at motion time in case the arm moved while we waited.
    const double duration = duration_from(robot.readOnce().q);

    std::array<double, 7> q_start{};
    double time = 0.0;
    robot.control([&](const franka::RobotState& state, franka::Duration period) -> franka::JointPositions {
      if (time == 0.0) {
        q_start = state.q_d;
      }
      time += period.toSec();
      const double tau = std::min(time / duration, 1.0);
      const double s = tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau);
      std::array<double, 7> q_cmd{};
      for (size_t i = 0; i < 7; i++) {
        q_cmd[i] = q_start[i] + s * (q_goal[i] - q_start[i]);
      }
      franka::JointPositions output(q_cmd);
      return tau >= 1.0 ? franka::MotionFinished(output) : output;
    });
    std::cout << "Reached home." << std::endl;
  } catch (const franka::Exception& e) {
    std::cerr << "Franka error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
