TWO_BALLS_DEF = """
  bodies {
    name: "ball_1"
    colliders {
        position {
          x: 0.0
          y: 0.0
          z: 0.0
        }
        sphere {
          radius: 0.2
        }
        color: "#4472C4"
    }
    frozen {
          rotation: {
            x: 1
            y: 1
            z: 1
          }
          position: {
            y: 1
            z: 1
          }
        }
    mass: 1.0
  }
  bodies {
    name: "ball_2"
    colliders {
        position {
          x: 0.0
          y: 0.0
          z: 0.0
        }
        sphere {
          radius: 0.2
        }
        color: "#355E3B"
    }
    frozen {
          rotation: {
            x: 1
            y: 1
            z: 1
          }
          position: {
            y: 1
            z: 1
          }
        }
    mass: 1.0
  }

  bodies {
      name: "left_plane"
      colliders {
        position {
          x: -1.0
          y: 0.0
          z: 0.0
        }
        box {
          halfsize {
            x: 0.1
            y: 1
            z: 1
          }
        }
      }
      frozen {
        all: true
      }
      mass: 1.0
  }
  bodies {
      name: "right_plane"
      colliders {
        position {
          x: 1.0
          y: 0.0
          z: 0.0
        }
        box {
          halfsize {
            x: 0.1
            y: 1
            z: 1
          }
        }
      }
      frozen {
        all: true
      }
      mass: 1.0
  }
  forces {
    name: "ball_1"
    body: "ball_1"
    strength: 1
    thruster {}
  }
  forces {
    name: "ball_2"
    body: "ball_2"
    strength: 1
    thruster {}
  }
  defaults {
    qps {
      name: "ball_1"
      pos {
        x: -0.5
        y: 0.0
        z: 0.0
      }
      rot {
        x: 0.0
        y: 0.0
        z: 0.0
      }
      vel {
        x: 0.0
        y: 0.0
        z: 0.0
      }
      ang {
        x: 0.0
        y: 0.0
        z: 0.0
      }
    }
    qps {
      name: "ball_2"
      pos {
        x: 0.5
        y: 0.0
        z: 0.0
      }
      rot {
        x: 0.0
        y: 0.0
        z: 0.0
      }
      vel {
        x: 0.0
        y: 0.0
        z: 0.0
      }
      ang {
        x: 0.0
        y: 0.0
        z: 0.0
      }
    }
    qps {
      name: "left_plane"
      pos {
        x: 0.0
        y: 0.0
        z: 0.0
      }
      rot {
        x: 0.0
        y: 0.0
        z: 0.0
      }
      vel {
        x: 0.0
        y: 0.0
        z: 0.0
      }
      ang {
        x: 0.0
        y: 0.0
        z: 0.0
      }
    }
    qps {
      name: "right_plane"
      pos {
        x: 0.0
        y: 0.0
        z: 0.0
      }
      rot {
        x: 0.0
        y: 0.0
        z: 0.0
      }
      vel {
        x: 0.0
        y: 0.0
        z: 0.0
      }
      ang {
        x: 0.0
        y: 0.0
        z: 0.0
      }
    }
  }
  elasticity: 1.0
  friction: 0.0
  velocity_damping: 0.0
  dt: 0.02
  substeps: 2
  dynamics_mode: "pbd"
"""

UR5E_DEF = """
  bodies {
    name: "shoulder_link"
    colliders {
      position {
        y: 0.06682991981506348
      }
      rotation {
        x: 90.0
      }
      capsule {
        radius: 0.05945208668708801
        length: 0.13365983963012695
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "upper_arm_link"
    colliders {
      position {
        z: 0.21287038922309875
      }
      rotation {
      }
      capsule {
        radius: 0.05968618765473366
        length: 0.5446449518203735
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "forearm_link"
    colliders {
      position {
        z: 0.1851803958415985
      }
      rotation {
      }
      capsule {
        radius: 0.05584339052438736
        length: 0.48926496505737305
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "wrist_1_link"
    colliders {
      position {
        y: 0.10467606782913208
      }
      rotation {
        x: 90.0
      }
      capsule {
        radius: 0.038744933903217316
        length: 0.10467606782913208
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "wrist_2_link"
    colliders {
      position {
        z: 0.052344050258398056
      }
      rotation {
      }
      capsule {
        radius: 0.03879201412200928
        length: 0.10468810051679611
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "wrist_3_link"
    colliders {
      position {
        y: -0.04025782644748688
      }
      rotation {
        x: 90.0
      }
      capsule {
        radius: 0.01725015603005886
        length: 0.08051565289497375
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "floor"
    colliders {
      plane {
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
    frozen {
      all: true
    }
  }
  joints {
    name: "shoulder_pan_joint"
    parent: "floor"
    child: "shoulder_link"
    parent_offset {
      z: 0.163
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angular_damping: 0.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "shoulder_lift_joint"
    parent: "shoulder_link"
    child: "upper_arm_link"
    parent_offset {
      y: 0.138
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 0.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "elbow_joint"
    parent: "upper_arm_link"
    child: "forearm_link"
    parent_offset {
      y: -0.13
      z: 0.425
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 0.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "wrist_1_joint"
    parent: "forearm_link"
    child: "wrist_1_link"
    parent_offset {
      z: 0.3919999897480011
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 0.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "wrist_2_joint"
    parent: "wrist_1_link"
    child: "wrist_2_link"
    parent_offset {
      y: 0.12700000405311584
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angular_damping: 0.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "wrist_3_joint"
    parent: "wrist_2_link"
    child: "wrist_3_link"
    parent_offset {
      z: 0.1
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 0.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  actuators {
    name: "shoulder_pan_joint"
    joint: "shoulder_pan_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "shoulder_lift_joint"
    joint: "shoulder_lift_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "elbow_joint"
    joint: "elbow_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "wrist_1_joint"
    joint: "wrist_1_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "wrist_2_joint"
    joint: "wrist_2_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "wrist_3_joint"
    joint: "wrist_3_joint"
    strength: 100.0
    torque {
    }
  }
  elasticity: 1.0
  friction: 0.0
  angular_damping: 0.0
  dt: 0.02
  substeps: 8
  dynamics_mode: "pbd"
"""
