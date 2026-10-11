# DAMIAO IMU sensor data message
# Published at the device output rate (~200 Hz typical)

uint8 device_id          # IMU device ID (from frame header)

# Accelerometer (m/s²)
float32 accel_x
float32 accel_y
float32 accel_z

# Gyroscope (rad/s)
float32 gyro_x
float32 gyro_y
float32 gyro_z

# Euler angles (rad)
float32 roll
float32 pitch
float32 yaw

# Quaternion (w, x, y, z)
float32 q_w
float32 q_x
float32 q_y
float32 q_z

# Temperature (°C)
float32 temperature
