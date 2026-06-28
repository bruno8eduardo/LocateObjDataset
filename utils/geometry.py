import numpy as np

class Geometry:

    minimal_distance_param = 0.01

    droneToMundoR = np.array([[0,1,0],[1,0,0],[0,0,-1]])
    mundoToDroneR = np.transpose(droneToMundoR)
    cameraToDroneR = np.array([[0,0,1],[1,0,0],[0,1,0]])
    droneToCameraR = np.transpose(cameraToDroneR)
    cameraToMundoR = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    mundoToCameraR = np.transpose(cameraToMundoR)
    cameraToOpenglR = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

    @staticmethod
    def inv_K(K):
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        K_inv = np.array([[1/fx, 0, -cx/fx],
                [0, 1/fy, -cy/fy],
                [0, 0, 1]])
        return K_inv
    
    @staticmethod
    def yaw_pitch_roll_to_rotation_matrix(yaw, pitch, roll):
        # Converter ângulos de graus para radianos
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)

        # Matrizes de rotação básicas
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0,             1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rx = np.array([
            [1, 0,           0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])

        # Matriz de rotação composta: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        return R

