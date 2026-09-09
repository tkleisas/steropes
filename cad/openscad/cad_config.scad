// Shared dimensions for the reference toolhead, in millimetres.
//
// These constants are mirrored by `constants:` in cad/manifest.yaml and
// by the collision primitives in steropes/gantry.py — the drift guard in
// tests/test_cadimport.py fails if any of the three diverge.

CARRIAGE_W = 40;                    // X width  (2 * gantry.CARRIAGE_HALF_MM[0])
CARRIAGE_D = 40;                    // Y depth  (2 * gantry.CARRIAGE_HALF_MM[1])
CARRIAGE_H = 20;                    // Z height (2 * gantry.CARRIAGE_HALF_MM[2])

FINGER_DIA = 8;                     // plunger shaft Ø (2 * gantry.FINGER_RADIUS_MM)
FINGER_LEN = 30;                    // shaft length (2 * gantry.FINGER_HALF_LEN_MM)
FINGER_OFFSET = [12, 0, -25];       // toolhead frame (gantry.FINGER_OFFSET_MM)
TOOLCAM_OFFSET = [-12, 0, -11];     // toolhead frame (gantry.TOOLCAM_OFFSET_MM)

M3_CLEAR = 3.2;                     // M3 clearance hole

SERVO_W = 23;                       // 9 g servo body, for the carriage pocket
SERVO_D = 12.5;
SERVO_H = 8;

$fn = 48;
