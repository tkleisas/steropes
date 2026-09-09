// Carriage plate: the toolhead's backbone, riding the gantry rails.
// Authored in the toolhead body frame — origin at the carriage centre,
// Z up — so the exported STL drops into the MJCF `toolhead` body with
// no offset.
//
// Export: openscad -o carriage_plate.stl -D 'PART="carriage_plate"' carriage_plate.scad
include <cad_config.scad>

PART = "preview";

module carriage_plate() {
    difference() {
        cube([CARRIAGE_W, CARRIAGE_D, CARRIAGE_H], center = true);

        // M3 mounting holes, one per corner, 5 mm in from the edges
        for (sx = [-1, 1], sy = [-1, 1])
            translate([sx * (CARRIAGE_W / 2 - 5), sy * (CARRIAGE_D / 2 - 5), 0])
                cylinder(d = M3_CLEAR, h = CARRIAGE_H + 2, center = true);

        // finger bore through the plate at the plunger's XY offset
        translate([FINGER_OFFSET[0], FINGER_OFFSET[1], 0])
            cylinder(d = FINGER_DIA + 1, h = CARRIAGE_H + 2, center = true);

        // servo pocket open at the top face (9 g servo body)
        translate([0, -8, CARRIAGE_H / 2 - SERVO_H / 2 + 0.5])
            cube([SERVO_W, SERVO_D, SERVO_H + 1], center = true);

        // toolcam cable pass-through at the camera's XY offset
        translate([TOOLCAM_OFFSET[0], TOOLCAM_OFFSET[1], 0])
            cube([10, 14, CARRIAGE_H + 2], center = true);
    }
}

if (PART == "preview" || PART == "carriage_plate")
    carriage_plate();
