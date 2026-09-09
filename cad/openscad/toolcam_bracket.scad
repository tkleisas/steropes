// Toolcam bracket: hangs the downward toolhead camera off the carriage.
// Local frame: origin at the bracket's top-face centre — the manifest
// places the mesh at gantry.TOOLCAM_OFFSET_MM in the toolhead frame —
// Z up (parts extend toward -Z, below the carriage).
//
// Export: openscad -o toolcam_bracket.stl -D 'PART="toolcam_bracket"' toolcam_bracket.scad
include <cad_config.scad>

PART = "preview";

PLATE_T = 2.5;        // top plate thickness
CAM_W = 25;           // camera module outline
CAM_D = 24;
CAM_POCKET_H = 6;

module toolcam_bracket() {
    difference() {
        union() {
            // horizontal plate against the carriage underside (top face z=0)
            translate([0, 0, -PLATE_T / 2])
                cube([18, 18, PLATE_T], center = true);
            // camera cradle hanging below the plate
            translate([0, 0, -PLATE_T - CAM_POCKET_H / 2])
                cube([CAM_W, CAM_D, CAM_POCKET_H], center = true);
        }
        // camera pocket, open at the bottom
        translate([0, 0, -PLATE_T - CAM_POCKET_H / 2 - 1])
            cube([CAM_W - 3, CAM_D - 3, CAM_POCKET_H + 2], center = true);
        // lens bore straight through plate and cradle
        translate([0, 0, -PLATE_T - CAM_POCKET_H / 2])
            cylinder(d = 8, h = 3 * (PLATE_T + CAM_POCKET_H), center = true);
        // 2x M3 mounting holes through the top plate
        for (sy = [-1, 1])
            translate([0, sy * 6, -PLATE_T / 2])
                cylinder(d = M3_CLEAR, h = PLATE_T + 2, center = true);
    }
}

if (PART == "preview" || PART == "toolcam_bracket")
    toolcam_bracket();
