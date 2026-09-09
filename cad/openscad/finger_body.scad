// Finger body: the pogo-pin style plunger that taps the DUT screen.
// Authored in the finger body frame — origin at the shaft centre
// (matching the collision cylinder in gantry.toolhead_xml), Z up,
// tip toward -Z.
//
// Export: openscad -o finger_body.stl -D 'PART="finger_body"' finger_body.scad
include <cad_config.scad>

PART = "preview";

module finger_body() {
    difference() {
        union() {
            // shaft — this is exactly the collision envelope, keep it
            // FINGER_DIA x FINGER_LEN centred on the origin
            cylinder(d = FINGER_DIA, h = FINGER_LEN, center = true);
            // spring collar at the top end (visual only; above the
            // collision envelope, it never meets the DUT)
            translate([0, 0, FINGER_LEN / 2 - 1.5])
                cylinder(d = 13, h = 3, center = true);
        }
        // retaining-pin cross hole through the collar
        translate([0, 0, FINGER_LEN / 2 - 1.5])
            rotate([0, 90, 0])
                cylinder(d = M3_CLEAR, h = 14, center = true);
    }
}

if (PART == "preview" || PART == "finger_body")
    finger_body();
