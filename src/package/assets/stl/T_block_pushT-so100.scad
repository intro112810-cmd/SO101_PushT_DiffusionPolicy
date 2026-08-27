
// pushT-so100 canonical T-block - training data truth
// Full dims: top 100x30x30, stem 30x70x30, height 30mm
// MuJoCo half-sizes: 0.05 0.015 0.015 / 0.015 0.035 0.015
union(){
  translate([0, 35, 15]) cube([100, 30, 30], center=true);
  translate([0, -15, 15]) cube([30, 70, 30], center=true);
}
