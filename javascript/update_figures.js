// Figure 5 interactively loads static images, stored locally
function update_figure_5() {
  var z = document.getElementById("Figure_5_z").value;
  var filename = "./images/figure_5/darkfield_STE_image_" + z + ".svg";
  var image = document.getElementById("Figure_5_image");
  image.src = filename;
}

// Figure 6 interactively loads static images, stored locally
function update_figure_6() {
  var z = document.getElementById("Figure_6_z").value;
  var filename = "./images/figure_6/fluorescence_depletion_image_" + z + ".svg";
  var image = document.getElementById("Figure_6_image");
  image.src = filename;
}

// Figure 7 interactively loads static images, stored locally
function update_figure_7() {
  var angle = document.getElementById("Figure_7_angle").value;
  var filename = "./images/figure_7/phase_STE_image_" + angle + ".svg";
  var image = document.getElementById("Figure_7_image");
  image.src = filename;
}