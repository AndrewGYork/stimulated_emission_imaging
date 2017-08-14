// Figure 2 interactively loads static images, stored locally
function update_figure_2() {
  var sample_type = document.getElementById("Figure_2_sample_type").value;
  var imaging_modality = document.getElementById("Figure_2_imaging_modality").value;
  var filename = "./images/figure_2/" + sample_type + "_" + imaging_modality + ".svg";
  var image = document.getElementById("Figure_2_image");
  image.src = filename;
}

// Figure 5 interactively loads static images, stored locally
function update_figure_4() {
  var my_powers = document.getElementById("Figure_5_data_subset").value;
  var filename = "./images/figure_4/STE_v_" + my_powers + ".svg";
  var image = document.getElementById("Figure_4_image");
  image.src = filename;
}


// Figure 6ab interactively loads static images, stored locally
function update_figure_6ab() {
  var z = document.getElementById("Figure_6_z").value;
  var filename = "./images/figure_6/darkfield_STE_image_" + z + ".svg";
  var image = document.getElementById("Figure_6ab_image");
  image.src = filename;
}

// Figure 6cd interactively loads static images, stored locally
function update_figure_6cd() {
  var view = document.getElementById("Figure_6_view").value;
  var filename = "./images/figure_6/darkfield_STE_image_" + view + ".svg";
  var image = document.getElementById("Figure_6cd_image");
  image.src = filename;
}


// Figure 7 interactively loads static images, stored locally
function update_figure_7() {
  var angle = document.getElementById("Figure_7_angle").value;
  var filename = "./images/figure_7/STE_crimson_bead_" + angle + "_phase.svg";
  var image = document.getElementById("Figure_7_image");
  image.src = filename;
}


// Figure S3 interactively loads static images, stored locally
function update_figure_S3() {
  var angle = document.getElementById("Figure_S3_angle").value;
  var filename = "./images/figure_S3/phase_STE_image_" + angle + ".svg";
  var image = document.getElementById("Figure_S3_image");
  image.src = filename;
}
