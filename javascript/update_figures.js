// Figure 1a interactively loads mp4 video, stored locally
function update_figure_1a() {
  var figure_1a_microscope_type = document.getElementById("figure_1a_microscope_type").value;
  var video_name = "./images/figure_1a/" + figure_1a_microscope_type + "/fig_1a_video.mp4";
  var poster_name = "./images/figure_1a/" + figure_1a_microscope_type + "/img0.png"
  var vid = document.getElementById('Figure_1a_video');
  vid.src = video_name;
  vid.poster = poster_name;
  vid.play();
}

// Figure 1b interactively loads mp4 video, stored locally
function update_figure_1b() {
  var figure_1b_microscope_type = document.getElementById("figure_1b_microscope_type").value;
  var video_name = "./images/figure_1b/" + figure_1b_microscope_type + "/fig_1b_video.mp4";
  var poster_name = "./images/figure_1b/" + figure_1b_microscope_type + "/img0.png"
  var vid = document.getElementById('Figure_1b_video');
  vid.src = video_name;
  vid.poster = poster_name;
  vid.play();
}

// Figure 1c interactively loads mp4 video, stored locally
function update_figure_1c() {
  var figure_1c_microscope_type = document.getElementById("figure_1c_microscope_type").value;
  var video_name = "./images/figure_1c/" + figure_1c_microscope_type + "/fig_1c_video.mp4";
  var poster_name = "./images/figure_1c/" + figure_1c_microscope_type + "/img0.png"
  var vid = document.getElementById('Figure_1c_video');
  vid.src = video_name;
  vid.poster = poster_name;
  vid.play();
}

// Figure 2 interactively loads static images, stored locally
function update_figure_2() {
  var sample_type = document.getElementById("Figure_2_sample_type").value;
  var imaging_method = document.getElementById("Figure_2_imaging_method").value;
  var filename = "./images/figure_2/" + sample_type + "_" + imaging_method + ".svg";
  var image = document.getElementById("Figure_2b_image");
  image.src = filename;
}

// Figure 3 interactively loads static images, stored locally
function update_figure_3() {
  var angle = document.getElementById("Figure_3_angle").value;
  var filename = "./images/figure_3/STE_crimson_bead_" + angle + "_phase.svg";
  var image = document.getElementById("Figure_3_image");
  image.src = filename;
}

// Figure 4 interactively loads static images, stored locally
function update_figure_4() {
  var dye = document.getElementById("Figure_4_dye").value;
  var measurement = document.getElementById("Figure_4_measurement").value;
  var filename = "./images/figure_4/" + measurement + "_v_fluence_" + dye + ".png";
  var image = document.getElementById("Figure_4_image");
  image.src = filename;
}

// Figure 5 interactively loads static images, stored locally
function update_figure_5() {
  var sample_type = document.getElementById("Figure_5_sample_type").value;
  var fit_parameter = document.getElementById("Figure_5_fit_parameter").value;
  var filename = "./images/figure_5/fluorescence_depletion_" + sample_type + "_brightness_" + fit_parameter + ".svg";
  var image = document.getElementById("Figure_5_image");
  image.src = filename;
}

// Figure 6 interactively loads static images, stored locally
function update_figure_6() {
  var z = document.getElementById("Figure_6_z").value;
  var filename = "./images/figure_6/darkfield_STE_image_" + z + ".svg";
  var image = document.getElementById("Figure_6_image");
  image.src = filename;
}

// Figure A5 interactively loads static images, stored locally
function update_figure_A5() {
  var view = document.getElementById("Figure_A5_view").value;
  var filename = "./images/figure_A5/darkfield_STE_image_" + view + ".svg";
  var image = document.getElementById("Figure_A5_image");
  image.src = filename;
}

// Figure A6 interactively loads static images, stored locally
function update_figure_A6() {
  var my_powers = document.getElementById("Figure_A6_data_subset").value;
  var filename = "./images/figure_A6/STE_v_" + my_powers + ".svg";
  var image = document.getElementById("Figure_A6_image");
  image.src = filename;
}

// Figure A10 interactively loads static images, stored locally
function update_figure_A10() {
  var angle = document.getElementById("Figure_A10_angle").value;
  var filename = "./images/figure_A10/phase_STE_image_" + angle + ".svg";
  var image = document.getElementById("Figure_A10_image");
  image.src = filename;
}
