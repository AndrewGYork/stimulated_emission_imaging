// Figure 5 interactively loads static images, stored locally
function update_figure_5() {
  var z = document.getElementById("Figure_5_z").value;
  var filename = "./images/figure_5/darkfield_STE_image_" + z + ".svg";
  var image = document.getElementById("Figure_5_image");
  image.src = filename;
}