//run("Image Sequence...", "open=/home/user/Desktop/temp10/png/img0.png sort");
selectWindow("png");

// Split into pieces

run("Duplicate...", "duplicate range=2-28");
rename("Intro");
intro_size = (28 - 2) + 1;
print("intro size is", intro_size);

selectWindow("png");
run("Duplicate...", "duplicate range=29-75");
rename("1x");
cycle_size = (75 - 29) + 1;
print("cycle size is", cycle_size);

selectWindow("png");
run("Duplicate...", "duplicate range=76-97");
rename("Outro");
outro_size = (97-76) + 1;
print("outro size is", outro_size);

selectWindow("png");
run("Duplicate...", "duplicate range=97-97");
rename("Pause");




//Repeat stimulation at triple speed
selectWindow("1x");
run("Duplicate...", "duplicate range=1-47");
rename("3x");
run("Concatenate...", "  title=3x image1=3x image2=3x image3=3x image4=[-- None --]");
run("Grouped Z Project...", "projection=[Average Intensity] group=3");
selectWindow("3x");
close();
selectWindow("AVG_3x");
rename("3x");



//Repeat stimulation at 9x speed
selectWindow("3x");
run("Duplicate...", "duplicate range=1-47");
rename("9x");
run("Concatenate...", "  title=9x image1=9x image2=9x image3=9x image4=[-- None --]");
run("Grouped Z Project...", "projection=[Average Intensity] group=3");
selectWindow("9x");
close();
selectWindow("AVG_9x");
rename("9x");



//Repeat stimulation at 27x speed
selectWindow("9x");
run("Duplicate...", "duplicate range=1-47");
rename("27x");
run("Concatenate...", "  title=27x image1=27x image2=27x image3=27x image4=[-- None --]");
run("Grouped Z Project...", "projection=[Average Intensity] group=3");
selectWindow("27x");
close();
selectWindow("AVG_27x");
rename("27x");



//Repeat the last frame
run("Concatenate...", "  title=Pause image1=Pause image2=Pause image3=[-- None --]");
run("Concatenate...", "  title=Pause image1=Pause image2=Pause image3=[-- None --]");
run("Concatenate...", "  title=Pause image1=Pause image2=Pause image3=[-- None --]");
run("Concatenate...", "  title=Pause image1=Pause image2=Pause image3=[-- None --]");
run("Concatenate...", "  title=Pause image1=Pause image2=Pause image3=[-- None --]");
pause_size = 32;
print("pause size is", pause_size);



//Create slow-down frames
selectWindow("1x");
run("Duplicate...", "duplicate range=1-47");
rename("1x_slow");
selectWindow("3x");
run("Duplicate...", "duplicate range=1-47");
rename("3x_slow");
selectWindow("9x");
run("Duplicate...", "duplicate range=1-47");
rename("9x_slow");
selectWindow("27x");
run("Duplicate...", "duplicate range=1-47");
rename("27x_slow");


//Start stim emission photon count
num_photons = 0;

x_shift = 25;
//spontaneous emission counter position
fl_phot_x = 349 - x_shift;
fl_phot_x_2 = 330 - x_shift;
fl_phot_y = 280;
fl_phot_y_2 = 316;
fl_phot_x_3 = 277
fl_phot_y_3 = 342;

//stimulated emission counter position
ste_phot_x = 849 - x_shift;
ste_phot_x_2 = 750;
ste_phot_y = 280;
ste_phot_y_2 = 316;

//time counter position
time_x = 23;
time_y = 560;

//time counter precision
time_prec = 3;

//time steps
t_spont_ns = 3;
num_frames = cycle_size * (2*1 + 2*3 + 2*9 + 2*27);
t_step_1x = t_spont_ns / num_frames;
t_step_3x = t_step_1x * 3;
t_step_9x = t_step_1x * 9;
t_step_27x = t_step_1x * 27;

//initialize time counter so that timer starts from excited state
//time_ns = -1 * t_step_1x * intro_size;

//initialize time counter so that it starts from 0
time_ns = 0

//Label photon count and timer for intro
for (i=1; i<=intro_size; i++) {
	selectWindow("Intro");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);
	time_ns += t_step_1x;

	//spontaneous emission always 0 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("0 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("Intro_temp");
	} else {
		rename("new_intro_piece");
		run("Concatenate...", " title=Intro_temp image1=Intro_temp image2=new_intro_piece image3=[-- None --]");
	}
}
selectWindow("Intro");
close();
selectWindow("Intro_temp");
rename("Intro");

//Label photon count and timer for 1x
next_photon_event = 1;
speed_multiplier = 1;
for (i=1; i<=cycle_size; i++) {
	if (i==round(next_photon_event)) {
		next_photon_event += cycle_size/speed_multiplier;
		num_photons++;
	}

	selectWindow("1x");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);
	time_ns += t_step_1x;

	//spontaneous emission always 0 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("0 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("1x_temp");
	} else {
		rename("new_1x_piece");
		run("Concatenate...", " title=1x_temp image1=1x_temp image2=new_1x_piece image3=[-- None --]");
	}
}
selectWindow("1x");
close();
selectWindow("1x_temp");
rename("1x");

//Label photon count and timer for 3x
next_photon_event = 1;
speed_multiplier = 3;
for (i=1; i<=cycle_size; i++) {
	if (i==round(next_photon_event)) {
		next_photon_event += cycle_size/speed_multiplier;
		num_photons++;
	}

	selectWindow("3x");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);
	time_ns += t_step_3x;

	//spontaneous emission always 0 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("0 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("3x_temp");
	} else {
		rename("new_3x_piece");
		run("Concatenate...", " title=3x_temp image1=3x_temp image2=new_3x_piece image3=[-- None --]");
	}
}
selectWindow("3x");
close();
selectWindow("3x_temp");
rename("3x");

//Label photon count and timer for 9x
next_photon_event = 1;
speed_multiplier = 9;
for (i=1; i<=cycle_size; i++) {
	if (i==round(next_photon_event)) {
		next_photon_event += cycle_size/speed_multiplier;
		num_photons++;
	}

	selectWindow("9x");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);
	time_ns += t_step_9x;

	//spontaneous emission always 0 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("0 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("9x_temp");
	} else {
		rename("new_9x_piece");
		run("Concatenate...", " title=9x_temp image1=9x_temp image2=new_9x_piece image3=[-- None --]");
	}
}
selectWindow("9x");
close();
selectWindow("9x_temp");
rename("9x");

//Label photon count and timer for 27x
next_photon_event = 1;
speed_multiplier = 27;
for (i=1; i<=cycle_size; i++) {
	if (i==round(next_photon_event)) {
		next_photon_event += cycle_size/speed_multiplier;
		num_photons++;
	}

	selectWindow("27x");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);
	time_ns += t_step_27x;

	//spontaneous emission always 0 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("0 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("27x_temp");
	} else {
		rename("new_27x_piece");
		run("Concatenate...", " title=27x_temp image1=27x_temp image2=new_27x_piece image3=[-- None --]");
	}
}
selectWindow("27x");
close();
selectWindow("27x_temp");
rename("27x");

//Label photon count and timer for 27x slowdown
next_photon_event = 1;
speed_multiplier = 27;
for (i=1; i<=cycle_size; i++) {
	if (i==round(next_photon_event)) {
		next_photon_event += cycle_size/speed_multiplier;
		num_photons++;
	}

	selectWindow("27x_slow");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);
	time_ns += t_step_27x;

	//spontaneous emission always 0 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("0 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("27x_slow_temp");
	} else {
		rename("new_27x_slow_piece");
		run("Concatenate...", " title=27x_slow_temp image1=27x_slow_temp image2=new_27x_slow_piece image3=[-- None --]");
	}
}
selectWindow("27x_slow");
close();
selectWindow("27x_slow_temp");
rename("27x_slow");

//Label photon count and timer for 9x slowdown
next_photon_event = 1;
speed_multiplier = 9;
for (i=1; i<=cycle_size; i++) {
	if (i==round(next_photon_event)) {
		next_photon_event += cycle_size/speed_multiplier;
		num_photons++;
	}

	selectWindow("9x_slow");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);
	time_ns += t_step_9x;

	//spontaneous emission always 0 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("0 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("9x_slow_temp");
	} else {
		rename("new_9x_slow_piece");
		run("Concatenate...", " title=9x_slow_temp image1=9x_slow_temp image2=new_9x_slow_piece image3=[-- None --]");
	}
}
selectWindow("9x_slow");
close();
selectWindow("9x_slow_temp");
rename("9x_slow");


//Label photon count and timer for 3x slowdown
next_photon_event = 1;
speed_multiplier = 3;
for (i=1; i<=cycle_size; i++) {
	if (i==round(next_photon_event)) {
		next_photon_event += cycle_size/speed_multiplier;
		num_photons++;
	}

	selectWindow("3x_slow");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);
	time_ns += t_step_3x;

	//spontaneous emission always 0 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("0 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("3x_slow_temp");
	} else {
		rename("new_3x_slow_piece");
		run("Concatenate...", " title=3x_slow_temp image1=3x_slow_temp image2=new_3x_slow_piece image3=[-- None --]");
	}
}
selectWindow("3x_slow");
close();
selectWindow("3x_slow_temp");
rename("3x_slow");


//Label photon count and timer for 1x slowdown
next_photon_event = 1;
speed_multiplier = 1;
for (i=1; i<=cycle_size; i++) {
	if (i==round(next_photon_event)) {
		next_photon_event += cycle_size/speed_multiplier;
		num_photons++;
	}

	selectWindow("1x_slow");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);
	time_ns += t_step_1x;

	//spontaneous emission always 0 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("0 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("1x_slow_temp");
	} else {
		rename("new_1x_slow_piece");
		run("Concatenate...", " title=1x_slow_temp image1=1x_slow_temp image2=new_1x_slow_piece image3=[-- None --]");
	}
}
selectWindow("1x_slow");
close();
selectWindow("1x_slow_temp");
rename("1x_slow");



//Label photon count and timer for outro
num_photons++
for (i=1; i<=outro_size; i++) {
	selectWindow("Outro");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);
	time_ns += t_step_1x;

	//spontaneous emission now 1 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("1 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("Outro_temp");
	} else {
		rename("new_outro_piece");
		run("Concatenate...", " title=Outro_temp image1=Outro_temp image2=new_outro_piece image3=[-- None --]");
	}
}
selectWindow("Outro");
close();
selectWindow("Outro_temp");
rename("Outro");

//Label photon count and timer for pause
for (i=1; i<=pause_size; i++) {
	selectWindow("Pause");
	run("Duplicate...", "duplicate range=i-i");
	//setTool("text");
	setForegroundColor(0, 0, 0);

	//timer
	setFont("SansSerif", 30, "bold antialiased");
	setColor("black");
	drawString("Elapsed time: " + d2s(time_ns,time_prec) + " nanoseconds", time_x, time_y);

	//spontaneous emission now 1 photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString("1 photons", fl_phot_x, fl_phot_y);
	drawString("fluorescence", fl_phot_x_2, fl_phot_y_2);
	setFont("SansSerif", 20, "bold antialiased");
	setColor("black");
	drawString("(spontaneous emission)", fl_phot_x_3, fl_phot_y_3);

	//stimulated emission num photons
	setFont("SansSerif", 27, "bold antialiased");
	setColor("black");
	drawString(num_photons+" photons", ste_phot_x, ste_phot_y);
	drawString("stimulated emission", ste_phot_x_2, ste_phot_y_2);

	if(i==1) {
		rename("Pause_temp");
	} else {
		rename("new_pause_piece");
		run("Concatenate...", " title=Pause_temp image1=Pause_temp image2=new_pause_piece image3=[-- None --]");
	}
}
selectWindow("Pause");
close();
selectWindow("Pause_temp");
rename("Pause");


//Concatenate everything but intro and outro
run("Concatenate...", " title=FF image1=1x image2=3x image3=[-- None --]");
run("Concatenate...", " title=FF image1=FF image2=9x image3=[-- None --]");
run("Concatenate...", " title=FF image1=FF image2=27x image3=[-- None --]");
run("Concatenate...", " title=FF image1=FF image2=27x_slow image3=[-- None --]");
run("Concatenate...", " title=FF image1=FF image2=9x_slow image3=[-- None --]");
run("Concatenate...", " title=FF image1=FF image2=3x_slow image3=[-- None --]");
run("Concatenate...", " title=FF image1=FF image2=1x_slow image3=[-- None --]");



//Assemble full video
run("Concatenate...", " title=fl_vs_ste_png_out image1=Intro image2=FF image3=Outro image4=Pause image5=[-- None --]");

