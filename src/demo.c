#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
//#include <sys/time.h>
#include <time.h>
#include <winsock.h>
#include "gettimeofday.h"

#include <Windows.h>

#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
image get_image_from_stream(CvCapture *cap);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static float *avg;

static pthread_mutex_t mattex;

static int exit_pending = 0;

const char* win1 = "Instream";
const char* win2 = "Detected";

double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void fetch() {
}
void detect() {
	Sleep(500);
	while (!exit_pending) {
		pthread_mutex_lock(&mattex);
		det = in;
		pthread_mutex_unlock(&mattex);
		det_s = resize_image(in, net.w, net.h);


		float nms = .4;

		layer l = net.layers[net.n - 1];
		float *X = det_s.data;
		float *prediction = network_predict(net, X);

		memcpy(predictions[demo_index], prediction, l.outputs * sizeof(float));
		mean_arrays(predictions, FRAMES, l.outputs, avg);
		l.output = avg;

		free_image(det_s);
		if (l.type == DETECTION) {
			get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
		}
		else if (l.type == REGION) {
			get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
		}
		else {
			error("Last layer must produce detections\n");
		}

		if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);

		printf("\033[2J");
		printf("\033[1;1H");
		printf("\nFPS:%.1f\n", fps);
		printf("Objects:\n\n");

		images[demo_index] = det;
		//det = images[(demo_index + FRAMES / 2 + 1) % FRAMES];
		demo_index = (demo_index + 1) % FRAMES;

		draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

	}
}


void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix)
{
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);
	
	//filename = "small.mp4";
	if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
		
    }

    if(!cap) error("Couldn't connect to webcam.\n");

	//double target_framerate = cvGetCaptureProperty(cap, CV_CAP_PROP_FPS);
	//printf("\nCapture Stream FPS:%.1f          ", target_framerate);


    layer l = net.layers[net.n-1];
    int j;

	avg =(float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
	/*

	fetch_in_thread(0);
    det = in;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;

    for(j = 0; j < FRAMES/2; ++j){
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
        det = in;
        det_s = in_s;
    }*/

	pthread_t fetch_thread;
	pthread_t detect_thread;

	int showWindow = prefix == 0;

    int count = 0;
    if(showWindow){
		cvNamedWindow("Demo", CV_WINDOW_NORMAL);
		cvMoveWindow("Demo", 0, 0);
		cvResizeWindow("Demo", 600, 500);
		cvNamedWindow(win2, CV_WINDOW_NORMAL);
		cvMoveWindow(win2, 600, 0);
		cvResizeWindow(win2, 600, 500);
	}


	if (pthread_mutex_init(&mattex, NULL) != 0)
	{
		printf("\n mutex init failed\n");
		return 1;
	}

	//if(pthread_create(&detect_thread, 0, detect, 0)) error("Thread creation failed");
	int frames_in = 0;
	double before = get_wall_time();

	while (!exit_pending) {

		++frames_in;

		in = get_image_from_stream(cap);
		if (!in.data) {
			error("Stream closed.");
			break;
		}


		show_image(in, "Demo");
		show_image(det, win2);
		int i = cvWaitKey(1);
		if (i == 27)exit_pending = 1;

		pthread_mutex_lock(&mattex);
		free_image(in);
		pthread_mutex_unlock(&mattex);

		double after = get_wall_time();
		if (after - before > 1.) //update every second
		{
			float curr = (float)frames_in / (after - before);
			fps = curr;
			before = after;
			frames_in = 0;
			printf("\r Stream FPS:%.1f          ", fps);
		}
	}

//	pthread_join(detect_thread,0);

	cvDestroyAllWindows();
}

#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

