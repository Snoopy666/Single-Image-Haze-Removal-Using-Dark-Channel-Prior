#include "dehaze.h"
#include <iostream>
#include <ctime>

int main() {
    DeHaze img;
    clock_t start = clock();

    img.loadImage("E:/1.JPG");
    img.showImage("src", DeHaze::SRC);
    img.getDarkChannelPrior();
    img.showImage("dark", DeHaze::DARK);
    //img.saveImage("E:/2.bmp", DeHaze::DARK);
    img.getAtmosphericLight();
    img.getTransmission();
    img.showImage("tran", DeHaze::TRAN);
    //img.saveImage("E:/3.bmp", DeHaze::TRAN);
    img.gFilter();
    img.showImage("gtran", DeHaze::GTRAN);
    //img.saveImage("E:/4.bmp", DeHaze::GTRAN);
    img.recoverSceneRadiace();
    img.showImage("dst", DeHaze::DST);
    //img.saveImage("E:/5.bmp",DeHaze::DST);
    clock_t end = clock();
    cout << "Time consumed : " << (float)(end - start) / CLOCKS_PER_SEC << "s" << endl;
    //cin.get();
    return 0;
}
