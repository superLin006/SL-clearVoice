/* Timestamp a mono 16 kHz S16_LE pipe against CLOCK_MONOTONIC.
 * feed [--pace] emits 10 ms microphone-sized packets; capture emits the input
 * unchanged and records 10 ms output availability. Model warmup must finish
 * before feed starts. Subtract feed.input_clock_ms from capture origins.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
enum { RATE = 16000, HOP = 160, CAPACITY = 65536 };
static void fail(const char *s) { perror(s); exit(1); }
static double now_ms(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) fail("clock_gettime");
    return 1000.0*t.tv_sec + t.tv_nsec*1e-6;
}
static void until(double at) {
    struct timespec t;
    t.tv_sec=(time_t)(at/1000);
    t.tv_nsec=(long)((at-t.tv_sec*1000.0)*1e6);
    int e;
    do { e=clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&t,NULL); } while(e==EINTR);
    if(e) { errno=e; fail("clock_nanosleep"); }
}
static int cmp(const void *a,const void *b) {
    double x=*(const double *)a,y=*(const double *)b;
    return (x>y)-(x<y);
}
int main(int argc,char **argv) {
    if(argc<2 || argc>3 || (strcmp(argv[1],"feed") && strcmp(argv[1],"capture")))
        { fputs("Usage: pcm_timing feed [--pace] | capture\n",stderr); return 2; }
    int feed=!strcmp(argv[1],"feed"), pace=argc==3 && !strcmp(argv[2],"--pace");
    if(argc==3 && (!feed || !pace)) return 2;
    static double origins[CAPACITY], intervals[CAPACITY];
    int16_t pcm[HOP];
    size_t samples=0,blocks=0,bytes;
    double start=now_ms(),first=0,last=0,max_interval=0,max_origin=-INFINITY;
    while((bytes=fread(pcm,1,sizeof(pcm),stdin))!=0) {
        if(bytes%2) { fputs("Incomplete S16_LE input\n",stderr); return 1; }
        size_t n=bytes/2;
        if(feed && pace) until(start+(samples+n)/16.0);
        double ready=now_ms();
        if(!blocks) first=ready;
        double origin=ready-samples/16.0,interval=blocks?ready-last:0;
        origins[blocks%CAPACITY]=origin;
        intervals[blocks%CAPACITY]=interval;
        max_origin=fmax(max_origin,origin);
        max_interval=fmax(max_interval,interval);
        if(fwrite(pcm,1,bytes,stdout)!=bytes || fflush(stdout)) fail("PCM write");
        last=ready;samples+=n;++blocks;
    }
    if(ferror(stdin)) fail("PCM read");
    size_t stored=blocks<CAPACITY?blocks:CAPACITY;
    qsort(origins,stored,sizeof(double),cmp);
    qsort(intervals,stored,sizeof(double),cmp);
    size_t p95=stored?(size_t)(.95*(stored-1)):0;
    fprintf(stderr,"{\"role\":\"%s\",\"paced\":%s,\"samples\":%zu,\"blocks\":%zu,"
        "\"input_clock_ms\":%.6f,\"first_ready_clock_ms\":%.6f,"
        "\"p95_origin_ms\":%.6f,\"max_origin_ms\":%.6f,"
        "\"p95_interval_ms\":%.6f,\"max_interval_ms\":%.6f}\n",
        argv[1],pace?"true":"false",samples,blocks,start,first,
        stored?origins[p95]:0,stored?max_origin:0,stored?intervals[p95]:0,max_interval);
    return 0;
}
