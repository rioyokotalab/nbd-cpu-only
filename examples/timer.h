
#pragma once

#include <string>
#include <map>
#include <sys/time.h>

static timeval t;
static std::map<std::string,timeval> timer;

void start(std::string event) {
  gettimeofday(&t, NULL);
  timer[event] = t;
}

void stop(std::string event) {
  gettimeofday(&t, NULL);
  printf("%-20s : %f s\n", event.c_str(), (int64_t)t.tv_sec-timer[event].tv_sec+((int64_t)t.tv_usec-timer[event].tv_usec)*1e-6);
}
