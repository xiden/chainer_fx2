#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>

#pragma pack(push, 1)
struct Row {
	int ctm;
	double open;
	double low;
	double high;
	double close;
	double volume;
};
#pragma pack(pop)

int main(int argc, char** argv) {
	if (argc < 2) {
		printf("usage : exthst <tail_rows>\n");
		return 0;
	}
	FILE* fpi = fopen("USDJPY.hst", "rb");
	if (fpi == NULL) {
		printf("Can not open.\n");
		return -1;
	}
	auto rows = atoi(argv[1]);
	Row r;
	char buf[256];

	_fseeki64(fpi, -sizeof(Row), SEEK_END);
	fread(&r, sizeof(r), 1, fpi);

	auto dtlen = strftime(buf, 256, "%Y%m%d_%H%M", _localtime32((__time32_t*)&r.ctm));
	sprintf(buf + dtlen, "_%d.csv", rows);

	FILE* fpo = fopen(buf, "wt");

	_fseeki64(fpi, -sizeof(Row) * rows, SEEK_END);
	for (int i = 0; i < rows; i++) {
		fread(&r, sizeof(r), 1, fpi);
		strftime(buf, 20, "%Y-%m-%d,%H:%M", _localtime32((__time32_t*)&r.ctm));
		fprintf(
			fpo,
			"%s,%.3f,%.3f,%.3f,%.3f,%.3f\n",
			buf,
			r.open,
			r.low,
			r.high,
			r.close,
			r.volume
		);
	}
	fclose(fpi);
	fclose(fpo);
	return 0;
}
