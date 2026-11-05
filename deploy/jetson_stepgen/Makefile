CFLAGS ?= -O2 -Wall -Wextra -std=gnu11
bb8_stepgen: bb8_stepgen.c bb8_protocol.h
	$(CC) $(CFLAGS) -o $@ bb8_stepgen.c -lm
install: bb8_stepgen
	install -m 755 bb8_stepgen /usr/local/bin/bb8_stepgen
	install -m 644 bb8-stepgen.service /etc/systemd/system/bb8-stepgen.service
clean:
	rm -f bb8_stepgen
.PHONY: install clean
