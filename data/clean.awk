#!/usr/bin/awk -f
!/^>/ { printf toupper($0) }
