#!/usr/bin/env bash
for i in `ls progress*.txt`; do
convert \
  -size 1500x300 \
  xc:White \
  -gravity West \
  -weight 700 \
  -font "AndaleMono" \
  -pointsize 32 \
  -annotate +80+0 "`cat $i | sed -e 's/WER/\\\ WER/g'`" \
  $i.png
done
convert -delay 20 -loop 1 progress*.txt.png test.gif
