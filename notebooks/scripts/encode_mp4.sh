ffmpeg -y -framerate 3 \
  -pattern_type glob -i '275.rep/*.png' \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -preset veryfast \
  275.rep.mp4