import os, glob

if __name__ == '__main__':
    # Cluster parameters.
    queue_name = 'fast.master.q'
    memory = '5M'
    # Route to the images.
    images = glob.glob('bike/*.bmp')
    images += glob.glob('cars/*.bmp')
    images += glob.glob('person/*.bmp')
    images += glob.glob('none/*.bmp')
    cmd_mask = "convert %s -limit thread 1 -interpolate spline -resize 1024x768 -sharpen
    0x1 -normalize -resize 640x480 %s"
    print 'Launching %d tasks...' % (len(images))
    for input_filename in images:
        output_filename = input_filename[:-3] + 'png'
        current_cmd = cmd_mask % (input_filename, output_filename)
        cmd = 'qsub -b y -cwd -V -q %s -l mem_token=%s,mem_free=%s %s' % (queue_name,memory, memory, current_cmd)
        print cmd
        print os.popen(cmd).read()