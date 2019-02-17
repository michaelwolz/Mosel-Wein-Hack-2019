import frames


def main():
    run_video_analysis()


def run_video_analysis():
    va = frames.VideoAnalysis()
    va.run("data/RGB/video_rgb-13_09_2017-09_56.avi")
    # va.run_version_2("data/RGB/video_rgb-13_09_2017-09_56.avi")


if __name__ == '__main__':
    main()
