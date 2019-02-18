import video_analysis


def main():
    run_video_analysis()


def run_video_analysis():
    va = video_analysis.VideoAnalysis()
    va.run_version_2("data/RGB/video_rgb-13_09_2017-09_56.avi")
    # va.run_version_2("data/RGB/video_rgb-13_09_2017-09_56.avi")
    # va.run_version_2("data/RGB/video_rgb-13_09_2017-09_56.avi")


if __name__ == '__main__':
    main()
