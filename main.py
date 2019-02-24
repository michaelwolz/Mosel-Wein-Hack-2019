import VideoAnalysis


def main():
    run_video_analysis()


def run_video_analysis():
    va = VideoAnalysis.VideoAnalysis()
    va.run_ai_version("data/RGB/video_rgb-13_09_2017-10_02.avi")


if __name__ == '__main__':
    main()
