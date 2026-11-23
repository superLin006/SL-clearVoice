# 多麦 WPE 和波束回放

`wpe_online_cpp.cpp` 对选定的六个通道执行在线 WPE 和延时求和波束，输入是 16 kHz 交织 float32 文件，输出是同样采样率的单声道 float32 文件。程序从文件读取完整录音，再逐帧处理。用它生成供板端增强链回放的单声道参考。

在仓库根目录构建。传入一个输出路径可将二进制放在 `out/`。

~~~sh
bash scripts/build_wpe_online_board.sh out/wpe_online_cpp
~~~

参数依次为输入、输出、输入通道数、圆周位置数、选用的通道顺序、角度和半径。下面是历史 001 阵列的例子。其通道顺序和几何信息对应这份录音。

~~~sh
OMP_NUM_THREADS=4 OMP_PROC_BIND=close taskset -c 4-7 ./out/wpe_online_cpp \
  input.f32 output.f32 8 8 0,1,5,4,2,3 294 0.0476
~~~

设 `BYPASS=1` 只执行波束。默认使用六个位置、通道 0–5、86° 和 42 mm。通道不足或有效麦克风信号过低时，程序输出第一个通道。切换到其他设备时需使用该设备的麦克风映射与阵列几何。

[RK3588 实时增强链](../../scripts/board/LIVE.md)接收单声道 S16_LE PCM，可将回放结果用 FFmpeg 从 float32 转为 S16_LE。
