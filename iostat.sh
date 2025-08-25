iostat -d -k 1 | awk '
BEGIN {
    # 初始化总读写量
    total_kb_read = 0;
    total_kb_wrtn = 0;
    # 标志，用于跳过iostat的初始“自启动以来”报告
    # iostat -d 1 会先输出一个总报告，然后才开始输出间隔报告
    # 我们需要跳过这个总报告和它后面的一个空白行以及下一个“Device”头
    skip_initial_report_block = 1;
    header_found = 0; # 用于判断是否是第二个 "Device" 头
}

# 匹配 "Device" 行
/^Device/ {
    header_found++;
    if (header_found == 2) {
        # 找到第二个 "Device" 头，表示后续是间隔数据了
        skip_initial_report_block = 0;
        print "----------------------------------------------------------------------------------------------------";
        printf "%-10s %12s %12s %16s %16s\n", "Device", "Read BW(KB/s)", "Write BW(KB/s)", "Total Read(MB)", "Total Write(MB)";
        print "----------------------------------------------------------------------------------------------------";
    }
    next; # 跳过当前行（即“Device”头本身）
}

# 忽略初始报告的行，包括空白行、Linux版本信息、avg-cpu等
skip_initial_report_block || /^\s*$/ || /^Linux/ || /^avg-cpu/ {
    next;
}

# 处理数据行
{
    # iostat -d -k 输出的列：
    # $1: Device
    # $4: kB_read/s (读带宽)
    # $5: kB_wrtn/s (写带宽)
    # $6: kB_read (本间隔内读取的总KB)
    # $7: kB_wrtn (本间隔内写入的总KB)

    device_name = $1;
    read_bandwidth = $4;
    write_bandwidth = $5;
    interval_kb_read = $6;
    interval_kb_wrtn = $7;

    # 累加总数据量
    total_kb_read += interval_kb_read;
    total_kb_wrtn += interval_kb_wrtn;

    # 打印结果：带宽（每秒刷新）、累积总量（不刷新）
    # 将累积总量从KB转换为MB，方便阅读
    printf "%-10s %12.2f %12.2f %16.2f %16.2f\n",
           device_name,
           read_bandwidth,
           write_bandwidth,
           total_kb_read / 1024,  # KB转MB
           total_kb_wrtn / 1024;  # KB转MB
}
'

