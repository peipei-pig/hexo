---
title: "ubuntu常见shell命令"
mathjax: true
categories:
  - 其它
tags:
  - shell
description: 记录最常用的shell命令
---

<!-- more -->

---

## 1. 磁盘占用与排序（du/sort）

### 常用写法

```bash
# 按“当前目录的直接子项”汇总（人类可读），并按大小倒序
du -h --max-depth=1 . | sort -hr

# 仅统计每个条目总大小（不显示子层级），并对条目排序
du -sh -- * | sort -h
```
---

## 2. 文本搜索（grep）

### 基础

```bash
grep "keyword" file.txt          # 在单个文件中查找
grep -n "keyword" file.txt       # 显示行号
grep -i "keyword" file.txt       # 忽略大小写
```

### 目录递归与上下文

```bash
grep -rin --color=auto "keyword" .      # 递归、忽略大小写、行号、高亮
grep -nC 3 "keyword" file.txt           # 上下各 3 行
grep -nA 2 "keyword" file.txt           # 后 2 行
grep -nB 2 "keyword" file.txt           # 前 2 行
```

### 精确匹配与正则

```bash
grep -rw "\<token\>" .                  # 按“整词”匹配
grep -E "err(or)?|fail(ed)?" app.log    # 扩展正则
grep -rF "literal*text" .               # 纯字符串（不当正则），更快
```

### 排除文件/目录

```bash
grep -rin "keyword" . \
  --exclude-dir={.git,node_modules,dist} \
  --exclude="*.min.js"
```

---

## 3. 文件路径查找（find/locate）

### find：灵活但实时扫描（慢）

```bash
# 按文件名（大小写不敏感）
find /path -type f -iname "*name*"

# 限制搜索深度
find . -maxdepth 2 -type d -name "build"

# 查找大文件（> 100MB）并按大小降序列出前 20 个
find /var -type f -size +100M -printf '%s\t%p\n' | sort -nr | head -20

# 查找最近 1 天内修改的文件
find . -type f -mtime -1

# 对结果执行命令（安全处理空格）
find . -type f -name "*.log" -print0 | xargs -0 gzip
```

> **跳过系统目录且压制报错**

```bash
find / \( -path /proc -o -path /sys -o -path /run \) -prune -o \
  -type f -name "*.conf" -print 2>/dev/null
```

### locate/plocate：基于索引（快）

```bash
sudo apt-get install -y plocate
sudo updatedb                 # 通常自动定时更新
locate filename_or_pattern
```

---

## 4. 常见网络工具安装包

```bash
# ping
sudo apt-get install -y iputils-ping

# ifconfig（老工具，仍常见）
sudo apt-get install -y net-tools
# 现代替代：ip（通常已自带于 iproute2）
ip addr
ip link
ip route

# killall
sudo apt-get install -y psmisc
```

---

## 5. 进程查杀（kill/pkill/killall）

```bash
ps -ef | grep python3 | awk '{print $2}' | xargs kill -9
```

### 更安全的做法

```bash
# 优雅终止（SIGTERM）；无 PID 时不执行 (-r)
pgrep -f python3 | xargs -r kill

# 直接按名称匹配（优雅终止），必要时再 -9
pkill -f python3
pkill -9 -f python3

# 避免匹配到 grep 自身
ps -ef | grep '[p]ython3' | awk '{print $2}' | xargs -r kill
```

> 建议先尝试 `SIGTERM`（默认），无响应再用 `SIGKILL`（`-9`）。

---

## 6. 高频命令清单与示例

### 系统/资源

```bash
top                     # 实时概览
htop                    # 更友好（需：sudo apt-get install -y htop）
free -h                 # 内存
df -h                   # 磁盘分区容量
du -sh * | sort -h      # 目录占用
uname -a                # 内核信息
lsb_release -a          # 发行版信息
```

### 进程/网络

```bash
ps aux | less
pstree -p               # 进程树（需：sudo apt-get install -y psmisc）
lsof -i :8080           # 端口占用（需：sudo apt-get install -y lsof）
ss -lntp                # 监听端口 + 进程
```

### 文本/日志

```bash
less file.log
tail -f file.log
wc -l file.txt
sort file | uniq -c | sort -nr
cut -d',' -f1,3 file.csv
sed -n '1,20p' file.txt
awk -F: '{print $1,$3}' /etc/passwd
```

### 文件/归档/传输

```bash
tar -czf logs.tgz logs/        # 压缩
tar -xzf logs.tgz              # 解压
zip -r src.zip src/            # zip（需：sudo apt-get install -y zip unzip）
rsync -av --progress src/ dst/
scp file user@host:/path/
```

### 权限/链接

```bash
chmod +x run.sh
chown user:group file
ln -s /real/path link_name
```

### 服务与日志（systemd）

```bash
systemctl status nginx
sudo systemctl start nginx
journalctl -u nginx --since "1 hour ago"
```

### 其他

```bash
which python3
command -v node
date "+%F %T"
nohup python3 app.py >out.log 2>&1 &
tmux new -s work              # 终端复用（需：sudo apt-get install -y tmux）
```

---

## 7. 小贴士与常见坑

* **隐藏文件**：`*` 不匹配隐藏项，可用 `.* *` 组合或开启 `dotglob`。
* **防止参数被当作选项**：当文件名以 `-` 开头时加 `--`，如 `rm -- -weirdfile`。
* **xargs 安全**：二进制文件/空格用 `-0` 配合 `-print0`；无结果时不执行用 `-r`。
* **优雅停服务优先**：`kill -TERM` → 不行再 `kill -KILL`。
* **权限**：系统目录操作慎用 `sudo`，写前先 `ls`/`du`/`stat` 确认。
* **grep 正则 vs 字符串**：纯文本匹配更稳更快用 `-F`。
* **find 性能**：大目录用 `-maxdepth` 限制层级或改用 `locate/plocate`。

---

