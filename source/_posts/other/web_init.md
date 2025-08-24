---
title: ubuntu搭建技术博客指南
categories:
  - 其它
description: web网页搭建实战记录
---

<!-- more -->

- [1. 安装 Hexo 环境](#1-安装-Hexo-环境)
- [2. 选择与配置 Hexo 主题](#2-选择与配置-Hexo-主题)
- [3. 撰写与管理博客内容](#3-撰写与管理博客内容)
- [4. SEO 优化](#4-SEO-优化)
- [5. 博客部署](#5-博客部署)
- [6. 维护与优化](#6-维护与优化)


本指南详细介绍了如何在 Ubuntu 服务器上搭建并部署一个 Hexo 技术博客，包括从环境安装到后期维护的完整步骤。

### 1. 安装 Hexo 环境

搭建 Hexo 博客首先需要安装 Node.js（Hexo 基于 Node.js）、npm、Git 以及 Hexo CLI 工具。请按照以下步骤配置环境：

#### 安装 Node.js 和 npm：
在 Ubuntu 上，通过包管理器或 Node 官方仓库安装 Node.js。建议安装 LTS 版本（如 Node 14+）。执行以下命令添加 NodeSource 仓库并安装 Node.js：

```bash
curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

安装完成后，检查版本以确保 Node 正常可用：

```bash
node -v  # 应返回类似 v18.20.6 的版本号
npm -v   # 验证 npm 是否正常安装
```

#### 安装 Git：
Git 是 Hexo 部署和备份的常用工具。Ubuntu 通常预装 Git，若未安装，请执行：

```bash
sudo apt-get install -y git
```

安装后，配置 Git 的全局用户名和邮箱：

```bash
git config --global user.name "Your Name"
git config --global user.email "youremail@example.com"
```

#### 安装 Hexo CLI：
通过 npm 全局安装 Hexo CLI：

```bash
sudo npm install -g hexo-cli
```

安装成功后，通过 `hexo -v` 检查版本，确保 Hexo CLI 可用。

#### 初始化 Hexo 博客：
选择博客文件夹（例如 `/var/www/hexo` 或当前用户主目录下的 `my-blog` 文件夹），并在该目录下初始化 Hexo 博客：

```bash
sudo mkdir -p /var/www/hexo && sudo chown $USER:$USER /var/www/hexo
cd /var/www/hexo
hexo init
npm install
```

初始化完成后，Hexo 会生成默认的博客结构，包括 `_config.yml` 配置文件、`scaffolds/` 模板目录、`source/` 内容目录和 `themes/` 主题目录等。可以通过运行 `hexo server` 预览本地博客。

#### 开启防火墙：
为了确保服务器安全，建议开启防火墙。Ubuntu 自带 UFW 防火墙，可以开启 SSH、HTTP(S) 以及 Hexo 默认预览端口 4000：

```bash
sudo apt-get install ufw  
sudo ufw allow "OpenSSH"  
sudo ufw allow 4000  
sudo ufw allow http  
sudo ufw allow https  
sudo ufw enable
```

### 2. 选择与配置 Hexo 主题

Hexo 默认主题为 Landscape，但为了打造一个简洁美观的技术博客，我们推荐使用 NexT 主题，它功能强大且外观优雅。以下是主题的安装和配置步骤：

#### 获取 NexT 主题：
在 Hexo 博客根目录下执行以下命令来克隆 NexT 主题：

```bash
cd /var/www/hexo
git clone https://github.com/theme-next/hexo-theme-next themes/next
```

#### 修改主题配置：
克隆完成后，打开 `_config.yml` 配置文件，将 `theme` 配置从默认的 `landscape` 改为 `next`：

```yaml
# _config.yml
theme: next
```

#### 安装主题依赖：
根据需要安装 NexT 主题的依赖，并启用你所需的功能。

#### 生成常用页面：
为了完善网站结构，使用以下命令生成标签、分类、归档等页面：

```bash
hexo new page "tags"
hexo new page "categories"
hexo new page "archives"
hexo new page "about"
```

编辑每个页面的 `index.md`，在 Front-matter 中指定页面类型：

```yaml
title: 标签
date: 2025-03-06 15:00:00
type: "tags"
```

#### 导航栏菜单定制：
在 `themes/next/_config.yml` 中找到 `menu` 设置，并添加新创建的页面：

```yaml
menu:
  home: / || home
  categories: /categories/ || th
  tags: /tags/ || tags
  archives: /archives/ || archive
  about: /about/ || user
```

保存修改后，重新生成站点，新的导航栏菜单即会显示。

### 3. 撰写与管理博客内容

Hexo 使用 Markdown 格式来撰写文章，非常适合技术博客。以下是如何管理和编写文章的步骤：

#### 新建博文：
使用 Hexo CLI 创建新的文章：

```bash
hexo new "文章标题"
```

这将在 `source/_posts/` 目录下创建一个 Markdown 文件，文件的开头是 Front-matter，用于配置文章的元数据（如标题、日期、分类和标签等）：

```yaml
title: 深度学习入门指南  
date: 2025-03-06 15:00:00  
categories:  
  - 人工智能  
  - 深度学习  
tags:  
  - 神经网络  
  - 入门教程 
```

#### 使用 Markdown 撰写内容：
在 Front-matter 下方，用 Markdown 语法撰写正文。Hexo 默认支持 GFM（GitHub Flavored Markdown），可以方便地书写格式，例如：

```markdown
# 一级标题
## 二级标题
**粗体**、*斜体*强调
```

#### 插入图片和资源：
启用 `post_asset_folder: true` 后，每篇文章会有独立的资源目录。可以将图片文件放入该文件夹，并在文章中引用：

```markdown
![](my-post/images/example.png)
```

#### 草稿管理与发布：
启用草稿功能后，新创建的文章会先放在 `_drafts/` 下。完成后，使用 `hexo publish "文章标题"` 将其发布。

#### 文章结构和分页：
Hexo 支持文章分类和标签自动整理。你也可以通过 `<!-- more -->` 来手动截断摘要，提高首页加载速度。

### 4. SEO 优化

为了让更多人看到你的技术博客，进行 SEO（搜索引擎优化）非常重要。以下是一些优化措施：

#### 站点标题与元信息：
在 `_config.yml` 中填写有助于 SEO 的站点基本信息，包括 `title`（标题）、`description`（描述）和 `keywords`（关键词）。

#### 链接优化：
修改永久链接格式，简化 URL 结构：

```yaml
permalink: :category/:title/
```

#### 站点地图：
生成站点地图帮助搜索引擎抓取所有页面：

```bash
npm install hexo-generator-sitemap hexo-generator-baidu-sitemap --save
```

并在 `_config.yml` 中添加配置：

```yaml
sitemap:
  path: sitemap.xml
baidusitemap:
  path: baidusitemap.xml
```

#### 机器人协议：
在 `source/` 目录下创建 `robots.txt` 文件，并写入规则：

```makefile
User-agent: *
Allow: /
Disallow: /admin/
Sitemap: https://你的域名/sitemap.xml
```

好的，以下是我重新生成并保持完整的 Hexo Deploy 自动部署部分，确保没有省略任何细节：

---

### 5. 博客部署

完成内容创作和优化后，就需要将博客部署上线。Hexo 生成的是纯静态网页，可以部署在任意静态服务器或托管平台上。这里介绍在 Ubuntu 服务器上使用 Nginx 部署的方案，并讨论 Nginx 配置和 Git 自动部署方法。

#### 本地生成静态文件：
Hexo 提供命令将 Markdown 内容生成静态网页。一般在本地或服务器上运行：

```bash
hexo clean        # 清理上次生成的文件
hexo generate (hexo g)   # 生成最新静态网页
```

生成的文件位于博客目录下的 `public/` 文件夹，其中包含博客的所有 HTML、CSS、JS、图片等静态资源。这个 `public` 文件夹即是最终部署的网站内容。

#### Nginx 部署静态站点：
在服务器上安装 Nginx 并配置站点，以提供 Web 服务：

安装 Nginx：

```bash
sudo apt-get install -y nginx
```

安装后启动 Nginx 服务：

```bash
sudo systemctl start nginx  # 可设置开机自启
```

配置站点：在 `/etc/nginx/sites-available/` 目录下创建配置文件，如 `hexo.conf`，内容如下：

```nginx
server {
    listen 80;
    server_name example.com;  # 将此替换为你的域名或服务器IP

    root /var/www/hexo/public;
    index index.html index.htm;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

上述配置指定服务器监听 80 端口，`server_name` 为你的域名（需要将域名解析指向该服务器）。`root` 指向 Hexo 生成的 `public` 目录，`index` 声明默认首页文件。

启用站点配置：将配置文件链接到 `sites-enabled`：

```bash
ln -s /etc/nginx/sites-available/hexo.conf /etc/nginx/sites-enabled/
nginx -t  # 测试配置语法正确性
systemctl reload nginx  # 重新加载 Nginx 配置
```

执行以上命令后，博客站点即可通过域名访问。如果暂时没有域名，使用服务器 IP 也能访问（此时可将 `server_name` 改为 `_` 通配符）。

#### 配置 HTTPS（可选）：
建议为博客配置 SSL 证书。可以使用 Certbot 获取 Let’s Encrypt 免费证书。步骤如下：

```bash
apt-get install -y certbot python3-certbot-nginx  
certbot --nginx -d example.com -d www.example.com
```

按提示完成域名所有权验证后，Certbot 会自动生成证书并配置 Nginx 将站点升级为 HTTPS。

#### Hexo Deploy 自动部署：
每次更新内容后都要重新生成并上传文件，使用 Hexo 的部署功能可以简化流程。Hexo 支持多种部署方式，其中 Git 部署是常用方案之一。基本思路是利用 Git 把生成的静态文件推送到服务器或托管服务。概括了这种思路：在服务器上安装 Nginx 提供网页服务，用 Git 实现代码上传自动化，这样本地执行一次 `hexo d`（deploy）就能让网站更新。

##### 推送到远程托管：
将博客静态文件部署到像 GitHub Pages、Coding Pages 这类平台。这需要在 `_config.yml` 中配置：

```yaml
deploy:
  type: git
  repo: https://github.com/yourname/yourrepo.git
  branch: main  # 或 gh-pages 分支等
```

然后运行 `hexo generate && hexo deploy`，Hexo 会把 `public` 文件夹内容推送到指定仓库的分支。对于 GitHub Pages，如果 repo 是 `yourname.github.io` 则直接用主分支；若是项目仓库，可以用 `gh-pages` 分支托管。

部署后，GitHub Pages 服务将托管你的静态博客，你可以使用自定义域名绑定它。但注意：如果你希望博客运行在自己的服务器上（而非第三方平台），则这种方案不涉及你的服务器 Nginx。另外，国内访问 GitHub Pages 可能不稳定，需结合实际情况考虑。

##### 推送到自己服务器：
搭建属于自己的 Git 自动化部署流程，实现将本地更新一键部署到服务器。步骤如下：

1. 在服务器上创建一个裸仓库（bare repository），用于接收推送。例如创建 `/home/git/hexo.git` 裸仓库。
2. 编写 Git 钩子（`post-receive`）：裸仓库的 `hooks/post-receive` 脚本会在收到新推送时执行。脚本内容可以是将更新的内容检出到 Nginx 目录。例如：

   ```bash
   GIT_WORK_TREE=/var/www/hexo git checkout -f  # 将仓库内容强制检出到 /var/www/hexo
   cd /var/www/hexo && hexo generate            # （若推送的是源码而非生成文件，则需要在服务器执行生成）
   ```

3. 给脚本可执行权限：

   ```bash
   chmod +x post-receive
   ```

   这样，每当推送到该仓库时，它就会把更新部署到博客目录并生成最新页面。

4. 本地 Hexo 配置部署：将 `_config.yml` 中的 `deploy.repo` 设置为上述裸仓库的地址（通过 SSH）。例如：

   ```yaml
   deploy:
     type: git
     repo: ssh://[email protected]/home/git/hexo.git
     branch: master
   ```

5. 然后执行 `hexo clean && hexo deploy`。Hexo 会通过 Git 推送到服务器仓库，触发 `post-receive` 钩子，实现自动部署。完成后，Nginx 会立刻提供新内容服务，无需手动登录服务器操作。

通过这种方案，可以在本地写好文章后一条命令完成部署，非常高效。许多开源博客部署脚本和工具也是基于类似原理实现的。初次设置可能稍显繁琐，但一旦配置成功，日常更新将非常便捷。

**提示：** 使用 Git 自动部署需确保服务器开放 Git 所用的 SSH 端口（默认为 22），并配置好公钥免密登录，以便 Hexo 在本地能顺利推送到服务器。如果你的服务器 SSH 端口不是 22，可在部署配置中加入端口号或在 `.ssh/config` 中配置别名。对于不熟悉 Git 钩子的新手，也可以考虑使用简单的 `rsync` 脚本同步文件或借助 CI 平台实现部署，但原理类似。

--- 

### 6. 维护与优化

博客搭建完成并不意味着一劳永逸，定期的维护和优化能保证博客稳定、安全，并持续提升用户体验。

- **插件扩展：** Hexo 拥有丰富的插件生态，可根据需要安装插件以增强功能。
- **备份与版本控制：** 使用 Git 管理博客源码，定期备份。
- **更新与升级：** 关注 Hexo 的版本更新、插件更新等。
