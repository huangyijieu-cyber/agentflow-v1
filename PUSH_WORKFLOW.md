# Git 推送流程

本项目通过 Clash Verge 访问 GitHub。推送 `idea` 分支前，先同步远端，再推送本地提交：

```bash
git -c http.proxy=http://127.0.0.1:7897 pull --rebase origin idea
git -c http.proxy=http://127.0.0.1:7897 push origin idea
```

不要使用强制推送，以免覆盖远端提交。
