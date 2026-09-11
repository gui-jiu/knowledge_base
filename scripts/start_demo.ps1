# 掌柜智库 - 一键启动演示脚本
# 用法：在项目根目录执行  powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  掌柜智库 RAG 系统 - 启动演示" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 设置 PYTHONPATH
$env:PYTHONPATH = $ProjectRoot

Write-Host ""
Write-Host "[1/2] 启动导入服务 (端口 8000)..." -ForegroundColor Yellow
Start-Process -FilePath "uv" `
    -ArgumentList "run","python","-m","uvicorn","app.import_process.api.import_service:app","--host","0.0.0.0","--port","8000" `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Normal

Start-Sleep -Seconds 3

Write-Host "[2/2] 启动查询服务 (端口 8001)..." -ForegroundColor Yellow
Start-Process -FilePath "uv" `
    -ArgumentList "run","python","-m","uvicorn","app.query_process.api.query_service:app","--host","0.0.0.0","--port","8001" `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Normal

Start-Sleep -Seconds 5

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  服务已启动！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  文档导入页: http://localhost:8000/import.html" -ForegroundColor White
Write-Host "  智能问答页: http://localhost:8001/chat.html" -ForegroundColor White
Write-Host "  接口文档:   http://localhost:8001/docs" -ForegroundColor White
Write-Host ""
Write-Host "  关闭服务: 直接关闭弹出的两个命令行窗口" -ForegroundColor Gray
Write-Host "==========================================" -ForegroundColor Green
