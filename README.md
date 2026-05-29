# Guangxi Spot Trading Assistant

广西现货电力交易辅助决策平台，面向售电公司每日现货报价和风险判断。系统读取本地原始 Excel 数据，自动整理日前价格、实时价格、负荷、新能源、水电、省间联络线等信息，并生成目标日价格预测、供需边际分析和小时级报价建议。

## 当前能力

- 自动读取广西原始现货数据目录。
- 自动识别 96 点和 24 点数据。
- 忽略整理表，只使用原始数据表格。
- 展示日前价格、实时价格、价差、历史趋势。
- 供需边际曲线口径：
  - 供给：水电 + 新能源出力
  - 需求：统调负荷 + 省间联络线
- 支持目标日预测模式：
  - 日前价格：XGBoost + PyTorch LSTM + 相似小时基准
  - 实时价格：可选择价差跟随法、相似日直接法、松紧度修正法、保守区间法
- 支持日前价格模型回测：
  - MAE
  - RMSE
  - 高价小时命中率
  - 低价小时命中率
  - XGBoost、LSTM、混合集成模型对比

## 目录结构

```text
backend/       FastAPI 后端、导入器、分析与预测模型
frontend/      单页网页看板
docs/          数据口径、接口和预测模式说明
scripts/       本地启动脚本
```

以下内容不会上传到 GitHub：

```text
data/
现货交易电网信息/
repo-read/
```

## 本地数据放置

把原始数据放在项目根目录下：

```text
现货交易电网信息/
```

系统会自动跳过：

```text
现货交易电网信息/0现货数据整理/
```

## E 盘环境启动

当前深度学习环境安装在：

```text
E:\power-trading-assistant-venv
```

启动服务：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_server_e_drive.ps1
```

启动后打开：

```text
http://127.0.0.1:8000/
```

## 重新导入数据

网页上点击“重新导入原始数据”，或调用接口：

```text
POST /api/import
```

## 常用接口

```text
GET /api/dates
GET /api/report/{market_date}
GET /api/prices/{market_date}
GET /api/forecast/day-ahead/{market_date}
GET /api/evaluation/day-ahead?end_date=2026-05-26&days=5
```

实时价格方案参数：

```text
GET /api/report/{market_date}?real_time_method=spread_follow
GET /api/report/{market_date}?real_time_method=similar_direct
GET /api/report/{market_date}?real_time_method=tightness_adjusted
GET /api/report/{market_date}?real_time_method=conservative_range
```

## 下一步建议

- 增加策略报告 Word/PDF 导出。
- 增加未来目标日手动上传入口。
- 增加模型训练结果持久化，避免每次回测重复训练。
- 增加日前价格和实时价格的分场景评估。
