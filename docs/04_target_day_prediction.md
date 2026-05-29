# 目标日预测模式

## 触发条件

当某个交易日有广西供需类原始数据，但缺少日前价格或实时价格时，系统进入目标日预测模式。

当前已验证样例：`2026-05-26`，该日有供需曲线，日前价格和实时价格均缺失。

## 日前价格预测

日前价格采用混合集成架构：

- XGBoost 分量：使用历史日前价格、小时、星期、统调负荷、新能源、水电、省间联络线、备用、供需比例、历史同小时价格等特征训练。
- LSTM 序列分量：已接入 PyTorch LSTM，使用过去 14 天同小时日前价格序列训练，输出目标日 24 小时日前价预测。
- 相似同小时基准：根据最近历史同小时均值提供兜底锚点。

集成权重：

- XGBoost：55%
- LSTM/序列分量：30%
- 相似同小时基准：15%

接口：

```text
GET /api/forecast/day-ahead/{market_date}
```

## E 盘运行环境

深度学习依赖安装在：

```text
E:\power-trading-assistant-venv
```

启动服务：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_server_e_drive.ps1
```

当前 LSTM 运行状态可在预测接口的 `components.lstm` 字段查看：

```json
{
  "lstm": "pytorch_lstm"
}
```

如果没有使用 E 盘环境启动，且当前 Python 没有安装 PyTorch，系统会自动退回 `sequence_fallback`。

## 实时价格预测方案

实时价格不固定单一模型，页面提供下拉框让用户选择：

- 价差跟随法 `spread_follow`：用相似日和历史实时-日前价差修正日前预测价，适合作为默认方案。
- 相似日直接法 `similar_direct`：直接参考相似供需小时的实时价格，适合供需结构重复性强的日期。
- 松紧度修正法 `tightness_adjusted`：用供需松紧度修正日前预测价，适合缺少稳定实时样本时使用。
- 保守区间法 `conservative_range`：扩大实时价格区间，适合风险厌恶型报价。

接口：

```text
GET /api/report/{market_date}?real_time_method=spread_follow
GET /api/report/{market_date}?real_time_method=similar_direct
GET /api/report/{market_date}?real_time_method=tightness_adjusted
GET /api/report/{market_date}?real_time_method=conservative_range
```
