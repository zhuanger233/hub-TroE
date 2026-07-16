"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. geocode_city：城市名 → 结构化经纬度
  2. get_weather_by_coordinates：经纬度 → 天气
  3. get_weather 保留为兼容入口，供尚未改造的 MCP / CLI 模式继续使用

Function Call 模式会把前两个函数分别暴露给 LLM，让模型根据第一个工具的
结果生成第二个工具调用，从而形成链式工具调用。
"""

import json

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def geocode_city(city: str) -> str:
    """
    把城市名称解析为经纬度，返回便于下一个工具消费的 JSON 字符串。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        JSON 字符串。成功时包含 latitude、longitude 和 location_name；
        失败时 ok=false，并包含 error。
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            # 中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
            # 而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
            # 策略：先按用户输入查；若命中的只是低级行政点（feature_code 纯 PPL），
            # 且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。
            def _geocode(name: str):
                resp = client.get(GEOCODING_URL, params={
                    "name": name, "count": 10, "language": "zh", "format": "json",
                })
                resp.raise_for_status()
                return resp.json().get("results") or []

            results = _geocode(city)
            is_low_admin = all(
                str(r.get("feature_code", "")).startswith("PPL")
                and not str(r.get("feature_code", "")).startswith("PPLA")
                for r in results
            ) if results else True
            has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
            if is_low_admin and not has_suffix:
                retry = _geocode(city + "市")
                if retry:
                    results = retry
    except httpx.HTTPError as e:
        return json.dumps({
            "ok": False,
            "query": city,
            "error": f"城市坐标查询失败：{e}",
        }, ensure_ascii=False)

    if not results:
        return json.dumps({
            "ok": False,
            "query": city,
            "error": f"未找到城市 '{city}'，请尝试其他写法",
        }, ensure_ascii=False)

    # 在候选里优先取行政级别更高的，其次取有人口数据的，避免落到同名小村庄。
    def _rank(r):
        fc = str(r.get("feature_code", ""))
        admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
        pop = r.get("population") or 0
        return (admin_priority, pop)

    loc = max(results, key=_rank)
    city_name = loc.get("name", city)
    country = loc.get("country", "")
    admin1 = loc.get("admin1", "")
    location_name = " ".join(part for part in (country, admin1, city_name) if part)

    return json.dumps({
        "ok": True,
        "query": city,
        "location_name": location_name,
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
    }, ensure_ascii=False)


def get_weather_by_coordinates(
    latitude: float,
    longitude: float,
    location_name: str | None = None,
) -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    Args:
        latitude: 纬度，范围 -90 到 90
        longitude: 经度，范围 -180 到 180
        location_name: 可选，由 geocode_city 返回的地点名称，仅用于展示

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    if not -90 <= latitude <= 90:
        return f"纬度超出范围：{latitude}，应在 -90 到 90 之间"
    if not -180 <= longitude <= 180:
        return f"经度超出范围：{longitude}，应在 -180 到 180 之间"

    try:
        with httpx.Client(timeout=10.0) as client:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
    except httpx.HTTPError as e:
        return f"天气数据获取失败：{e}"

    data = weather_resp.json()
    cur = data["current"]
    daily = data["daily"]
    weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
    display_name = location_name or "指定坐标"

    lines = [
        f"【{display_name}】天气报告",
        f"坐标：{latitude:.2f}°N, {longitude:.2f}°E",
        "",
        f"当前天气：{weather_desc}",
        f"  温度：{cur['temperature_2m']}°C",
        f"  相对湿度：{cur['relative_humidity_2m']}%",
        f"  风速：{cur['wind_speed_10m']} km/h",
        "",
        "未来3天预报：",
    ]
    forecast_days = min(
        3,
        len(daily.get("time", [])),
        len(daily.get("weather_code", [])),
        len(daily.get("temperature_2m_max", [])),
        len(daily.get("temperature_2m_min", [])),
        len(daily.get("precipitation_sum", [])),
    )
    for i in range(forecast_days):
        day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
        lines.append(
            f"  {daily['time'][i]}：{day_desc}，"
            f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
            f"降水 {daily['precipitation_sum'][i]} mm"
        )

    return "\n".join(lines)


def get_weather(city: str) -> str:
    """兼容旧入口：在函数内部依次调用坐标查询和天气查询。"""
    location_result = json.loads(geocode_city(city))
    if not location_result.get("ok"):
        return location_result.get("error", "城市坐标查询失败")
    return get_weather_by_coordinates(
        latitude=location_result["latitude"],
        longitude=location_result["longitude"],
        location_name=location_result.get("location_name"),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    print(get_weather(args.city))
