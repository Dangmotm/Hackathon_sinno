# Chạy game trên localhost

## 1) Cài thư viện

```bash
pip install -r requirements-web.txt
```

Nếu bạn đã có sẵn môi trường train cũ thì thường chỉ cần thêm Flask:

```bash
pip install flask
```

## 2) Chạy server local

```bash
python run_web.py
```

hoặc

```bash
python web_server.py
```

## 3) Mở trình duyệt

Vào:

```text
http://127.0.0.1:5000
```

## 4) Điều khiển

- `W A S D`: di chuyển
- `Mouse`: ngắm
- `Left Click`: bắn
- `Space`: blink
- `Q` hoặc `Right Click`: khiên
- `R`: reset round

## Các file mới đã thêm

- `web_server.py`: Flask server cho localhost
- `run_web.py`: file chạy nhanh
- `templates/index.html`: giao diện web
- `static/css/base.css`: layout, glass UI, responsive
- `static/css/game.css`: HUD, overlay, thanh máu, control panel
- `static/js/app.js`: render canvas + input + gọi API step/reset

## Lưu ý

- Bản này **giữ nguyên logic game Python** của bạn.
- CSS chỉ lo giao diện web; phần nhân vật, đạn, map được vẽ đẹp hơn bằng `canvas` trong browser.
- Nếu máy chưa cài `stable-baselines3` hoặc không load được checkpoint, server sẽ tự fallback sang scripted bot.
