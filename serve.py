# -*- coding: utf-8 -*-
"""
Server chạy ổn định bằng waitress (thay cho server dev của Flask).
Xử lý upload ảnh/video lớn từ điện thoại không bị "mất kết nối".

Cách dùng:
    py serve.py            -> HTTP thường (upload ảnh/video, dùng cho điện thoại)
    py serve.py --https    -> HTTPS tự ký (bắt buộc khi dùng CAMERA điện thoại)

Điện thoại cùng WiFi truy cập:  http://<IP-laptop>:5000
(hoặc https://<IP-laptop>:5000 nếu chạy --https, rồi bấm "Tiếp tục" khi cảnh báo)
Xem IP laptop bằng lệnh: ipconfig  (dòng IPv4 Address)
"""
import os
import socket
import sys

import app as flask_app

PORT = 5000


def local_ip():
    """Đoán IP LAN của máy để in ra cho tiện."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def ensure_cert():
    """Tạo chứng chỉ tự ký (cert.pem/key.pem) nếu chưa có."""
    cert, key = "cert.pem", "key.pem"
    if os.path.exists(cert) and os.path.exists(key):
        return cert, key
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    print("* Đang tạo chứng chỉ tự ký (chỉ chạy 1 lần)...")
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "smokecam")])
    builder = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(k.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
    )
    crt = builder.sign(k, hashes.SHA256())
    open(cert, "wb").write(crt.public_bytes(serialization.Encoding.PEM))
    open(key, "wb").write(k.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    return cert, key


def main():
    use_https = "--https" in sys.argv
    ip = local_ip()
    scheme = "https" if use_https else "http"

    print("=" * 56)
    print("  SMOKE GUARD AI — server đang chạy")
    print("=" * 56)
    print(f"  Trên laptop     : {scheme}://localhost:{PORT}")
    print(f"  Trên điện thoại : {scheme}://{ip}:{PORT}   (cùng WiFi)")
    if use_https:
        print("  (HTTPS tự ký — điện thoại cảnh báo thì bấm Nâng cao → Tiếp tục)")
    else:
        print("  Chế độ HTTP: upload ảnh/video OK. Muốn dùng CAMERA điện")
        print("  thoại thì chạy:  py serve.py --https")
    print("  Nhấn Ctrl+C để dừng.")
    print("=" * 56)

    if use_https:
        # HTTPS: dùng server dev của Flask với chứng chỉ tự ký thật (ổn định hơn 'adhoc')
        import ssl
        cert, key = ensure_cert()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        flask_app.app.run(host="0.0.0.0", port=PORT, ssl_context=ctx, threaded=True)
    else:
        # HTTP: waitress — chịu tải upload file lớn tốt, không "mất kết nối"
        from waitress import serve
        serve(flask_app.app, host="0.0.0.0", port=PORT, threads=8,
              max_request_body_size=210 * 1024 * 1024, channel_timeout=600)


if __name__ == "__main__":
    main()
