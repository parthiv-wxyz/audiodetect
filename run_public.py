import os
import sys
import re
import socket
import time
import subprocess
import threading

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def find_available_port(start_port=7860):
    for port in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port

CLOUDFLARED_PATH = os.path.abspath("cloudflared.exe")
PORT = find_available_port(7860)

def main():
    print(f"[1/2] Launching Gradio Forensic App on port {PORT}...")
    from gradio_app import demo, theme, custom_css
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=PORT,
        share=False,
        prevent_thread_lock=True,
        show_error=True,
        theme=theme,
        css=custom_css
    )
    print(f"[1/2] Local server active at http://127.0.0.1:{PORT}")

    print(f"[2/2] Opening Cloudflare Tunnel to http://127.0.0.1:{PORT}...")
    cmd = [CLOUDFLARED_PATH, "tunnel", "--url", f"http://127.0.0.1:{PORT}"]
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )
    
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    public_url = None
    
    for line in iter(proc.stdout.readline, ''):
        line_clean = line.strip()
        match = url_pattern.search(line_clean)
        if match and not public_url:
            public_url = match.group(0)
            print("\n" + "="*70)
            print("🚀 PUBLIC ACCESS LINK ACTIVE:")
            print(f"👉 {public_url}")
            print("="*70 + "\n")
            with open("public_url.txt", "w", encoding="utf-8") as f:
                f.write(public_url + "\n")
            sys.stdout.flush()

    proc.wait()

if __name__ == "__main__":
    main()
