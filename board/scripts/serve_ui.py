"""本机UI与板端HTTP API代理；界面不读取参考预测来生成文字。"""
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import argparse,json

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def handle_request(self):
        if self.path.startswith('/api/'):
            data=self.rfile.read(int(self.headers.get('Content-Length','0'))) if self.command=='POST' else None
            req=Request(self.server.board+self.path,data=data,method=self.command,headers={'Content-Type':'application/json'})
            try:
                with urlopen(req,timeout=5) as response:body=response.read();code=response.status
            except HTTPError as e:body=e.read();code=e.code
            except Exception as e:body=json.dumps({'error':str(e)}).encode();code=503
            self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
        elif self.path=='/':
            body=(Path(__file__).parent/'ui/index.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
        else:self.send_error(404);return
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    do_GET=handle_request
    do_POST=handle_request
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board-api',default='http://192.168.2.99:8765');p.add_argument('--port',type=int,default=8766);a=p.parse_args()
    s=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);s.daemon_threads=True;s.board=a.board_api.rstrip('/')
    print(f'B2Tnet UI http://127.0.0.1:{a.port}',flush=True);s.serve_forever()
