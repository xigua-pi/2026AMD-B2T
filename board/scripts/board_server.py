#!/usr/bin/env python3
"""在 PYNQ 上运行真实 DMA 推理，并通过本机 HTTP 服务发布增量结果。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

import numpy as np
from pynq import Clocks, MMIO, Overlay, allocate

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src/ps/utils"))
from stream_decoder import StreamingDecoder, display_text

VERSION = "b2tnet-activity-sorted-v2"
SENTINEL = -32768  # 对称饱和输出不包含此值，用于检测尚未写完的 DMA 行。
CHANNELS = [0, 12, 24, 48, 72, 96, 128, 160]


class Demo:
    """单线程独占 FPGA；网页线程只读取快照、提交控制请求。"""

    def __init__(self, overlay_dir: Path, log_dir: Path, trace_dir: Path, golden: Path, ranking: Path, counters: Path):
        self.root = overlay_dir
        self.trace_dir = trace_dir
        self.golden = np.load(golden,allow_pickle=False)
        self.ranking = json.loads(ranking.read_text())["records"]
        self.order = [r["record"] for r in self.ranking]
        self.rank_by_id = {r["record"]:r for r in self.ranking}
        self.expected_counters = json.loads(counters.read_text())
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((self.trace_dir / "manifest.json").read_text())
        raw_records = {int(r["record_index"]): r for r in self.manifest["records"]}
        self.records = {i:raw_records[i] for i in self.order}
        self.condition = threading.Condition()
        self.events = deque(maxlen=512)
        self.event_id = 0
        self.cancel = threading.Event()
        self.thread = None
        self.next_record = None
        self.history = deque(maxlen=12)
        self.state = {
            "version": VERSION, "connected": True, "status": "idle",
            "board": "PYNQ-Z2 · XC7Z020", "format": "W8A8 / H16 / ACC32",
            "record": self.order[0], "rank": 1, "clock_mhz": None, "input_frames": 0,
            "total_frames": 0, "output_frames": 0, "text": "",
            "reference": "", "elapsed_seconds": 0, "error_flags": 0,
            "channels": CHANNELS, "history": [], "wave": [], "wave_start": 0,
            "source": "Stanford Brain-to-Text · HeldOutBlocks · 20 ms 分箱神经活动特征",
            "transport": "整句 DMA 连续计算 / 句内循环状态保留 / 句间重置",
            "message": "已连接板端服务，等待开始。",
        }
        self.publish("state", dict(self.state))

    def publish(self, kind: str, data: dict):
        """保留事件序号，以便浏览器断线后恢复。"""
        with self.condition:
            self.event_id += 1
            self.events.append((self.event_id, kind, data))
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(self.state)

    def update(self, **values):
        with self.condition:
            self.state.update(values)
            snapshot = dict(self.state)
        self.publish("state", snapshot)

    def start(self, record: int, clock: float, loop: bool):
        """展示时使用较低 PL 时钟，所有输入和输出仍由实际 DMA 驱动。"""
        if record not in self.records or clock not in (5.0, 10.0, 50.0):
            raise ValueError("不支持的记录或时钟；时钟可选 5、10、50 MHz。")
        if self.thread is not None and self.thread.is_alive() and self.state["status"]=="idle":
            self.thread.join(timeout=1.0)
        with self.condition:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError("FPGA 正在运行，请先停止。")
            self.cancel.clear()
            self.next_record = None
            self.thread = threading.Thread(target=self.run, args=(record, clock, loop), daemon=True)
            self.thread.start()

    def control(self, body):
        action = body.get("action")
        if action == "start":
            self.start(int(body.get("record", self.order[0])), float(body.get("clock", 5)), bool(body.get("loop", True)))
        elif action == "stop":
            self.cancel.set()
        elif action == "next":
            with self.condition:
                self.next_record = self.order[(self.order.index(int(self.state["record"]))+1)%len(self.order)]
        else:
            raise ValueError("未知控制命令")

    def load(self, clock):
        """加载已有验证版本，不修改模型和 RTL；同步设置演示 PL 时钟。"""
        from pynq.pl_server.device import Device
        # 冷启动早期的空设备探测会被 PYNQ 缓存；管理器就绪后允许重新探测。
        if not Device.devices and Path("/sys/class/fpga_manager/fpga0/firmware").exists():
            del Device._devices
        print("PYNQ devices:", Device.devices, flush=True)
        overlay = Overlay(str(self.root / "h144_dma_overlay.bit"), download=True)
        Clocks.fclk0_mhz = clock
        dma = overlay.axi_dma_0
        description = overlay.ip_dict["h144_accel_0"]
        mmio = MMIO(int(description["phys_addr"]), int(description["addr_range"]))
        if mmio.read(0x38)!=0x00020001 or mmio.read(0x40)!=2:
            raise RuntimeError("比特流版本/活动模式不匹配")
        return overlay, dma, mmio, float(Clocks.fclk0_mhz)

    def run(self, first, clock, loop):
        # PYNQ 的 DMA/中断对象需要线程自己的 asyncio 事件循环。
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        overlay = None
        try:
            self.update(status="loading", message="正在加载 B2Tnet Overlay…", error=None)
            overlay, dma, mmio, actual_clock = self.load(clock)
            bit_hash = hashlib.sha256((self.root / "h144_dma_overlay.bit").read_bytes()).hexdigest()
            self.update(clock_mhz=actual_clock, bitstream_sha256=bit_hash)
            index = first
            while not self.cancel.is_set():
                completed = self.run_record(index, dma, mmio, actual_clock)
                if not completed or not loop:
                    break
                self.update(status="between", message="本句完成，准备下一条公开数据。")
                if self.cancel.wait(1.2):
                    break
                with self.condition:
                    index = self.next_record if self.next_record is not None else self.order[(self.order.index(index)+1)%len(self.order)]
                    self.next_record = None
            self.update(status="restoring", message="恢复硬件状态。")
        except Exception as exc:
            traceback.print_exc()
            self.update(status="error", message=str(exc), error=type(exc).__name__)
        finally:
            # 停止或异常时重新配置，终止旧 DMA 后才释放 CMA 缓冲区。
            # 重新加载也将 5/10 MHz 演示时钟恢复至原 Overlay 的 50 MHz。
            if overlay is not None:
                try:
                    Overlay(str(self.root / "h144_dma_overlay.bit"), download=True)
                except Exception as exc:
                    self.update(status="error", message="恢复 Overlay 失败：" + str(exc))
            event_loop.close()
            if self.state["status"]!="error":
                self.update(status="idle",message="可选择记录重新开始。")

    def run_record(self, index, dma, mmio, clock):
        meta = self.records[index]
        path = self.trace_dir / meta["trace_file"]
        with np.load(path, allow_pickle=False) as trace:
            inputs = np.asarray(trace["input"], dtype=np.int8)
            expected = np.asarray(self.golden[f"r{index:03d}m2"], dtype=np.int16)
        if inputs.shape != (int(meta["input_frames"]), 192):
            raise ValueError("输入形状与冻结 manifest 不一致")
        decoder = StreamingDecoder(len(inputs))
        source = allocate(shape=inputs.shape, dtype=np.int8)
        output = allocate(shape=expected.shape, dtype=np.int16)
        active = False
        try:
            source[:] = inputs
            source.flush()
            output[:] = SENTINEL
            output.flush()
            mmio.write(0x08, int(meta["session_id"]))
            mmio.write(0x0C, len(inputs))
            mmio.write(0x10, expected.size)
            self.update(status="running", record=index, rank=self.rank_by_id[index]["rank"], session=meta["session"],
                        input_frames=0, total_frames=len(inputs), output_frames=0,
                        text="", reference="", elapsed_seconds=0, error_flags=0,
                        wave=[], wave_start=0, verified=None,
                        message="FPGA 正在连续接收并计算，文字随实际输出更新。")
            dma.recvchannel.transfer(output)
            mmio.write(0x00, 1)
            started = time.monotonic()
            dma.sendchannel.transfer(source)
            active = True
            consumed = 0
            first_partial = None
            maximum_visible_rows = 0
            # 演示低时钟延长事务，超时随输入长度和时钟保守缩放。
            deadline = started + max(30.0, len(inputs) * 0.01 * 50.0 / clock)
            while True:
                if self.cancel.is_set():
                    return False
                now = time.monotonic()
                if now > deadline:
                    raise TimeoutError("板端事务超时，需检查 DMA/Overlay 状态。")
                flags = int(mmio.read(0x34))
                if flags:
                    raise RuntimeError("加速器错误标志：0x%08x" % flags)
                received = min(len(inputs), int(mmio.read(0x1C)))
                status = int(mmio.read(0x04))
                done = bool(status & 2) and dma.sendchannel.idle and dma.recvchannel.idle
                # HP 端口不与 ARM cache 一致；写入开始后绝不能再 flush 输出。
                # 先 invalidate，再只消费已经实际落入 DDR 的完整 32-logit 行。
                output.invalidate()
                prefix = np.asarray(output).copy()
                ready = np.all(prefix != SENTINEL, axis=1)
                missing = np.flatnonzero(~ready)
                available = int(missing[0]) if len(missing) else len(prefix)
                if available > consumed:
                    decoder.feed(prefix[consumed:available])
                    consumed = available
                    if not done:
                        maximum_visible_rows = max(maximum_visible_rows, available)
                        if first_partial is None:
                            first_partial = now - started
                wave_start = max(0, received - 400)
                wave = inputs[wave_start:received, CHANNELS].astype(int).tolist()
                self.update(input_frames=received, output_frames=consumed,
                            text=display_text(decoder.raw), elapsed_seconds=round(now-started, 3),
                            wave_start=wave_start, wave=wave, error_flags=flags)
                if done:
                    dma.sendchannel.wait()
                    dma.recvchannel.wait()
                    active = False
                    break
                self.cancel.wait(0.06)
            actual = np.asarray(output).copy()
            decoder.feed(actual[consumed:])
            elapsed = time.monotonic() - started
            mismatches = int(np.count_nonzero(actual != expected))
            if mismatches:
                raise RuntimeError("本次硬件输出与冻结整数参考有 %d 个差异" % mismatches)
            result = {
                "record": index, "rank": self.rank_by_id[index]["rank"], "session": meta["session"],
                "low_elements":int(mmio.read(0x44)),"total_elements":int(mmio.read(0x48)),
                "text": display_text(decoder.raw), "reference": display_text(meta["reference"]),
                "input_frames": len(inputs), "output_frames": len(actual),
                "elapsed_seconds": elapsed, "clock_mhz": clock,
                "mismatches": mismatches, "error_flags": int(mmio.read(0x34)),
                "first_partial_seconds": first_partial,
                "partial_rows_before_done": maximum_visible_rows,
                "pl_cycle_count": int(mmio.read(0x2C)),
                "max_frame_cycles": int(mmio.read(0x30)),
                "input_byte_count": int(mmio.read(0x14)),
                "output_word_count": int(mmio.read(0x18)),
                "flush_completed_count": int(mmio.read(0x28)),
                "signal_duration_seconds": len(inputs) * 0.02,
            }
            if result["input_byte_count"] != inputs.size or result["output_word_count"] != expected.size:
                raise RuntimeError("板端输入输出计数不匹配")
            target=self.expected_counters[str(index)]["2"]
            if result["low_elements"]!=target["low_elements"] or result["total_elements"]!=target["total_elements"]:
                raise RuntimeError("活动策略计数不匹配")
            self.history.appendleft(result)
            (self.log_dir / ("record_%03d_%d.json" % (index, time.time()))).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            self.update(status="completed", text=result["text"], reference=result["reference"],
                        verified=True, mismatches=0, history=list(self.history),
                        message="本次输出逐位校验通过；下一句将重新初始化循环状态。")
            return True
        finally:
            if active:
                # DMA 还可能写内存，必须先重配置终止硬件，再释放缓冲区。
                Overlay(str(self.root / "h144_dma_overlay.bit"), download=True)
            source.close()
            output.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, body, status=200):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path)
        if path.path == "/api/state":
            return self.send_json(self.server.demo.snapshot())
        if path.path == "/api/records":
            return self.send_json([{"record": i, "session": r["session"],
                                    "seconds": round(r["input_frames"] * 0.02, 1),"rank":self.server.demo.rank_by_id[i]["rank"]}
                                   for i, r in self.server.demo.records.items()])
        if path.path != "/api/events":
            return self.send_json({"error": "not found"}, 404)
        demo = self.server.demo
        after = int(self.headers.get("Last-Event-ID", "0"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            # 新订阅先发当前快照，避免恢复大量陈旧波形。
            with demo.condition:
                after = demo.event_id
                state = dict(demo.state)
            self.wfile.write(("id: %d\nevent: state\ndata: %s\n\n" % (
                after, json.dumps(state, ensure_ascii=False))).encode())
            self.wfile.flush()
            while True:
                with demo.condition:
                    pending = [e for e in demo.events if e[0] > after]
                    if not pending:
                        demo.condition.wait(5)
                        pending = [e for e in demo.events if e[0] > after]
                if not pending:
                    self.wfile.write(b": heartbeat\n\n")
                for event_id, kind, data in pending:
                    self.wfile.write(("id: %d\nevent: %s\ndata: %s\n\n" % (
                        event_id, kind, json.dumps(data, ensure_ascii=False))).encode())
                    after = event_id
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if self.path != "/api/control":
            return self.send_json({"error": "not found"}, 404)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 4096:
                raise ValueError("请求过长")
            body = json.loads(self.rfile.read(size))
            self.server.demo.control(body)
            self.send_json({"ok": True})
        except (ValueError, TypeError, KeyError) as exc:
            self.send_json({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--trace-dir",type=Path,required=True)
    parser.add_argument("--golden",type=Path,required=True)
    parser.add_argument("--ranking",type=Path,required=True)
    parser.add_argument("--counters",type=Path,required=True)
    parser.add_argument("--log-dir",type=Path,default=Path(__file__).resolve().parents[1]/"results/live")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.demo = Demo(args.overlay_dir,args.log_dir,args.trace_dir,args.golden,args.ranking,args.counters)
    print("B2Tnet_BOARD_SERVER_READY", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
