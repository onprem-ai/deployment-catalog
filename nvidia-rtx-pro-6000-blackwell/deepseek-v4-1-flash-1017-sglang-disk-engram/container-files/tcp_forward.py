"""Plain TCP forwarder 0.0.0.0:<listen> -> 127.0.0.1:<target> (debug self-test build only)."""
import asyncio
import sys


async def pipe(r, w):
    try:
        while True:
            b = await r.read(65536)
            if not b:
                break
            w.write(b)
            await w.drain()
    except Exception:
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


async def handle(cr, cw, target):
    try:
        sr, sw = await asyncio.open_connection("127.0.0.1", target)
    except Exception:
        cw.close()
        return
    await asyncio.gather(pipe(cr, sw), pipe(sr, cw))


async def main(listen, target):
    srv = await asyncio.start_server(lambda r, w: handle(r, w, target), "0.0.0.0", listen)
    print("tcp_forward 0.0.0.0:%d -> 127.0.0.1:%d" % (listen, target), flush=True)
    async with srv:
        await srv.serve_forever()


asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
