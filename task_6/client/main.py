from threading import Thread
import context
import client
import simulator
from log import get_logger
import animate


def simulate(sim: simulator.Simulator):
    sim.run()


if __name__ == '__main__':
    log = get_logger("client", "/tmp/tail_client.log")
    log.debug("Loading buckets.json")
    context.read_buckets()
    log.debug("Loading contexts.json")
    context.read_contexts()

    kontext = context.banner_contexts.contexts[0]
    kontext.log = log
    log.debug(kontext.to_string())

    client = client.Client(context=kontext, host="127.0.0.1", port=8000, log=log)
    log.debug("Simulator run ...")
    simulator = simulator.Simulator(cln=client, ctx=kontext, log=log)
    thread = Thread(target=simulate, args=(simulator,))
    thread.start()
    log.debug("Run animate ...")
    animate.run_animate(host="127.0.0.1", port=8000, context=kontext, simulator=simulator, log=log)

    simulator.stop()
    thread.join()
