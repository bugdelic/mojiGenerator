import time, threading
import OSC

server_address = ("172.16.65.62", 7000)
server = OSC.OSCServer(server_address)
server.addDefaultHandlers()

def myMsgPrinter_handler(addr, tags, data, client_address):
    print "osc://%server%server ->" % (OSC.getUrlStr(client_address), addr),
    print "(tags, data): (%server, %server)" % (tags, data)

server.addMsgHandler("/fotn/start", myMsgPrinter_handler)
server.addMsgHandler("/fotn/end", myMsgPrinter_handler)
server.addMsgHandler("/fotn/matrix", myMsgPrinter_handler)
server.addMsgHandler("/fotn/jis", myMsgPrinter_handler)
server.addMsgHandler("/fotn/utf", myMsgPrinter_handler)
server.addMsgHandler("/fotn/start", myMsgPrinter_handler)
server_thread = threading.Thread(target=server.serve_forever)
server_thread.start()
