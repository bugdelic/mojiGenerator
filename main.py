import dcgan
c = dcgan.BUGWORD()
import time, threading
import OSC

server_address = ("172.16.65.62", 7001) # Windows PC IP  (osc smartphone to Windows)
server_address = ("localhost", 7001)
server = OSC.OSCServer(server_address)
server.addDefaultHandlers()

log = []

send_address = ("172.16.65.62", 7002) #osc python code to Unity
send_address = ("localhost", 7002) #osc python code to Unity
client = OSC.OSCClient()

def send_res(taskId, imgurl):
    msg = OSC.OSCMessage("/hybrid")
    msg.append(taskId)
    msg.append(imgurl)
    client.sendto(msg, send_address)

def unison_handler(addr,tags, data, client_address):
    taskId = data[0]
    unicode1 = data[1]
    # unicode1 = "0x5342" みたいな感じ
    unicode2 = data[2]
    unicode_str1 = unicode1[2:]
    # unicode_str1 は 0xを除く
    unicode_str2 = unicode2[2:]
    
    # utf8 にする
    ut8_1 = eval("u'\u"+unicode_str1+"'").encode("utf8")
    ut8_2 = eval("u'\u"+unicode_str2+"'").encode("utf8")
    
    imgurl = "./out/"+unicode_str1+"_"+unicode_str2+".png"
    c.morphLoopTile(ut8_1,ut8_2).save(imgurl)
    send_res(taskId ,imgurl)



def myMsgPrinter_handler(addr, tags, data, client_address):
    print "osc://%server%server ->" % (OSC.getUrlStr(client_address), addr),
    print "(tags, data): (%server, %server)" % (tags, data)

def pair_handler(addr, tags, data, client_address):
    print "osc://%server%server ->" % (OSC.getUrlStr(client_address), addr),
    print "(tags, data): (%server, %server)" % (tags, data)
    unicode_str1 = data[0][2:]
    ut8_1 = eval("u'\u"+unicode_str1+"'").encode("utf8")
    unicode_str2 = data[1][2:]
    ut8_2 = eval("u'\u"+unicode_str2+"'").encode("utf8")
    c.morphLoopTile(ut8_1,ut8_2).save("./out/"+unicode_str1+"_"+unicode_str2+".png")
    return

def alone_handler(addr, tags, data, client_address):
    print "gan osc://%server%server ->" % (OSC.getUrlStr(client_address), addr),
    print "(tags, data): (%server, %server)" % (tags, data)
    unicode_str1 = data[0][2:]
    print unicode_str1
    ut8_1 = eval("u'\u"+unicode_str1+"'").encode("utf8")
    print ut8_1
    log.append(ut8_1)
    if len(log) > 1:
        img = c.morphLoopTile(log[-1],log[-2])
        img.save("./out/"+unicode_str1+"_.png")
    return

server.addMsgHandler("/unison",unison_handler)

server.addMsgHandler("/fotn/pair", pair_handler)
server.addMsgHandler("/fotn/alone", alone_handler)
server.addMsgHandler("/fotn/start", myMsgPrinter_handler)
server.addMsgHandler("/fotn/end", myMsgPrinter_handler)
server.addMsgHandler("/fotn/matrix", myMsgPrinter_handler)
server.addMsgHandler("/fotn/jis", myMsgPrinter_handler)
#server.addMsgHandler("/fotn/utf", myMsgPrinter_handler)
server.addMsgHandler("/fotn/utf", alone_handler)
server.addMsgHandler("/fotn/start", myMsgPrinter_handler)
server_thread = threading.Thread(target=server.serve_forever)
print "thread start"
server_thread.start()
