# TEMPORARY: print stdin as one workflow annotation.
import sys

s = sys.stdin.read().replace("%", "%25").replace("\r", "").replace("\n", "%0A")
print("::notice::" + s)
