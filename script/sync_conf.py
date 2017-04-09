#!/usr/bin/env python

import os
from os.path import dirname, join

#host file
hostfile_name = "hostfile"

app_dir = dirname(dirname(os.path.realpath(__file__)))
proj_dir = dirname(dirname(app_dir))
hostfile = join(app_dir, hostfile_name)

fp = open(hostfile)

for ip in fp.readlines():
	ip = ip.strip()
	if ip != "node63":
		cmd_cp = "scp "+join(app_dir,"conf/")+"admm.conf " + " ubuntu@%s:"%(ip)+join(app_dir,"conf/")
		cmd_cp_admm = "scp "+join(app_dir,"bin/")+"admm " + " ubuntu@%s:"%(ip)+join(app_dir,"bin/")
		print cmd_cp
		print cmd_cp_admm
		os.system(cmd_cp_admm)
		os.system(cmd_cp)
fp.close()
