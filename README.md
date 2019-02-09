# lucid-finetuning

    git clone git@github.com:aGIToz/kFineTuning.git
    ln -s kFineTuning/flowers17 .

    python train.py
    cp googlenetLucid.pb googlenetLucid-finetuned-flower17.pb

    # Manually set do_finetuning = False in train.py
    python train.py
    cp googlenetLucid.pb googlenetLucid-default.pb

    time cat googlenet-node-names | grep Branch_3_b_1x1_act/Relu | python vis.py googlenetLucid-finetuned-flower17.pb - grid-finetuned
    open grid-finetuned.png
    # ftp://mokk.bme.hu/User/daniel/tmp/grid-finetuned.png
    time cat googlenet-node-names | grep Branch_3_b_1x1_act/Relu | python vis.py googlenetLucid-default.pb - grid-default
    open grid-default.png
    # ftp://mokk.bme.hu/User/daniel/tmp/grid-default.png

    # Manually set weights=None in train.py line InceptionV1(include_top=False, weights=...
    # do_finetuning = True again at this point.
    python train.py
    cp googlenetLucid.pb googlenetLucid-fromscratch-flower17.pb
    time cat googlenet-node-names | grep Branch_3_b_1x1_act/Relu | python vis.py googlenetLucid-fromscratch-flower17.pb - grid-fromscratch
    open grid-fromscratch.png
    # ftp://mokk.bme.hu/User/daniel/tmp/grid-fromscratch.png
