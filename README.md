# lucid-finetuning

    git clone git@github.com:aGIToz/kFineTuning.git
    ln -s kFineTuning/flowers17 .

    python train.py
    bash freeze-googlenet.sh
    cp googlenetLucid.pb googlenetLucid-finetuned-flower17.pb

    # Manually set do_finetuning = False in train.py
    python train.py
    bash freeze-googlenet.sh
    cp googlenetLucid.pb googlenetLucid-default.pb

    time cat googlenet-node-names | grep Branch_3_b_1x1_act/Relu | python vis.py googlenetLucid-finetuned-flower17.pb - grid-finetuned
    open grid-finetuned.png
    time cat googlenet-node-names | grep Branch_3_b_1x1_act/Relu | python vis.py googlenetLucid-default.pb - grid-default
    open grid-default.png
