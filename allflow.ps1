for ($i=0; $i -lt 100; $i++) {
    python .\chainer_fx.py test9.ini -nN5Relu -d0_50000_50000 -g0 -mtrain -t10
    python .\chainer_fx.py test9.ini -mtesthr
}
for ($i=0; $i -lt 100; $i++) {
    python .\chainer_fx.py test9.ini -nN10Relu -d0_50000_50000 -g0 -mtrain -t10
    python .\chainer_fx.py test9.ini -mtesthr
}
for ($i=0; $i -lt 100; $i++) {
    python .\chainer_fx.py test9.ini -nN15Relu -d0_50000_50000 -g0 -mtrain -t10
    python .\chainer_fx.py test9.ini -mtesthr
}
