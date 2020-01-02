from .models import smallnet
import torchvision.models as models



def build_model(params, num_outputs=None):
    """
    Build a ResNet model.
    Optionally pretrain it.
    """
    architectures = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    architectures.extend(["smallnet"])

    arch = params.architecture
    assert arch in architectures

    if arch == "smallnet":
        model = smallnet(params.num_classes, params.num_channels, params.num_fc, params.maxpool_size, params.kernel_size, params.non_linearity)
    else:
        model = models.__dict__[arch]()

    # if params.instance_model:
    #     params.num_classes = DATASETS[params.dataset]['num_classes'] if num_outputs is None else num_outputs
    # else:
    #     params.num_outputs = DATASETS[params.dataset]['num_classes'] if num_outputs is None else num_outputs
    #
    # # if arch == 'resnet10':
    # #     model = resnet10(params)
    # if arch == 'resnet18':
    #     model = resnet18(params)
    # elif arch == 'resnet34':
    #     model = resnet34(params)
    # elif arch == 'resnet50':
    #     model = resnet50(params)
    # elif arch == 'resnet101':
    #     model = resnet101(params)
    # elif arch == 'resnet152':
    #     model = resnet152(params)
    #
    # logger.info("Model: {}".format(model))
    # countParams = lambda x: sum([p.numel() for p in x.parameters() if p.requires_grad])
    # prettyNumber = lambda x: "%.2fM" % (x / 1e6) if x >= 1e6 else "%.2fK" % (x / 1e3)
    # logger.info("Number of parameters: %s" % prettyNumber(countParams(model)))

    return model
