import copy
import torch
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)

# from utils.quantizer import (
#     AXQuantizer,
#     get_quantization_config,
# )
from utils.ax_quantizer import(
    load_config,
    AXQuantizer,
)
from utils.train_utils import (
    load_model,
    train_one_epoch,
    imagenet_data_loaders,
    dynamo_export,
    onnx_simplify,
    evaluate,
)
import utils.quantized_decomposed_dequantize_per_channel



def train():
    data_loader, data_loader_test = imagenet_data_loaders("dataset/imagenet/")
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)

    float_model = load_model("./resnet50/resnet50_pretrained_float.pth", "resnet50").to("cuda")
    float_path = "./resnet50/resnet50_float.onnx"
    dynamo_export(float_model, example_inputs, float_path)

    # quantizer
    global_config, regional_configs = load_config("./resnet50/config.json")
    quantizer = AXQuantizer()
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    exported_model = torch.export.export_for_training(float_model, example_inputs).module()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    num_epochs = 1
    num_train_batches = 10
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(prepared_model.parameters(), lr=0.001, momentum=0.9)  # 更小的学习率

    # train
    num_epochs_between_evals = 2
    for nepoch in range(num_epochs):
        train_one_epoch(prepared_model, criterion, optimizer, data_loader, "cuda", num_train_batches)

        checkpoint_path = "./resnet50/checkpoint/checkpoint_%s.pth" % nepoch
        torch.save(prepared_model.state_dict(), checkpoint_path)

        if (nepoch + 1) % num_epochs_between_evals == 0:
            prepared_model_copy = copy.deepcopy(prepared_model)
            quantized_model = convert_pt2e(prepared_model_copy)
            top1, top5 = evaluate(quantized_model, data_loader_test)
            print('Epoch %d: Evaluation accuracy, %2.2f' % (nepoch, top1.avg))

    torch.save(prepared_model.state_dict(), "./resnet50/checkpoint/last_checkpoint.pth")

    # evaluate
    quantized_model = convert_pt2e(prepared_model)
    # top1, top5 = evaluate(quantized_model, data_loader_test)

    # export
    qat_path = "./resnet50/resnet50_qat.onnx"
    dynamo_export(quantized_model, (example_inputs,), qat_path)

    # onnx simplify
    sim_path = "./resnet50/resnet50_qat_sim.onnx"
    onnx_simplify(qat_path, sim_path)


if __name__ == "__main__":
    train()
