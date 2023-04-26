from ptflops import get_model_complexity_info
import model
def count_flops(model):
    macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

model = model.dce_isp3()
count_flops(model)