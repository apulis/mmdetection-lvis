from mmcv.runner import get_dist_info


class textLogger:

    def log(self, text): 
        rank, _ = get_dist_info()
        if rank == 0: 
            print(text)