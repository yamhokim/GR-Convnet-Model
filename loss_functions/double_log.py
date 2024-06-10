import torch

class DoubleLogLoss:

    def __init__(self, mean_reduction: bool = True):
        self.mean_reduction = mean_reduction

    def check_range(self, y, yhat):
        assert ((y < -1).sum() + (y > 1).sum()).item() == 0, "Target outside valid range"
        assert ((yhat < -1).sum() + (yhat > 1).sum()).item() == 0, "Predicted value outside valid range"

    def fp_error_recentre(self, yhat, tolerance = 1e-5):
        max_filter_map = torch.zeros_like(yhat)
        min_filter_map = torch.zeros_like(yhat)
        max_filter_map[yhat > 1] += tolerance
        min_filter_map[yhat < -1] += tolerance
        yhat = yhat - max_filter_map
        yhat = yhat + min_filter_map
        return yhat

    def __call__(self, yhat, y):
        """
        yhat should be a tensor of predictions of shape [batch_size, num_classes], with each element being
        between -1 and 1 (tanh activation after final layer).

        y should be a tensor of target labels of shape [batch_size, num_classes].
        IMPORTANT_NOTE : each element of y will be a tensor of shape [num_classes]. Each element of this
                        SHOULD be either -1 or 1 when training for cls, with -1 (not 0) indicating that the object
                        is not a particular class and 1 indicating that it is.
        """
        self.check_range(y=y, yhat=yhat)
        y_less_than_loss = - torch.log(1 + (1 / (1 + y + 1e-5))*(yhat - y))
        yhat_less_than_loss = - torch.log(1 + (1 / (1 - y + 1e-5))*(y - yhat))
        output = torch.where(yhat < y, y_less_than_loss, yhat_less_than_loss)
        if self.mean_reduction:
            output = output.mean()
        return output


if __name__ == "__main__":
    crit = DoubleLogLoss()
    for i in range(50):
        random_target = (torch.rand(224, 224) * 2) - 1 # take tgt ~ Unif(-1,1) 
        random_yhat = (torch.rand(224, 224) * 2) - 1 # take yhat ~ Unif(-1,1)
        loss = crit(random_target, random_yhat)
        assert not loss.isnan().item()

    # tests target at edge case values (0, 1, -1)
    for i in range(50):
        target = torch.ones(100)
        target[:50] *= -1
        target[-10:] *= 0
        random_yhat = (torch.rand(100) * 2) - 1
        loss = crit(random_yhat, target)
        assert not loss.isnan().item()

    for i in range(10):
        random_target = (torch.rand(224, 224) * 100) - 1 # take tgt from invalid range
        random_yhat = (torch.rand(224, 224) * 100) - 1 # take yhat from invalid range
        # Makes sure that assertion error is throw for invalid range values.
        try:
            loss = crit(random_target, random_yhat)
            raise Exception("Assertion error not thrown")
        except AssertionError:
            pass
