import torch
from diffusers import DiffusionPipeline


class TaskVector:
    def __init__(self, finetuned_checkpoint=None, pretrained_state_path = "res/state_dict_base.pth", vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            with torch.no_grad():
                assert finetuned_checkpoint is not None and pretrained_state_path is not None
                with torch.no_grad():
                    pretrained_state_dict = torch.load(pretrained_state_path)
                    finetuned_state_dict = finetuned_checkpoint.unet.state_dict()
                    self.vector = {}
                    for key in pretrained_state_dict:
                        if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                            continue
                        self.vector[key] = finetuned_state_dict[key].to("cpu") - pretrained_state_dict[key].to("cpu")

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __sub__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] - other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def __truediv__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] / other
        return TaskVector(vector=new_vector)

    def __mul__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] * other
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_model, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            # new_state_dict.to("cuda")
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key].to("cpu") + scaling_coef * self.vector[key].to("cpu")
                new_state_dict[key].to("cuda")

        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

def merge_rnd_mix(task_vectors):
    """Randomly mix multiple task vectors together."""
    if len(task_vectors) == 0:
        return task_vectors[0]

    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            _rand_indices = torch.randint(0, len(task_vectors), task_vectors[0].vector[key].shape)
            new_vector[key] = task_vectors[0].vector[key] * (_rand_indices == 0)
            for i in range(1, len(task_vectors)):
                new_vector[key] += task_vectors[i].vector[key] * (_rand_indices == i)

    return TaskVector(vector=new_vector)


def merge_max_abs(task_vectors):
    """Mix multiple task vectors together by highest parameter value."""
    if len(task_vectors) == 0:
        return task_vectors[0]

    with torch.no_grad():
        new_vector = {}

        # Iterate over keys in the first task vector
        for key in task_vectors[0].vector:
            # Get the initial tensor for the current key
            max_abs_tensor = task_vectors[0].vector[key].to("cuda")

            # Iterate over the remaining task vectors
            for task_vector in task_vectors[1:]:
                current_tensor = task_vector.vector[key].to("cuda")

                # Update max_abs_tensor to keep the element-wise maximum absolute values
                max_abs_tensor = torch.where(current_tensor.abs() >= max_abs_tensor.abs(), current_tensor, max_abs_tensor)

            # Assign the final tensor to the new_vector dictionary
            new_vector[key] = max_abs_tensor

    return TaskVector(vector=new_vector)
