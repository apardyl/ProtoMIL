import dataclasses
from dataclasses import dataclass
from typing import Tuple, List


@dataclass(frozen=True)
class Settings:
    base_architecture: str
    img_size: int
    prototype_number: int
    prototype_latent: int
    prototype_conv_dim: Tuple[int, int]

    joint_optimizer_lrs: dict
    joint_lr_step_size: int
    warm_optimizer_lrs: dict
    last_layer_optimizer_lr: dict

    num_train_epochs: int
    num_warm_epochs: int
    num_last_layer_iterations: int
    push_start: int
    push_epochs: List[int]

    num_classes: int = 2
    coef_crs_ent: float = 1
    coef_clst: float = 0.8   #0.45 #1 #0.8   # 37.13 # 74.25  # 0.8
    coef_sep: float = -0.08   #-0.05 #-1 #-0.08   #-11.37 # -22.73  # -0.08
    coef_l1: float = 1e-4

    prototype_activation_function: str = 'log'
    add_on_layers_type: str = 'regular'
    loss_function: str = 'cross_entropy'
    class_specific: bool = True
    batch_norm_features: bool = False
    mil_pooling: str = 'gated_attention'

    # currently used only by mnist dataloader.
    bag_length_mean: int = 200
    bag_length_std: int = 150
    bag_length_min: int = 50
    bag_length_max: int = 600
    num_bags_train: int = 1000
    num_bags_test: int = 500
    positive_samples_in_bag_ratio_mean: float = 0.3
    positive_samples_in_bag_ratio_std: float = 0.25
    folds: int = 10
    fold_id: int = 0

    random_seed_presets: List[int] = dataclasses.field(
        default_factory=lambda: [631056511, 923928841, 53306087, 937272127, 207121037])
    random_seed_id: int = 0
    # overrides random seed preset and id.
    random_seed_value = None

    @property
    def coefs(self):
        return {
            'crs_ent': self.coef_crs_ent,
            'clst': self.coef_clst,
            'sep': self.coef_sep,
            'l1': self.coef_l1,
        }

    @property
    def prototype_shape(self):
        return self.prototype_number, self.prototype_latent, self.prototype_conv_dim[0], self.prototype_conv_dim[1]

    @property
    def dataset_settings(self):
        return {
            'bag_length_mean': self.bag_length_mean,
            'bag_length_std': self.bag_length_std,
            'bag_length_min': self.bag_length_min,
            'bag_length_max': self.bag_length_max,
            'positive_samples_in_bag_ratio_mean': self.positive_samples_in_bag_ratio_mean,
            'positive_samples_in_bag_ratio_std': self.positive_samples_in_bag_ratio_std,
            'num_bags_train': self.num_bags_train,
            'num_bags_test': self.num_bags_test,
            'folds': self.folds,
            'fold_id': self.fold_id,
        }

    @classmethod
    def as_params(cls):
        return [(field.name, field.type) for field in dataclasses.fields(cls) if field.type in {str, int, float, bool}]

    def new_from_params(self, params):
        fields = dict()
        for field in dataclasses.fields(self):
            field_name = field.name
            if hasattr(params, field_name) and getattr(params, field_name) is not None:
                fields[field_name] = getattr(params, field_name)
            else:
                fields[field_name] = getattr(self, field_name)
        return Settings(**fields)

    def __str__(self):
        return 'Settings:\n' + '\n'.join(
            '\t{name}: {value}'.format(name=field.name, value=getattr(self, field.name)) for field in
            dataclasses.fields(self))


MNIST_SETTINGS = Settings(
    base_architecture='resnet18_small',
    img_size=28,
    prototype_number=10,
    prototype_latent=64,
    prototype_conv_dim=(2, 2),
    joint_optimizer_lrs={
        'features': 1e-4,
        'add_on_layers': 3e-3,
        'prototype_vectors': 3e-3,
    },
    joint_lr_step_size=5,
    warm_optimizer_lrs={
        'features': 1e-3,
        'add_on_layers': 1e-3,
        'prototype_vectors': 1e-3,
        'attention': 1e-3,
        'last_layer': 1e-3,
    },
    last_layer_optimizer_lr={
        'attention': 1e-3,
        'last_layer': 1e-4,
    },
    num_train_epochs=101,
    num_warm_epochs=30,
    num_last_layer_iterations=20,
    push_start=30,
    push_epochs=[i for i in range(200) if i % 10 == 0]
)

COLON_CANCER_SETTINGS = Settings(
    base_architecture='resnet18_small',
    img_size=27,
    prototype_number=20,
    prototype_latent=128,
    prototype_conv_dim=(2, 2),
    loss_function='focal',
    joint_optimizer_lrs={
        'features': 1e-4,
        'add_on_layers': 3e-3,
        'prototype_vectors': 3e-3,
    },
    joint_lr_step_size=5,
    warm_optimizer_lrs={
        'add_on_layers': 3e-3,
        'prototype_vectors': 3e-3,
    },
    last_layer_optimizer_lr={
        'attention': 1e-3,
        'last_layer': 1e-4,
    },
    num_train_epochs=51,
    num_warm_epochs=5,
    num_last_layer_iterations=20,
    push_start=10,
    push_epochs=[i for i in range(200) if i % 10 == 0]
)

PASCAL_SETTINGS = Settings(
    base_architecture='resnet18',
    img_size=224,
    prototype_number=20*5,  # 20 classes, 5 prototypes per class
    prototype_latent=128,
    prototype_conv_dim=(1, 1),
    num_classes=20,
    loss_function='binary_cross_entropy',
    joint_optimizer_lrs={
        'features': 1e-4,
        'add_on_layers': 3e-3,
        'prototype_vectors': 3e-3,
    },
    joint_lr_step_size=5,
    warm_optimizer_lrs={
        # 'features': 1e-3,
        'add_on_layers': 3e-3,
        'prototype_vectors': 3e-3,
        # 'attention': 1e-3,
        # 'last_layer': 1e-3,
    },
    last_layer_optimizer_lr={
        'attention': 1e-3,
        'last_layer': 1e-4,
    },
    num_train_epochs=1000,
    num_warm_epochs=20, # 5,
    num_last_layer_iterations=20,
    push_start=40,
    push_epochs=[i for i in range(1000) if i % 20 == 0]
)