```python
class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes
```
you can actually call le(self) in __init__


```python
from typing import List, Iterable, Callable, Tuple

def create_nshot_task_label(k: int, q: int) -> torch.Tensor:

```
you can actually specify types in functions in Python3


```python
n_shot_taskloader = DataLoader(self.dataset,
                               batch_sampler=NShotTaskSampler(self.dataset, 100, n, k, q))


class DummyDataset(Dataset):
    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)

```
you can actually specify a class that samples your dataset, and returns a batch (the indices of samples belonging to your dataset)
then this actually makes calls to the dataset.__getitem__() with the specific ids of the current batch

```python
progress_bar = tqdm(total=subset_len)
for ...
    progress_bar.update(1)
progress_bar.close()
```


```python
callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/proto_nets/{param_str}.pth',
        monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
]
```
useful callback classes


```python
self.datasetid_to_filepath = self.df.to_dict()['filepath']
self.datasetid_to_class_id = self.df.to_dict()['class_id']
```
useful pandas function (makes a dictionary with idx pointing to field)

```python
fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_episode,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'distance': args.distance},
)
```
in pytorch...check all argument options

```python
class Callback:
    def on_epoch_end(self, epoch, logs=None):
```
logs is a dictionary and whatever is in will be printed :)
