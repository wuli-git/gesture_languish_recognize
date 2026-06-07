#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
融合训练: 原始数据 + 你自己录制的手势数据 → 训练新模型

用法:
  python train_from_my_data.py
  python train_from_my_data.py --my_dataset my_dataset --multiplier 8 --epochs 150
"""

import argparse
import csv
import os
from pathlib import Path

# 在 import matplotlib 之前设后端
for backend in ('TkAgg', 'Qt5Agg', 'QtAgg'):
    try:
        import matplotlib
        matplotlib.use(backend, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import tensorflow as tf

from data_augmentation import generate_augmented_dataset


CLASSES = ['你好', '再见', '对不起', '没关系', '谢谢',
           '上课', '下课', '不舒服', '厉害', '多少钱']
SEQUENCE_LENGTH = 30
FEATURE_DIM = 126


def get_args():
    parser = argparse.ArgumentParser(description='融合训练')
    parser.add_argument('--original_dataset',
                        default='model/dynamic_gesture_classifier/dynamic_gesture_dataset.csv')
    parser.add_argument('--my_dataset', default='my_dataset')
    parser.add_argument('--output', default='model/dynamic_gesture_classifier')
    parser.add_argument('--multiplier', type=int, default=8,
                        help='每样本增强倍数')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--validation_split', type=float, default=0.2)
    return parser.parse_args()


def load_original_data(path):
    rows = np.loadtxt(path, delimiter=',', dtype=np.float32, encoding='utf-8-sig')
    if rows.ndim == 1:
        rows = np.expand_dims(rows, axis=0)
    y = rows[:, 0].astype(np.int32)
    x = rows[:, 1:].reshape((-1, SEQUENCE_LENGTH, FEATURE_DIM))
    return x, y


def load_my_data(my_dataset_dir):
    """加载我自己录制的 CSV 文件（label_id 以当前 CLASSES 为准，忽略 CSV 第一列旧值）"""
    my_dir = Path(my_dataset_dir)
    x_list, y_list = [], []

    for label_id, class_name in enumerate(CLASSES):
        class_dir = my_dir / class_name
        if not class_dir.exists():
            continue
        for csv_file in class_dir.glob('*.csv'):
            try:
                rows = np.loadtxt(str(csv_file), delimiter=',', dtype=np.float32, encoding='utf-8-sig')
                if rows.ndim == 1:
                    rows = np.expand_dims(rows, axis=0)
                # 忽略 CSV 第一列的旧 label_id，用目录名对应的 index
                x = rows[:, 1:].reshape((-1, SEQUENCE_LENGTH, FEATURE_DIM))
                y = np.full(len(x), label_id, dtype=np.int32)
                x_list.append(x)
                y_list.append(y)
            except Exception as e:
                print(f'  跳过 {csv_file}: {e}')

    if not x_list:
        return None, None

    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)


def build_model(num_classes):
    inputs = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM))

    x = tf.keras.layers.Conv1D(64, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(128, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(128, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(256, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def stratified_split(x, y, validation_split, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label_id in np.unique(y):
        indexes = np.where(y == label_id)[0]
        rng.shuffle(indexes)
        val_count = max(1, int(round(len(indexes) * validation_split)))
        val_idx.extend(indexes[:val_count])
        train_idx.extend(indexes[val_count:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def save_tflite(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f'  TFLite: {path} ({len(tflite_model)/1024:.1f} KB)', flush=True)


# ═══════════════════════════════════════════════════════
# 实时训练曲线回调
# ═══════════════════════════════════════════════════════

class LivePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, labels, x_val, y_val):
        super().__init__()
        self.labels = labels
        self.x_val = x_val
        self.y_val = y_val
        self.hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self._init_fig()

    def _init_fig(self):
        plt.ion()
        self.fig = plt.figure('Training Live Monitor', figsize=(16, 10))
        self.fig.patch.set_facecolor('#1a1a2e')
        C = {'train': '#00d4ff', 'val': '#ff6b6b', 'grid': '#2d2d44',
             'text': '#e0e0e0', 'best': '#ffd700', 'lr': '#51cf66',
             'bar_bg': '#16213e', 'bar_fill': '#0f3460'}
        self.C = C

        gs = self.fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

        # Loss
        ax = self.fig.add_subplot(gs[0, 0]); ax.set_facecolor('#16213e')
        self.l_tl, = ax.plot([], [], color=C['train'], lw=2.2, label='Train Loss')
        self.l_vl, = ax.plot([], [], color=C['val'], lw=2.2, label='Val Loss')
        ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor=C['text'])
        ax.set_title('Loss', color=C['text'], fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', color=C['text']); ax.grid(True, alpha=0.25, color=C['grid'])

        # Accuracy
        ax = self.fig.add_subplot(gs[0, 1]); ax.set_facecolor('#16213e')
        self.l_ta, = ax.plot([], [], color=C['train'], lw=2.2, label='Train Acc')
        self.l_va, = ax.plot([], [], color=C['val'], lw=2.2, label='Val Acc')
        self.best_l = ax.axhline(0, color=C['best'], ls='--', lw=1, alpha=0.6, label='Best')
        ax.legend(loc='lower right', fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor=C['text'])
        ax.set_title('Accuracy', color=C['text'], fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', color=C['text']); ax.grid(True, alpha=0.25, color=C['grid'])

        # LR
        ax = self.fig.add_subplot(gs[0, 2]); ax.set_facecolor('#16213e')
        self.l_lr, = ax.plot([], [], color=C['lr'], lw=2.2, marker='o', ms=3)
        ax.set_title('Learning Rate', color=C['text'], fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', color=C['text']); ax.set_yscale('log')
        ax.grid(True, alpha=0.25, color=C['grid'])

        # Status
        ax = self.fig.add_subplot(gs[1, :2]); ax.set_facecolor('#16213e'); ax.axis('off')
        self.stxt = ax.text(0.05, 0.5, 'Initializing...', transform=ax.transAxes,
                            fontfamily='monospace', fontsize=13, color=C['text'],
                            va='center', bbox=dict(boxstyle='round', facecolor='#0f3460',
                            edgecolor='#333', alpha=0.8, pad=0.8))

        # Per-class bars
        self.ax_bar = self.fig.add_subplot(gs[1, 2]); self.ax_bar.set_facecolor('#16213e')
        self.bars = self.ax_bar.bar(range(len(self.labels)), [0]*len(self.labels),
                                     color=C['bar_fill'], edgecolor='#555', lw=0.8)
        self.ax_bar.set_title('Per-Class Accuracy', color=C['text'], fontsize=11, fontweight='bold')
        self.ax_bar.set_xticks(range(len(self.labels)))
        self.ax_bar.set_xticklabels(self.labels, fontsize=7, color=C['text'], rotation=30)
        self.ax_bar.set_ylim(0, 1.18)
        self.ax_bar.tick_params(axis='y', colors=C['text'])
        self.ax_bar.grid(axis='y', alpha=0.2, color=C['grid'])

        for a in self.fig.get_axes():
            a.tick_params(colors=C['text'])
            for s in a.spines.values(): s.set_color('#333')

        self.fig.tight_layout(pad=2.5)
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update(self, epoch):
        eps = np.arange(1, len(self.hist['loss'])+1)
        self.l_tl.set_data(eps, self.hist['loss'])
        self.l_vl.set_data(eps, self.hist['val_loss'])
        self.l_ta.set_data(eps, self.hist['accuracy'])
        self.l_va.set_data(eps, self.hist['val_accuracy'])
        self.l_lr.set_data(eps, self.hist['lr'])

        cur = self.hist['val_accuracy'][-1]
        if not np.isnan(cur) and cur > self.best_val_acc:
            self.best_val_acc = cur
            self.best_epoch = len(self.hist['val_accuracy'])
        self.best_l.set_ydata([self.best_val_acc, self.best_val_acc])

        for a in [self.l_tl.axes, self.l_ta.axes, self.l_lr.axes]:
            a.relim(); a.autoscale_view()

        self.stxt.set_text(
            f"  Epoch: {epoch:4d} / {str(self.params.get('epochs','?'))}\n"
            f"  Train Loss: {self.hist['loss'][-1]:.4f}    Val Loss: {self.hist['val_loss'][-1]:.4f}\n"
            f"  Train Acc:  {self.hist['accuracy'][-1]:.4f}    Val Acc:  {self.hist['val_accuracy'][-1]:.4f}\n"
            f"  Best Val:   {self.best_val_acc:.4f} (epoch {self.best_epoch})\n"
            f"  LR:         {self.hist['lr'][-1]:.6f}"
        )
        self.fig.suptitle(f'Epoch {epoch} | Train Acc: {self.hist["accuracy"][-1]:.4f} | '
                          f'Val Acc: {self.hist["val_accuracy"][-1]:.4f} | '
                          f'Best: {self.best_val_acc:.4f}',
                          color=self.C['text'], fontsize=14, fontweight='bold', y=0.98)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_end(self, epoch, logs=None):
        self.hist['loss'].append(logs.get('loss', 0))
        self.hist['accuracy'].append(logs.get('accuracy', 0))
        self.hist['val_loss'].append(logs.get('val_loss', 0))
        self.hist['val_accuracy'].append(logs.get('val_accuracy', 0))
        self.hist['lr'].append(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))
        self._update(epoch + 1)
        print(f'  Epoch {epoch+1:3d} | loss={logs["loss"]:.4f} | acc={logs["accuracy"]:.4f} | '
              f'val_loss={logs.get("val_loss",0):.4f} | val_acc={logs.get("val_accuracy",0):.4f} | '
              f'best={self.best_val_acc:.4f}', flush=True)

    def on_train_end(self, logs=None):
        from sklearn.metrics import confusion_matrix
        yp = np.argmax(self.model.predict(self.x_val, batch_size=32, verbose=0), axis=1)
        cm = confusion_matrix(self.y_val, yp)
        per_class = cm.diagonal() / cm.sum(axis=1)
        colors = plt.cm.RdYlGn(per_class)
        for bar, acc, c in zip(self.bars, per_class, colors):
            bar.set_height(acc); bar.set_color(c)
            self.ax_bar.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.02,
                             f'{acc:.1%}', ha='center', va='bottom', fontsize=7,
                             color=self.C['text'], fontweight='bold')

        self.fig.suptitle(f'DONE! Best Val Acc: {self.best_val_acc:.4f} (Epoch {self.best_epoch})',
                          color='#51cf66', fontsize=14, fontweight='bold', y=0.98)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        print(f'\n{"="*60}')
        print(f'  训练完成!')
        print(f'{"="*60}')
        plt.ioff()
        # 保存训练曲线图到文件
        save_path = Path(__file__).parent / 'visualizations' / 'live_training_curves.png'
        save_path.parent.mkdir(exist_ok=True)
        self.fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=self.C.get('bg', '#1a1a2e'))
        print(f'  训练曲线已保存: {save_path}')
        plt.close(self.fig)


def main():
    args = get_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载原始数据 ──
    print('=' * 60)
    print('加载原始数据...')
    x_orig, y_orig = load_original_data(args.original_dataset)
    print(f'  原始: {len(x_orig)} 条')

    # ── 加载你的数据 ──
    print('\n加载你的数据...')
    x_my, y_my = load_my_data(args.my_dataset)
    if x_my is not None:
        print(f'  你的: {len(x_my)} 条')
        for i, name in enumerate(CLASSES):
            cnt = np.sum(y_my == i)
            if cnt > 0:
                print(f'    [{i}] {name}: {cnt} 条')
    else:
        print('  没有找到你的数据! 请先运行 record_gesture.py')
        return

    # ── 合并 ──
    x_all = np.concatenate([x_orig, x_my], axis=0)
    y_all = np.concatenate([y_orig, y_my], axis=0)
    print(f'\n合并后: {len(x_all)} 条 (原始{len(x_orig)} + 你的{len(x_my)})')

    # 每类统计
    for i, name in enumerate(CLASSES):
        print(f'  [{i}] {name}: {np.sum(y_all == i)} 条')

    # ── 划分 ──
    x_train, y_train, x_val, y_val = stratified_split(x_all, y_all, args.validation_split)
    print(f'\n训练集: {len(x_train)} | 验证集: {len(x_val)}')

    # ── 增强 ──
    print(f'\n数据增强 (×{args.multiplier})...')
    x_train_aug, y_train_aug = generate_augmented_dataset(
        x_train, y_train, multiplier=args.multiplier,
    )
    print(f'  增强后训练集: {len(x_train_aug)} 条')

    # ── 训练 ──
    print(f'\n开始训练...')
    model = build_model(len(CLASSES))
    model.summary()

    live_plot = LivePlotCallback(CLASSES, x_val, y_val)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=25,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10,
            min_lr=1e-6, verbose=1,
        ),
        live_plot,
    ]

    history = model.fit(
        x_train_aug, y_train_aug,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        shuffle=True,
        verbose=0,  # LivePlotCallback 已输出每轮信息
    )

    # ── 评估 ──
    from sklearn.metrics import classification_report
    y_val_pred = np.argmax(model.predict(x_val, batch_size=32, verbose=0), axis=1)
    print('\n─── 验证集结果 ───')
    print(classification_report(y_val, y_val_pred, target_names=CLASSES, zero_division=0))

    # 从 LivePlotCallback 获取最佳指标 (比 history.history 更可靠)
    best_val_acc = float(live_plot.best_val_acc)
    best_epoch = int(live_plot.best_epoch)
    try:
        best_train_acc = float(live_plot.hist['accuracy'][best_epoch - 1])
    except (IndexError, KeyError):
        best_train_acc = 0.0
    print(f'最佳 epoch: {best_epoch} | val_acc={best_val_acc:.4f}')

    # ── 保存 ──
    print('\n保存模型...')
    h5_path = output_dir / 'dynamic_gesture_classifier.h5'
    tflite_path = output_dir / 'dynamic_gesture_classifier.tflite'
    model.save(h5_path)
    save_tflite(model, tflite_path)

    # 更新 label
    label_path = output_dir / 'dynamic_gesture_label.csv'
    with open(label_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        for c in CLASSES:
            writer.writerow([c])

    print(f'\n完成! 新模型已保存到:')
    print(f'   {h5_path}')
    print(f'   {tflite_path}')
    print(f'\n现在可以运行 python app.py 测试新模型了')


if __name__ == '__main__':
    main()
