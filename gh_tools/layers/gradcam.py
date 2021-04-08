import tensorflow as tf

class GradCamLayer2D(tf.keras.layers.Layer):
    '''Takes a model as input and computes the gradient with respect to output'''
    def __init__(self, model, feature_layer_name=None, pred_index=None, name='gradcam_model'):
        assert feature_layer_name is not None, "feature_layer_name is required use model.layers or model.summary to visualize'"
        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs, 
            outputs=[model.get_layer(feature_layer_name).output, model.output])
        self.pred_index = pred_index
        self.feature_layer_name = feature_layer_name
        super().__init__()

    def call(self, inputs):
        with tf.GradientTape() as tape:
            feature_output, preds = self.grad_model(inputs)
            if self.pred_index is None: pred_index = tf.argmax(preds[0])
            else: pred_index = self.pred_index
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, feature_output)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        heatmap = tf.map_fn(lambda ele: ele[0] @ ele[1][..., None], 
                            (feature_output, pooled_grads), 
                            fn_output_signature=tf.float32)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap, tf.range(tf.rank(heatmap))[1:])
        return preds, heatmap

    def get_config(self):
        return {
            "feature_layer_name": self.feature_layer_name,
            "pred_index": self.pred_index,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
