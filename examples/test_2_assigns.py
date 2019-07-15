import tensorflow as tf
import ngraph_bridge

'''
a var with 2 assigns, one with validate_shape, one without
'''


v = tf.Variable(0)
#new_v_0 = v.assign(10)
#new_v_1 = v.assign([10,20], validate_shape=False)
new_v_0 = tf.assign(v, 10)
new_v_1 = tf.assign(v, [10,20], validate_shape=False)
config = tf.ConfigProto()
sess = tf.Session(config = ngraph_bridge.update_config(config))

sess.run(v.initializer)


sess.run(new_v_1)
print(new_v_1.eval(session=sess))
