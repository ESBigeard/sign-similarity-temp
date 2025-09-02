import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from feature_extraction.pose_estimation.zelinka.from_2D_to_3D_pose import get_3D_pose, structureStats

def backpropagationBasedFiltering(
        lines0_values, # initial (logarithm of) bones lenghts
        rootsx0_values, # head position
        rootsy0_values,
        rootsz0_values,
        anglesx0_values, # angles of limbs
        anglesy0_values,
        anglesz0_values,
        tarx_values, # target
        tary_values,
        w_values, # weights of estimated points (likelihood of estimation)
        structure,
        dtype,
        learningRate=0.1,
        nCycles=1000,
        regulatorRates=[0.001, 0.1],
        logs: bool = True,
        loading_symbols: list = ["-", "\\", "|", "/"]
):

    T = rootsx0_values.shape[0]
    nBones, nPoints = structureStats(structure)
    nLimbs = len(structure)

    # vector of (logarithm of) bones length
    #   shape: (nLines,)
    lines = tf.Variable(lines0_values, dtype=dtype) 
    # print(lines.shape)
    # x cooordinates of head
    #   shape: (T, 1)
    rootsx = tf.Variable(rootsx0_values, dtype=dtype)
    # y cooordinates of head
    rootsy = tf.Variable(rootsy0_values, dtype=dtype) 
    # z cooordinates of head
    rootsz = tf.Variable(rootsz0_values, dtype=dtype)
    # x coordinate of angles 
    #   shape: (T, nLimbs)
    anglesx = tf.Variable(anglesx0_values, dtype=dtype)
    # y coordinate of angles 
    anglesy = tf.Variable(anglesy0_values, dtype=dtype)
    # z coordinate of angles 
    anglesz = tf.Variable(anglesz0_values, dtype=dtype)   

    # target
    #   shape: (T, nPoints)
    tarx = tf.placeholder(dtype=dtype)
    tary = tf.placeholder(dtype=dtype)
    # likelihood from previous pose estimator
    #   shape: (T, nPoints)
    w = tf.compat.v1.placeholder(dtype=dtype)
    
    # resultant coordinates. It's a list for now. It will be concatenated into a matrix later
    #   shape: (T, nPoints)
    x = [None for i in range(nPoints)]
    y = [None for i in range(nPoints)]
    z = [None for i in range(nPoints)]
    
    # head first
    x[0] = rootsx
    y[0] = rootsy
    z[0] = rootsz
    
    # now other limbs
    i = 0
    # for numerical stability of angles normalization
    epsilon = 1e-10
    # print(f"l = {lines.shape}")
    # print(f"Lines = {lines}")
    for a, b, l in structure:
        # print(f"i = {i}")
        # print(f"a = {a}")
        # print(f"b = {b}")
        # print(f"l = {l}")
        # limb length
        L = tf.exp(lines[l])
        # angle
        Ax = anglesx[0:T, i:(i + 1)]
        Ay = anglesy[0:T, i:(i + 1)]
        Az = anglesz[0:T, i:(i + 1)]
        # angle norm
        normA = tf.sqrt(tf.square(Ax) + tf.square(Ay) + tf.square(Az)) + epsilon
        # new joint position
        x[b] = x[a] + L * Ax / normA
        y[b] = y[a] + L * Ay / normA
        z[b] = z[a] + L * Az / normA
        i = i + 1
    
    # making a matrix from the list
    x = tf.concat(x, axis=1)
    y = tf.concat(y, axis=1)
    z = tf.concat(z, axis=1)

    # weighted MSE
    loss = tf.reduce_sum(w * tf.square(x - tarx) + w * tf.square(y - tary)) / (T * nPoints)
    
    # regularisation
    # reg1 is a sum of bones length
    reg1 = tf.reduce_sum(tf.exp(lines))
    # reg2 is a square of trajectory length
    dx = x[0:(T - 1), 0:nPoints] - x[1:T, 0:nPoints]
    dy = y[0:(T - 1), 0:nPoints] - y[1:T, 0:nPoints]
    dz = z[0:(T - 1), 0:nPoints] - z[1:T, 0:nPoints]
    reg2 = tf.reduce_sum(tf.square(dx) + tf.square(dy) + tf.square(dz)) / ((T - 1) * nPoints)
    
    optimizeThis = loss + regulatorRates[0] * reg1 + regulatorRates[1] * reg2

    # the backpropagation
    optimizer = tf.train.GradientDescentOptimizer(learningRate)
    train = optimizer.minimize(optimizeThis)
    init = tf.variables_initializer(tf.global_variables())
    sess = tf.Session()
    sess.run(init)
    for iCycle in range(nCycles):
        sess.run(train, {tarx: tarx_values, tary: tary_values, w: w_values})
        cycle_info = "iCycle = %3d, loss = %e" % (
        iCycle, sess.run([loss], {tarx: tarx_values, tary: tary_values, w: w_values})[0])

        if logs:
            symbol = loading_symbols[iCycle % len(loading_symbols)]
            sys.stdout.write(f'\r{symbol} | {cycle_info}')
            sys.stdout.flush()

    # returning final coordinates
    return sess.run([x, y, z], {})
    
def get_corrected_3D_pose(Xx, Xy, Xw):

    lines0, rootsx0, rootsy0, rootsz0, anglesx0, anglesy0, anglesz0, Yx0, Yy0, Yz0, structure = get_3D_pose(Xx, Xy, Xw)
    dtype = "float32"

    x_values, y_values, z_values = backpropagationBasedFiltering(
        lines0, rootsx0, rootsy0, rootsz0, anglesx0, anglesy0, anglesz0, Yx0, Yy0, Yz0, structure, dtype, 
        learningRate=0.1, nCycles=1000, regulatorRates=[0.001, 0.1], logs=True, loading_symbols=["-", "\\", "|", "/"]
    )

    Xx = x_values
    Xy = y_values
    Xz = z_values


    return Xx, Xy, Xz
