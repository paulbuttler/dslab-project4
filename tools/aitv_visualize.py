import numpy as np
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == '__main__':

    smpl_layer = SMPLLayer(model_type="smplh", gender="female")
    poses = np.zeros([1, smpl_layer.bm.NUM_BODY_JOINTS * 3])
    smpl_seq = SMPLSequence(poses, smpl_layer)

    v = Viewer()
    v.scene.add(smpl_seq)
    v.run()