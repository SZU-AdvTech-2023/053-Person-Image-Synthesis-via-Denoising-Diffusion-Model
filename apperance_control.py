from predict import Predictor

obj = Predictor()
src = 'data/fashion e-commerce images/b4d93d23-5c4b-4a96-920a-69d18b6ad2db1633784889018TokyoTalkiesBlackFloralCrepeDress1.jpg'
ref_img = 'data/deepfashion_256x256/target_edits/reference_img_0.png'
ref_mask = 'data/deepfashion_256x256/target_mask/upper/reference_mask_3.png'
ref_pose = 'data/deepfashion_256x256/target_pose/reference_pose_2.npy'
obj.predict_appearance(image=src, ref_img = ref_img, ref_mask = ref_mask,
ref_pose = ref_pose, sample_algorithm = 'ddim', nsteps = 50)