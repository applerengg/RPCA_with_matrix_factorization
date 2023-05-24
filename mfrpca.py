import numpy as np
import sklearn as sk
import sklearn.decomposition as decompositon
import cv2
from scipy.linalg import svdvals
from enum import Enum, auto
import os 


class MODE(Enum):
    """
    Used for determining the hyperparameter values according to the task 
    """
    BACKGROUND_EXTRACTION = auto()
    IMAGE_DENOISING = auto()

arr = np.ndarray # used for type hinting

def init_params(mode: MODE, shape):
    """
    Initialize: λ; β; ε; ρ0; S0; V0; Π0; k.
    """
    m, n = shape

    gamma = 0.05 if mode == MODE.BACKGROUND_EXTRACTION else 0.05
    lam = 20 if mode == MODE.BACKGROUND_EXTRACTION else 1
    beta = 1.618 if mode == MODE.BACKGROUND_EXTRACTION else 1.5
    eps = 0.1  # TODO: might need tuning
    ro_k = 0.01 if mode == MODE.BACKGROUND_EXTRACTION else 0.005
    ro_max = 0.1  # TODO: might need tuning
    r = 8  # TODO: might need tuning
    S_k = np.random.randn(m, n)
    V_k = np.random.randn(n, r)
    PI_k = np.random.randn(m, n) 
    # max_iter = 2_000
    max_iter = 10

    return gamma, lam, beta, eps, ro_k, ro_max, r, S_k, V_k, PI_k, max_iter


def MFRPCA(D: arr, mode: MODE ) -> tuple[np.ndarray, np.ndarray]:
    """
    1:while not converged do
        2: Compute T = D + Πk=ρk;
        3: Update Uk+1 by (12);
        4: Update Vk+1 by (16);
        5: Compute Wk = T - Uk+1 Vk+1T;
        6: Update Sk+1 by (17);
        7: Update Πk+1 by (9);
        8: Update ρk+1 by (10);
        9: Check the convergence condition
            10: ||D - Uk+1 Vk+1T - Sk+1||F ≤ ε * ||D||F .
    11: end while
    """
    m, n = D.shape
    gamma, lam, beta, eps, ro_k, ro_max, r, S_k, V_k, PI_k, max_iter = init_params(mode, D.shape)

    U_k = np.zeros((m, r))
    k = 0
    while k < max_iter:
        print(f"{k = }")
        k += 1

        # step 2:
        T: np.ndarray = D + PI_k / ro_k
        # step 3: U_k (12)
        U_k = update_U(T, S_k, V_k) 
        # step 4: V_k (16)
        V_k = update_V(T, S_k, U_k, lam, V_k, gamma, ro_k) 
        # intermediate step: U_k V_k^T
        U_dot_VT = np.dot(U_k, V_k.T)
        # step 5: W_k
        W_k: np.ndarray = T - U_dot_VT
        # step 6: S_k (17)
        S_k = update_S(W_k, ro_k)
        # step 7: PI_k (9)
        PI_k = update_PI(PI_k, ro_k, D, U_dot_VT, S_k)
        # step 8: ro_k (10)
        ro_k = update_ro(beta, ro_k, ro_max)
        
        # step 9-10: convergence check
        if np.linalg.norm(D - U_dot_VT - S_k, "fro") <= eps * np.linalg.norm(D, "fro"):
            # converged
            break

    # Low rank matrix L
    L = np.dot(U_k, V_k.T)
    return L, S_k

#######################################################################################################
#######################################################################################################
#######################################################################################################

def update_U(T, S, V) -> np.ndarray:
    """
    Update U by (12)
    """
    A, _, BT = np.linalg.svd(np.dot(T - S, V), compute_uv=True, full_matrices=False) # u, s, vh
    U = np.dot(A, BT)
    return U

def update_V(T: arr, S: arr, U: arr, lam, V_prev: arr, gamma, ro) -> np.ndarray:
    """
    Update V by (16)
    """
    delta_V_gamma_norm = get_delta_gamma_norm(V_prev, gamma)
    V = np.dot((T - S).T, U) - lam * delta_V_gamma_norm / ro
    return V

def get_delta_gamma_norm(V: np.ndarray, gamma):
    """
    Computed according to Lemma 1 (14)
    """
    A_V, SIGMA, B_VT = np.linalg.svd(V, compute_uv=True, full_matrices=False)
    singular_values = svdvals(V)    
    diag_l = np.power(np.e, (singular_values * -1) / gamma) / gamma
    diag_l = np.diag(diag_l)
    
    delta_V_gamma_norm = np.linalg.multi_dot([A_V, diag_l, B_VT]) # (14)
    return delta_V_gamma_norm

def update_S(W: np.ndarray, ro) -> np.ndarray:
    """
    Update S by (17)
    """
    W_ro_max = np.maximum(np.abs(W) - 1/ro, 0)
    W_sign = np.sign(W)
    S = np.multiply(W_ro_max, W_sign)
    return S

def update_PI(PI_prev: arr, ro, D: arr, U_dot_VT: arr, S: arr) -> np.ndarray:
    """
    Update PI by (9)
    """
    PI = PI_prev + ro * (D - U_dot_VT - S)
    return PI

def update_ro(beta, ro_prev, ro_max):
    """
    Update rho by (10)
    """
    return min(beta * ro_prev, ro_max)

#######################################################################################################
#######################################################################################################
#######################################################################################################

def video_frame_by_frame_background():
    """
    Test function: Applies MFRPCA algorithm to a video (frame by frame) for background extraction 
    """
    mode = MODE.BACKGROUND_EXTRACTION
    vid = cv2.VideoCapture("videos/112.mp4")
    f_ctr = 0
    while vid.isOpened():
        _, frame = vid.read()
        frame = cv2.resize(frame, (240, 180))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        L, S = MFRPCA(frame, mode)
        normalized = normalize_matrix(L)

        print(f_ctr)
        f_ctr += 1

        cv2.imshow("frame", frame)
        cv2.imshow("normalized", normalized)
        cv2.waitKey(1)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()


def image_background():
    """
    Test function: Applies MFRPCA algorithm to an image for background extraction 
    """

    mode = MODE.BACKGROUND_EXTRACTION
    # D = cv2.imread("./images/car.png", cv2.IMREAD_GRAYSCALE)
    # D = cv2.imread("./images/man_beach.png", cv2.IMREAD_GRAYSCALE)
    D = cv2.imread("./images/before.jpg", cv2.IMREAD_GRAYSCALE)
    print("D:\n", D, D.shape)
    
    L, S = MFRPCA(D, mode)
    print(f"{L = }")
    print(f"{np.linalg.matrix_rank(L) = }")

    normalized = normalize_matrix(L)
    # Lmin = np.min(L)
    # Lmax = np.max(L)
    # normalized = (L-Lmin) / (Lmax - Lmin) 
    print(normalized)
    
    cv2.imshow("D", D)
    cv2.imshow("normalized", normalized)
    D_minus_normalized = D - 255*normalized
    D_minus_normalized = normalize_matrix(D_minus_normalized)
    cv2.imshow("D_minus_normalized", D_minus_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_denoise():
    """
    Test function: Applies MFRPCA algorithm to an image for denoising
    """
    mode = MODE.IMAGE_DENOISING
    D = cv2.imread("./images/noisy_image.png", cv2.IMREAD_GRAYSCALE)
    print("D:\n", D, D.shape)
    
    L, S = MFRPCA(D, mode)
    print(f"{L = }")
    print(f"{np.linalg.matrix_rank(L) = }")

    normalized = normalize_matrix(L)    
    print(normalized)
    
    cv2.imshow("D", D)
    cv2.imshow("normalized", normalized)
    D_minus_normalized = D - 255*normalized
    D_minus_normalized = normalize_matrix(D_minus_normalized)
    cv2.imshow("D_minus_normalized", D_minus_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_all_at_once_background():
    """
    Test function: Extracts background video from a video by creating a single matrix composing of the frames
    """

    import moviepy.editor as mpe
    import sklearn.decomposition
    import matplotlib.pyplot as plt
    
    def create_data_matrix_from_video(clip, FPS, dims):
        all_frames = list()
        for i in range(int(FPS * clip.duration)):
            print(f"frame {i: 3}")
            frame = clip.get_frame(i/float(FPS))
            frame = rgb2gray(frame)
            all_frames.append(cv2.resize(frame, dims).flatten())
        return np.vstack(all_frames).T
        # return np.vstack([cv2.resize(rgb2gray(clip.get_frame(i/float(k))), dims)
        #                 .flatten() for i in range(k * int(clip.duration))]).T

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    vidname = "Video_003.avi"
    # vidname = "122.MP4"
    video = mpe.VideoFileClip(f"videos/{vidname}")
    FPS = video.fps
    w, h = video.size
    scale = 100/100   # Adjust scale to change resolution of image
    dims = (int(h * scale), int(w * scale))

    vid_matrix_filename = f"vid_{vidname}_{scale:.3f}scaled_matrix.npy"
    if os.path.isfile(vid_matrix_filename):
        print("load M"); 
        M = np.load(vid_matrix_filename)
    else:
        M = create_data_matrix_from_video(video, FPS, (dims[1], dims[0]))
        np.save(vid_matrix_filename, M) # save in order to be able to load from file later, instead of creating from scratch

    low_rank_matrix_filename = vid_matrix_filename.replace(".npy", "_lowrank.npy")
    if os.path.isfile(low_rank_matrix_filename):
        print("load low_rank"); 
        low_rank = np.load(low_rank_matrix_filename)
    else:
        low_rank, S = MFRPCA(M, mode=MODE.BACKGROUND_EXTRACTION)
        np.save(low_rank_matrix_filename, low_rank) # save in order to be able to load from file later, instead of creating from scratch

    normalized = normalize_matrix(low_rank)
    rows, cols = normalized.shape
    out_bg = cv2.VideoWriter(f'vid_{vidname}_background_extracted.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (dims[1], dims[0]), False)
    out_fg = cv2.VideoWriter(f'vid_{vidname}_foreground_extracted.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (dims[1], dims[0]), False)
    for i in range(cols):
        print(f"frame {i}")
        M_frame = np.reshape(M[:, i], dims)
        L_frame = np.reshape(normalized[:, i], dims)
        L_frame = np.uint8(L_frame * 255) # needs to be 0-255 int for writing video
        
        frame = normalize_matrix(M_frame - L_frame)
        frame = np.uint8(frame * 255) # needs to be 0-255 int for writing video

        out_bg.write(L_frame)
        out_fg.write(frame)

        # Post-processing steps: obtain clear foreground and background images
        # by applying opening and closing. 
        kernel = np.ones((3,3), np.uint8)
        
        L_frame2 = L_frame.copy()
        # L_frame2 = cv2.erode(L_frame2, kernel, iterations=2)
        # L_frame2 = cv2.dilate(L_frame2, kernel, iterations=2)
        
        frame2 = frame.copy()
        frame2 = cv2.erode(frame2, kernel, iterations=1)
        frame2 = cv2.dilate(frame2, kernel, iterations=1)
        frame2[frame2 > 90] = 0
        frame2[frame2 != 0] = 255

        cv2.imshow("L_frame", L_frame)
        cv2.imshow("frame", frame)
        cv2.imshow("L_frame2", L_frame2)
        cv2.imshow("frame2", frame2)
        k = cv2.waitKey(10)
        if k == 27: break
        
    out_bg.release()
    out_fg.release()
    cv2.destroyAllWindows()

    # plt.imsave(fname="outputs/original.jpg", arr=np.reshape(M[:,550], dims), cmap='gray')
    # plt.imsave(fname="outputs/low_rank.jpg", arr=np.reshape(low_rank[:,550], dims), cmap='gray')
    # plt.imsave(fname="outputs/low_rank_normalized.jpg", arr=np.reshape(normalized[:,550], dims), cmap='gray')


def normalize_matrix(mat: np.ndarray):
    """Helper for normalizing values in the matrix between 0 and 1."""
    return cv2.normalize(mat, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)



if __name__ == "__main__":
    # video_frame_by_frame_background()
    # image_background()
    video_all_at_once_background()
    # image_denoise()
