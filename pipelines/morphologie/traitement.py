from utils.io_utils import lire_image_rgb, redimensionner_bilineaire
from utils.color import rgb_vers_hsl, rgb_vers_gris
from pipelines.morphologie.filters import flou_gaussien, noyau_gaussien_1d, convolution_1d_lignes, convolution_1d_colonnes, gradient_sobel
from pipelines.morphologie.segmentation import seuil_otsu, histogramme_u8, egaliser_histogramme
from pipelines.morphologie.morphology import erosion_binaire, dilatation_binaire, ouverture_binaire, fermeture_binaire, composantes_connexes
from pipelines.morphologie.detection import extraire_composantes_utiles, estimer_nombre_depuis_composantes, detection_principale, detection_piece_unique, compter_pieces
from pipelines.morphologie.config import *
