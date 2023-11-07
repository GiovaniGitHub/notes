use smartcore::linalg::{
    naive::dense_matrix::DenseMatrix, qr::QRDecomposableMatrix, svd::SVDDecomposableMatrix,
    BaseMatrix, BaseVector,
};

use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::utils::types::TypeFactoration;
use crate::utils::utils::expand_matrix;

pub struct RBFRegression {
    pub num_center: usize,
    pub centers: DenseMatrix<f32>,
    pub beta: f32,
    pub weight: DenseMatrix<f32>,
    pub type_factoration: Option<TypeFactoration>,
}

impl RBFRegression {
    pub fn new(
        beta: f32,
        num_center: usize,
        num_cols: usize,
        type_factoration: Option<TypeFactoration>,
    ) -> RBFRegression {
        let mut coefficients_centers: Vec<f32> = Vec::new();
        let mut coefficients_weight: Vec<f32> = Vec::new();
        for _ in 0..num_center {
            coefficients_weight.push(1.0);
            for _ in 0..num_cols {
                coefficients_centers.push(1.0);
            }
        }

        RBFRegression {
            num_center,
            centers: DenseMatrix::from_array(num_center, num_cols, &coefficients_centers),
            beta,
            weight: DenseMatrix::from_array(num_center, 1, &coefficients_weight),
            type_factoration,
        }
    }

    pub fn fit(&mut self, x: &DenseMatrix<f32>, y: &DenseMatrix<f32>) {
        let (_, n_columns) = self.centers.shape();
        let x = expand_matrix(&x, n_columns);

        let (num_rows, num_cols) = x.shape();
        // let mut index: Vec<usize> = (0..num_rows).collect();

        // index.shuffle(&mut thread_rng());
        let mut index = [
            296, 165, 211, 322, 755, 685, 505, 412, 555, 502, 280, 103, 615, 933, 283, 619, 537,
            994, 719, 764, 669, 860, 786, 393, 507, 802, 316, 893, 468, 122, 203, 588, 362, 777,
            846, 575, 113, 155, 50, 182, 26, 11, 787, 790, 1, 22, 124, 809, 826, 477, 835, 471,
            992, 366, 585, 55, 195, 461, 937, 172, 414, 439, 562, 954, 670, 652, 107, 549, 207,
            756, 718, 761, 565, 983, 714, 649, 327, 40, 986, 771, 403, 126, 982, 959, 326, 67, 998,
            74, 47, 100, 361, 980, 636, 497, 836, 57, 128, 183, 552, 964, 214, 675, 550, 409, 469,
            0, 415, 285, 238, 359, 616, 168, 241, 278, 196, 679, 663, 219, 56, 865, 276, 929, 877,
            665, 224, 495, 284, 357, 694, 379, 631, 254, 102, 654, 480, 442, 384, 133, 805, 825,
            708, 267, 200, 932, 595, 428, 218, 722, 255, 79, 888, 697, 184, 487, 500, 645, 265,
            752, 947, 625, 709, 815, 742, 894, 188, 470, 702, 646, 143, 923, 299, 624, 300, 840,
            485, 554, 856, 198, 530, 813, 922, 696, 106, 916, 780, 454, 548, 539, 510, 597, 59,
            354, 692, 314, 7, 496, 734, 895, 434, 960, 572, 603, 311, 153, 591, 668, 16, 418, 268,
            490, 508, 321, 213, 108, 33, 710, 536, 347, 192, 286, 167, 38, 725, 73, 967, 593, 886,
            387, 568, 944, 205, 664, 43, 160, 609, 939, 902, 402, 4, 607, 151, 464, 762, 949, 117,
            66, 251, 175, 634, 243, 282, 93, 906, 811, 587, 261, 360, 479, 489, 210, 873, 330, 866,
            21, 872, 989, 990, 791, 138, 820, 137, 814, 567, 553, 913, 35, 934, 123, 613, 72, 955,
            837, 158, 839, 369, 808, 116, 760, 499, 818, 546, 706, 909, 187, 878, 257, 520, 228,
            309, 976, 125, 305, 91, 24, 19, 698, 396, 453, 129, 749, 829, 209, 14, 733, 656, 146,
            262, 242, 561, 941, 687, 823, 775, 425, 750, 39, 466, 341, 252, 98, 385, 364, 346, 273,
            216, 977, 789, 32, 908, 592, 758, 130, 757, 475, 946, 541, 912, 348, 127, 713, 928,
            630, 612, 766, 534, 253, 325, 900, 308, 225, 144, 85, 202, 633, 427, 569, 586, 77, 389,
            351, 509, 968, 988, 149, 707, 78, 925, 61, 754, 458, 407, 433, 370, 843, 582, 834, 517,
            851, 23, 566, 405, 831, 457, 644, 436, 524, 312, 105, 987, 169, 223, 642, 329, 785,
            417, 304, 432, 377, 647, 383, 82, 673, 259, 838, 83, 272, 700, 400, 927, 136, 810, 237,
            564, 339, 513, 201, 731, 279, 174, 574, 627, 951, 608, 981, 147, 302, 919, 506, 423,
            684, 446, 995, 171, 563, 807, 234, 63, 62, 610, 429, 800, 142, 712, 49, 431, 695, 53,
            177, 783, 448, 759, 522, 247, 271, 397, 512, 662, 623, 600, 621, 295, 930, 632, 179,
            751, 110, 806, 705, 571, 715, 920, 121, 333, 884, 556, 331, 643, 862, 419, 275, 119,
            905, 208, 164, 542, 463, 936, 492, 424, 81, 44, 874, 292, 134, 336, 10, 822, 945, 629,
            832, 854, 943, 386, 666, 559, 307, 622, 87, 313, 844, 516, 867, 889, 31, 18, 376, 674,
            863, 703, 375, 435, 875, 723, 486, 732, 602, 287, 420, 794, 691, 96, 965, 824, 483,
            114, 42, 904, 189, 950, 358, 230, 966, 88, 511, 763, 788, 827, 753, 303, 323, 842, 401,
            770, 773, 604, 157, 109, 324, 474, 297, 741, 352, 717, 236, 440, 940, 338, 498, 730,
            388, 996, 484, 3, 197, 465, 294, 86, 220, 245, 716, 529, 589, 381, 239, 140, 796, 80,
            638, 150, 746, 599, 577, 350, 518, 320, 94, 145, 337, 841, 95, 833, 398, 594, 481, 494,
            69, 176, 545, 961, 248, 899, 41, 845, 132, 422, 682, 493, 748, 367, 583, 736, 890, 910,
            334, 115, 781, 404, 6, 190, 864, 576, 70, 92, 60, 628, 680, 727, 373, 395, 711, 721,
            410, 15, 356, 869, 451, 979, 747, 349, 618, 68, 558, 772, 657, 288, 573, 606, 918, 531,
            667, 641, 803, 426, 170, 459, 199, 141, 581, 152, 868, 266, 776, 298, 892, 515, 244,
            891, 693, 289, 391, 17, 660, 897, 655, 701, 924, 525, 850, 413, 521, 570, 101, 526,
            861, 215, 212, 880, 528, 473, 870, 64, 12, 335, 677, 978, 76, 131, 948, 704, 317, 543,
            159, 917, 816, 227, 319, 310, 778, 27, 380, 560, 306, 896, 37, 120, 301, 232, 855, 580,
            661, 342, 997, 438, 881, 246, 738, 871, 48, 973, 449, 598, 13, 658, 315, 828, 578, 852,
            374, 353, 293, 222, 443, 456, 605, 519, 935, 250, 441, 161, 798, 853, 926, 984, 540,
            363, 963, 584, 501, 779, 281, 858, 345, 111, 962, 58, 460, 372, 75, 538, 793, 204, 527,
            240, 97, 745, 99, 720, 514, 690, 857, 921, 699, 185, 25, 371, 51, 503, 488, 812, 186,
            452, 821, 848, 421, 767, 885, 887, 532, 411, 29, 672, 193, 614, 445, 915, 406, 953,
            590, 974, 801, 849, 340, 270, 975, 328, 726, 226, 89, 332, 450, 9, 993, 883, 901, 601,
            194, 34, 859, 28, 166, 653, 163, 491, 178, 84, 914, 879, 156, 355, 999, 181, 958, 478,
            985, 65, 676, 104, 797, 221, 343, 191, 689, 467, 291, 139, 744, 678, 911, 444, 620,
            229, 154, 648, 769, 882, 626, 804, 830, 206, 737, 264, 231, 46, 635, 263, 256, 743,
            249, 260, 290, 765, 957, 523, 173, 233, 688, 847, 659, 792, 931, 30, 938, 972, 817, 52,
            533, 378, 683, 650, 482, 382, 365, 394, 903, 544, 258, 368, 637, 71, 535, 36, 671, 390,
            416, 724, 90, 135, 269, 639, 768, 551, 740, 611, 596, 617, 739, 274, 8, 399, 640, 235,
            952, 956, 547, 729, 970, 819, 735, 686, 799, 20, 991, 118, 728, 579, 472, 782, 54, 681,
            45, 318, 455, 392, 2, 277, 774, 504, 447, 942, 795, 5, 112, 180, 162, 217, 876, 557,
            408, 148, 971, 344, 462, 476, 430, 969, 907, 784, 898, 651, 437,
        ];
        let mut count = 0;
        for i in index {
            for j in 0..num_cols {
                self.centers.set(count, j, x.get(i, j));
            }
            count = count + 1;
            if count >= self.num_center {
                break;
            }
        }

        let gradient = calculate_gradient(&x, &self.centers, &self.beta);
        for i in 0..gradient.shape().0 {
            for j in 0..gradient.shape().1 {
                print!("{:?}|", gradient.get(i, j));
            }
            println!("");
        }
        match self.type_factoration {
            Some(TypeFactoration::SVD) => {
                self.weight = gradient
                    .transpose()
                    .matmul(&gradient)
                    .svd_solve_mut(gradient.transpose().matmul(&y).clone())
                    .unwrap();
            }
            Some(TypeFactoration::QR) => {
                self.weight = gradient
                    .transpose()
                    .matmul(&gradient)
                    .qr_solve_mut(gradient.transpose().matmul(&y).clone())
                    .unwrap();
            }
            Some(TypeFactoration::CHOLESKY) => {
                self.weight = gradient
                    .transpose()
                    .matmul(&gradient)
                    .qr_solve_mut(gradient.transpose().matmul(&y).clone())
                    .unwrap();
            }
            _ => {
                let value = gradient.transpose().matmul(&gradient);
                for i in 0..value.shape().0 {
                    println!("{:?}", value.get_row(i));
                }

                self.weight = gradient
                    .transpose()
                    .matmul(&gradient)
                    .svd_solve_mut(gradient.transpose().matmul(&y).clone())
                    .unwrap();
            }
        }
    }

    pub fn predict(&mut self, x: &DenseMatrix<f32>) -> DenseMatrix<f32> {
        let (_, n_columns) = self.centers.shape();
        let x = expand_matrix(&x, n_columns);

        return calculate_gradient(&x, &self.centers, &self.beta).matmul(&self.weight);
    }
}

pub fn calculate_gradient(
    x: &DenseMatrix<f32>,
    centers: &DenseMatrix<f32>,
    beta: &f32,
) -> DenseMatrix<f32> {
    let (num_row, _) = x.shape();
    let (num_center, _) = centers.shape();
    let mut gradient_vector: Vec<f32> = Vec::new();
    for i in 0..num_row {
        // let x_row = x.get_row(i);
        for j in 0..num_center {
            let norm: f32 = centers
                .get_row(j)
                .sub(&x.get_row(i))
                .iter()
                .map(|r| r.powf(2.0))
                .sum();
            gradient_vector.push((-beta * norm).exp());
        }
    }
    let resp = DenseMatrix::from_array(num_row, num_center, &gradient_vector);
    // println!("{:?}", resp.get_row(0));
    return resp;
}
