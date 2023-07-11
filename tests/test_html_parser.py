import unittest

from .test_common import CPTestCase
from src.html_parser import parse_html
from src.cosmos import get_all_papers


class TestParseHTML(CPTestCase):
    def test_parse_html(self):
        papers = parse_html(self.config)
        assert len(papers) == 3

        db_items = get_all_papers(self.config.db, 3, force=True)
        assert len(db_items) == 3

        expected = [
            {
                "url": "https://openaccess.thecvf.com/content/CVPR2023/html/Ci_GFPose_Learning_3D_Human_Pose_Prior_With_Gradient_Fields_CVPR_2023_paper.html",
                "pdf_url": "https://openaccess.thecvf.com/content/CVPR2023/papers/Ci_GFPose_Learning_3D_Human_Pose_Prior_With_Gradient_Fields_CVPR_2023_paper.pdf",
                "arxiv_id": "2212.08641",
                "title": "GFPose: Learning 3D Human Pose Prior With Gradient Fields",
                "abstract": "Learning 3D human pose prior is essential to human-centered AI. Here, we present GFPose, a versatile framework to model plausible 3D human poses for various applications. At the core of GFPose is a time-dependent score network, which estimates the gradient on each body joint and progressively denoises the perturbed 3D human pose to match a given task specification. During the denoising process, GFPose implicitly incorporates pose priors in gradients and unifies various discriminative and generative tasks in an elegant framework. Despite the simplicity, GFPose demonstrates great potential in several downstream tasks. Our experiments empirically show that 1) as a multi-hypothesis pose estimator, GFPose outperforms existing SOTAs by 20% on Human3.6M dataset. 2) as a single-hypothesis pose estimator, GFPose achieves comparable results to deterministic SOTAs, even with a vanilla backbone. 3) GFPose is able to produce diverse and realistic samples in pose denoising, completion and generation tasks.",
            },
            {
                "url": "https://openaccess.thecvf.com/content/CVPR2023/html/Xu_CXTrack_Improving_3D_Point_Cloud_Tracking_With_Contextual_Information_CVPR_2023_paper.html",
                "pdf_url": "https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_CXTrack_Improving_3D_Point_Cloud_Tracking_With_Contextual_Information_CVPR_2023_paper.pdf",
                "arxiv_id": "2211.08542",
                "title": "CXTrack: Improving 3D Point Cloud Tracking With Contextual Information",
                "abstract": "3D single object tracking plays an essential role in many applications, such as autonomous driving. It remains a challenging problem due to the large appearance variation and the sparsity of points caused by occlusion and limited sensor capabilities. Therefore, contextual information across two consecutive frames is crucial for effective object tracking. However, points containing such useful information are often overlooked and cropped out in existing methods, leading to insufficient use of important contextual knowledge. To address this issue, we propose CXTrack, a novel transformer-based network for 3D object tracking, which exploits ConteXtual information to improve the tracking results. Specifically, we design a target-centric transformer network that directly takes point features from two consecutive frames and the previous bounding box as input to explore contextual information and implicitly propagate target cues. To achieve accurate localization for objects of all sizes, we propose a transformer-based localization head with a novel center embedding module to distinguish the target from distractors. Extensive experiments on three large-scale datasets, KITTI, nuScenes and Waymo Open Dataset, show that CXTrack achieves state-of-the-art tracking performance while running at 34 FPS.",
            },
            {
                "url": "https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Deep_Frequency_Filtering_for_Domain_Generalization_CVPR_2023_paper.html",
                "pdf_url": "https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Deep_Frequency_Filtering_for_Domain_Generalization_CVPR_2023_paper.pdf",
                "arxiv_id": "2203.12198",
                "title": "Deep Frequency Filtering for Domain Generalization",
                "abstract": "Improving the generalization ability of Deep Neural Networks (DNNs) is critical for their practical uses, which has been a longstanding challenge. Some theoretical studies have uncovered that DNNs have preferences for some frequency components in the learning process and indicated that this may affect the robustness of learned features. In this paper, we propose Deep Frequency Filtering (DFF) for learning domain-generalizable features, which is the first endeavour to explicitly modulate the frequency components of different transfer difficulties across domains in the latent space during training. To achieve this, we perform Fast Fourier Transform (FFT) for the feature maps at different layers, then adopt a light-weight module to learn attention masks from the frequency representations after FFT to enhance transferable components while suppressing the components not conducive to generalization. Further, we empirically compare the effectiveness of adopting different types of attention designs for implementing DFF. Extensive experiments demonstrate the effectiveness of our proposed DFF and show that applying our DFF on a plain baseline outperforms the state-of-the-art methods on different domain generalization tasks, including close-set classification and open-set retrieval.",
            },
        ]

        for p, d, e in zip(papers, db_items, expected):
            for key in e.keys():
                assert p[key] == e[key] and d[key] == e[key]


if __name__ == "__main__":
    unittest.main()
