<h1>Clear Nights Ahead: Towards Multi-Weather <br> Nighttime Image Restoration</h1>
<a href="https://github.com/henlyta">Yuetong Liu</a>,
<a href="#">Yunqiu Xu</a>,
<a href="#">Yang Wei</a>,
<a href="#">Xiuli Bi</a>,
<a href="#">Bin Xiao</a>

---
 
📄 <strong><a href="https://arxiv.org/abs/2505.16479">Paper</a></strong>
🌐 <strong><a href="https://henlyta.github.io/ClearNight/index.html">Project Webpage</a></strong>
📁 <strong><a href="https://huggingface.co/datasets/YuetongLiu/AllWeatherNight">AllWeatherNight Dataset</a> </strong>

---

<!-- <div class="section">
    <h2>🗃️ Datasets</h2>  
    We propose a realistic dataset for nighttime multi-weather image restoration:<br> <br>
      📁 <strong>AllWeatherNight</strong>: <a href="https://huggingface.co/datasets/YuetongLiu/AllWeatherNight">Here</a>  
</div> -->

<div class="section">
    <h2>Abstract</h2>  
    Restoring nighttime images affected by multiple adverse weather conditions is a practical yet under-explored research problem, as multiple weather degradations usually coexist in the real world alongside various lighting effects at night. This paper first explores the challenging multi-weather nighttime image restoration task, where various types of weather degradations are intertwined with flare effects. To support the research, we contribute the AllWeatherNight dataset, featuring large-scale nighttime images with diverse compositional degradations. By employing illumination-aware degradation generation, our dataset significantly enhances the realism of synthetic degradations in nighttime scenes, providing a more reliable benchmark for model training and evaluation.
Additionally, we propose ClearNight, a unified nighttime image restoration framework, which effectively removes complex degradations in one go. Specifically, ClearNight extracts Retinex-based dual priors and explicitly guides the network to focus on uneven illumination regions and intrinsic texture contents respectively, thereby enhancing restoration effectiveness in nighttime scenarios. Moreover, to more effectively model the common and unique characteristics of multiple weather degradations, ClearNight performs weather-aware dynamic specificity and commonality collaboration that adaptively allocates optimal sub-networks associated with specific weather types. Comprehensive experiments on both synthetic and real-world images demonstrate the necessity of the AllWeatherNight dataset and the superior performance of ClearNight.<br /><br />
    <img src="https://github.com/henlyta/ClearNight/blob/page/static/image/frame.png?raw=True">
</div>

---


  <div class="section">
    <h2>🚀 Getting Started</h2>
<!--     <h3>🏋️‍♂️ Training</h3>
    <pre><code>python training_ClearNight.py --Retinex_decomp True</code></pre> -->
The source code will be released soon.
<!--<h3>🧪 Testing</h3> -->
<!--   <pre><code>python testing_ClearNight.py --Retinex_decomp True</code></pre> -->
  </div>

---

<div class="section">
    <h2>📖 Citation</h2>
    <div class="citation">If you find our work is helpful to your research, please cite the papers as follows: <br /><br />
      <pre>
      <code>    
@inproceedings{aaai2026clearnight,
  title={Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration},
  author={Liu, Yuetong and Xu, Yunqiu and Wei, Yang and Bi, Xiuli and Xiao, Bin},
  booktitle={AAAI},
  year={2026}
}       
      </code>
      </pre>
    </div>
</div>


  ---


  <div class="section">
    <h2>📬 Contact</h2>
    <p>If you have any questions, please contact <a href="mailto:d230201022@stu.cqupt.edu.cn">d230201022@stu.cqupt.edu.cn</a></p>
  </div>

  

