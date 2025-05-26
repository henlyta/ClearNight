<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Clear Nights Ahead</title>
  <style>
    body {
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      line-height: 1.6;
      margin: 40px;
      background-color: #f9f9f9;
      color: #333;
    }
    h1, h2, h3 {
      color: #2c3e50;
    }
    code, pre {
      background: #f4f4f4;
      padding: 10px;
      border-radius: 6px;
      display: block;
      overflow-x: auto;
    }
    a {
      color: #2980b9;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    .author-list {
      margin-top: -10px;
      margin-bottom: 20px;
    }
    .author-list a {
      display: inline-block;
      margin-right: 10px;
    }
    .section {
      margin-bottom: 40px;
    }
    .citation {
      background: #fdf6e3;
      border-left: 4px solid #e67e22;
      padding: 15px;
    }
  </style>
</head>
<body>

<h1>Clear Nights Ahead: Towards Multi-Weather <br> Nighttime Image Restoration</h1>
  <div class="author-list">
    <a href="https://github.com/henlyta">Yuetong Liu</a>
    <a href="#">Yunqiu Xu</a>
    <a href="#">Yang Wei</a>
    <a href="#">Xiuli Bi</a>
    <a href="#">Bin Xiao</a>
  </div>

<div class="section">
    📄 <strong><a href="https://arxiv.org/abs/2505.16479">Paper on arXiv</a></strong><br/>
    📂 <strong><a href="https://huggingface.co/datasets/YuetongLiu/AllWeatherNight">Dataset on Hugging Face (AllWeatherNight)</a></strong>
  </div>


<h3>Datasets</h3>

AllWeatherNight: <a href="https://huggingface.co/datasets/YuetongLiu/AllWeatherNight">here</a>

<h3>Training and Testing</h3>

##Train

python training_ClearNight.py --Retinex_decomp True

##Test

python testing_ClearNight.py --Retinex_decomp True

<h3>Citation</h3>

If you find our work is helpful to your research, please cite the papers as follows:
<div>
<pre>
@article{liu2025clearnight,
      title={Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration}, 
      author={Liu, Yuetong and Xu, Yunqiu and Wei, Yang and Bi, Xiuli and Xiao, Bin},
      year={2025},
      journal={arXiv preprint arXiv:2505.16479},
      year={2025}
}
</pre>
</div>
<h3>Contact</h3>
If you have any questions, please contact d230201022@stu.cqupt.edu.cn
</body>
</html>
