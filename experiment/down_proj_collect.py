import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import numpy as np
import matplotlib.pyplot as plt

model_name = "/home/xxf/NewVLLM/models/deepseek-v2-lite"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto', offload_buffers=True)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

print(model)

collect_inputs = []

def hook(module, input, output):
    ## collect inputs
    collect_inputs.append(input[0].cpu())

for name, layer in model.model.layers.named_modules():
    if "down_proj" in name and ".experts." in name:
        layer.register_forward_hook(hook)

text1 = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
text2 =  "Mathematics is an area of knowledge that includes the topics of numbers, formulas and related structures, shapes and the spaces in which they are contained"
text3 = "A computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations (computation). Modern digital electronic computers can"

text4 = "A mobile phone (or cellphone[a]) is a portable telephone that can make and receive calls over a radio frequency link while the user is moving within a telephone service area, as opposed to a fixed-location phone (landline phone). The radio frequency"
text5 = "Big data primarily refers to data sets that are too large or complex to be dealt with by traditional data-processing application software. Data with many entries (rows) offer greater statistical power"
text6 = "Large Language Models (LLMs) such as GPTs and LLaMa have ushered in a revolution in machine intelligence, owing to their exceptional capabilities in a wide range of machine learning tasks. However, the transition of LLMs from data centers to edge devices presents a set of challenges and opportunities."

# text7 = """
# Summarize the given article about AI in healthcare. AI has enabled the healthcare sector to revolutionize patient care in a number of ways. AI technologies are capable of accelerating the process of diagnosing diseases and have the ability to detect all kinds of abnormalities which may have otherwise gone undetected. Through the use of AI and machine learning, healthcare providers are able to reduce costs and increase efficiency by streamlining processes like appointment scheduling or hospital equipment tracking. AI is also being used to aid healthcare professionals in performing complex surgeries.
# Summarize the main points of the following article about social media and its impacts. Social media has become an incredibly powerful tool in the 21st century. It allows people to instantly share images and videos, connect with friends around the world and access news and updates from a variety of sources. While this can be a positive thing, there are some downsides, too. Social media can become a time-sink, taking time away from more productive activities, and can have a serious impact on a person’s mental health. It can also be used to spread false information and cyberbullying.
# Summarize the article under the  title "Risks of childhood obesity". Childhood obesity is a growing concern around the world. There is an increase in the prevalence of chronic diseases like type 2 diabetes and hypertension. As a result, there is an increasing risk of physical and psychological problems, like sleep apnea and social isolation. Childhood obesity is also associated with a higher risk of developing cardiovascular disease and other serious health conditions. Furthermore, this growing health concern can lead to financial costs for families, as well as schools and the government.
# Summarize this article about the history of the Internet in Japan. The Internet has had a long and convoluted history in Japan since its earliest beginnings in the early 1990s. After some initial experimentation with international networks, Japan opted to develop its own advanced research networks, allowing it to remain ahead of the curve in terms of knowledge and technological innovation. The development of the web played a large role in the growth of the industry, and the creation and adaptation of new technologies and services allowed the Internet to become an integral part of daily life in the nation.
# List the main theme of the article. This article provides an exploration of the implications of noise, both externally-introduced to a system, and internally-generated within a system, on collective intelligence. It is argued that external noise can drive a precarious collective intelligence system towards either order or chaos via ‘phase transitions’. Decomposition of external noise in terms of regime control and critical control is discussed as possible mechanisms for phase transitions. Internal noise is discussed with respect to the question: “How resilient is a collective intelligence system against internal noise?” Distinguishing between noise-supported and noise-stabilized collective intelligence systems concludes the article.
# Summarize the given text in 5-8 sentences. The German Health Interview and Examination Survey for Children and Adolescents (KiGGS) is one of the largest and most comprehensive surveys ever conducted on the health and lifestyle of young people in Germany. It started in 2003 and looks at the physical and mental health of German children and adolescents between the ages of 0 and 17. In addition to providing insight into the general health of the population, KiGGS also investigates other aspects of health such as physical activity, nutrition, and family environment, as well as socio-economic determinants of health.
# Summarize the following article, highlight all major points in the summary. Limited partnerships (LPs) are a flexible form of business structure that provides owners of the business both limited liability and tax advantages. With a limited partnership, there are at least one general partner who is personally liable for the obligations of the business, and limited partners who nominally manage the business, but have limited involvement and do not have personal liability. This can help limit the exposure of individual partners in a business venture, ensuring that the losses experience by one partner is contained to the extent of their invested capital.
# """

text8 = """
Provide a summary for the passage given below. Artificial intelligence (AI) is a broad field that seeks to mimic or augment human capabilities through information processing technology and algorithms. AI has been an area of study since the 1950s, but has gained popular attention in recent years due to vast improvements in processing power. While some fear existential threats such as a Terminator-like hypothetical AI takeover, most of the practical applications of AI are in mundane areas such as image recognition, natural language processing, and task automation. AI is used to improve or automate mundane tasks, quickly identify patterns, and predict future events.
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. "Hansel and Gretel" is a German fairy tale collected by the Brothers Grimm and published in 1812 as part of Grimm's Fairy Tales. It is also known as Little Step Brother and Little Step Sister. Question: Whom did Hansel and Gretel meet during their adventures? Helpful Answer:
Summarize the following text in one paragraph. Theories of personal identity have changed over time. Philosophers in the past have focused mainly on the self, whereas modern theorists focus more on personal narratives and narrative identities. One example is Locke's definition of personal identity which states that you are the same person over time if you have personal memories of your past or if you can remember past experiences. Another example is Parfit's rejection of this notion and focus on the importance of psychological continuity, which is when you remain the same even if you are presented with a different physical world or body.
Read the following paragraph, find the central theme and summarize it in one sentence. Blockchain technology is transforming how we interact with digital assets and store data in a secure and transparent manner. It is a decentralized digital ledger that records transactions and stores vital information across several interconnected nodes. Essentially, it mitigates the risk of single point of failure, enhances security, and eliminates the need for intermediary trust in financial transactions. Blockchain is gaining traction and is appreciated by several industries such as finance, supply chain, and health care.
Given a short story, rewrite it so that it takes place in a dystopian setting and maintain the original focus of the story. As Sarah looked out the window, admiring the bright sunny sky and colorful gardens of the park, she listened to the laughter and music filling the air. She thought it was amazing how the people were enjoying the concert on this perfect weekend, creating a blissful atmosphere. Little Timmy, her younger brother, beamed with excitement as he played with his friends, their cheerful giggles echoing around.
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Bavaria has a unique culture, largely because of the state's Catholic heritage and conservative traditions.Bavarians have traditionally been proud of their culture, which includes a language, cuisine, architecture, festivals and elements of Alpine symbolism. The state also has the second largest economy among the German states by GDP figures, giving it a status as a wealthy German region. Question: What is the weather like in Bavaria? Helpful Answer:
Rewrite the following paragraph to make it simpler and more appropriate for intermediate English language learners. The unprecedented blizzard knocked out power in large portions of the region, obliterating any semblance of the digitally connected world their denizens were accustomed to. For the first time in a long while, people were stranded in their homes without access to the internet or modern comforts. With long-forgotten board games suddenly gaining newfound relevancy, families spent hours together in these frigid circumstances, reminding them of a time when such simple pleasures were the norm.
Create a summary of the text below The correct way to write and illustrate a story is to start with a seed idea. Then, you will use what you know— your characters, setting, and plot— to bring the story to life. This can be depicted through illustrations, captions, and dialogues. As the story comes alive, use transitions and setting descriptions to break up the sequences and drive the story forward. Finally, use the climax to show what the characters have been striving for and how their actions have resulted in a satisfying endpoint.
Identify three main reasons from the given text why the author thinks public transportation needs improvement. Public transportation systems have been struggling to keep up with the growing demand in recent years. The infrastructure has been deteriorating and there is a lack of maintenance and modernization measures. The lack of funding for public transportation has led to less frequent service, overcrowded vehicles, and an insufficient number of routes to suit all users. The consequences of poor public transportation systems include increased traffic congestion and reduced air quality due to higher numbers of private vehicles on the road. Additionally, poorly planned transportation systems can contribute to social inequality by limiting the mobility of low-income individuals who cannot afford private vehicles.
Put yourself in the shoes of a movie critic who recently watched a film. Describe the plot and provide an opinion on the characters and overall film quality. Title: 'Fighting for Freedom'. Genre: Drama. Synopsis: The story revolves around an underprivileged boxer who dreams of becoming a champion. He eventually crosses paths with a renowned boxing coach who sees his potential and decides to train him. As the story unfolds, our protagonist fights through various challenges and ultimately becomes a symbol of hope for those from similar, humble beginnings.
Rewrite the given paragraph to make it more concise without losing important information. There are a variety of reasons that people love to go on vacations to tropical destinations like Hawaii. One of those reasons is that the weather is consistently warm, which is beneficial for those who come from cold climates and want to escape the frigid temperatures for a period of time. Another reason is that there are numerous gorgeous, picturesque beaches where vacationers can relax or engage in activities like snorkeling, surfing, or building sandcastles. Additionally, tropical locations tend to have a laid-back, carefree atmosphere that helps people unwind and forget about their everyday stress and responsibilities.
Summarize the following article about facial recognition technology in 100 words or less. Facial recognition technology is becoming increasingly commonplace in the modern world. Companies are using it to verify customers, law enforcement is using it to investigate suspects, and even our phones can now unlock with just a glance. While there is no doubt that facial recognition could have positive applications, there is also the potential for misuse. Privacy advocates have raised concerns with regard to the accuracy of the technology and the potential for it to be used by governments to track their citizens.
Assume you are a customer support representative for a tech company. The client received a product that did not meet their expectations and has written a complaint. Respond empathically and professionally, addressing their concerns and providing a solution. I've been using your company's software for a while, but the latest version is just terrible. It's buggy, slow, and it keeps crashing my computer. I don't know how you could release such a subpar product. I need a fix for this immediately or I'm going to demand a refund.
Please summarize the main events in the story and explain how the characters evolve throughout the narrative. Joel and Ellie live in a post-apocalyptic world where people are struggling to survive. They must travel together across the country, fighting off dangerous creatures and other survivors, to deliver Ellie to a group called the Fireflies. Ellie was bitten by one of the infected creatures but did not turn, making her immune and valuable to finding a cure. However, as they bond and grow close, Joel learns that Ellie's life will be sacrificed to create a potential vaccine, forcing him to make a difficult moral choice.
Identify three examples of irony found in this short passage. The bank heist had been meticulously planned down to the smallest detail, and every member of the gang had their role to play. Unfortunately, the robbery was doomed from the outset, because the mastermind behind it all had neglected to take his own advice. Ironically, he had relied on an unreliable accomplice, and now he would pay the price. The very same gang member who had so often warned others to never underestimate the power of the law was now staring down the barrel of a police officer's gun.
Summarize the points discussed in the following article. In recent years, the use of artificial intelligence (AI) in healthcare has become increasingly commonplace. AI applications promise to make healthcare more efficient, cost-effective, and accurate, but their application raises ethical questions. To ensure a safe technology, researchers need to consider the potential harm of AI systems and be emotionally engaged with the outcome of their machine learning processes. Artificial Intelligence in healthcare should be supervised and regulated to ensure transparency about its decision making processes and its resultant impacts. Furthermore, algorithms should be tested for bias to avoid any unfair impacts, and there should be clear guidelines for development, use and goverance of AI systems.
Explain the motives of the villain in the given story excerpt and discuss how their actions affected the main character. In the small town of Evergrove, the villain, Lord Malveron, had a deep-rooted disdain for the town's people, stemming from his neglected childhood. He devised a plan to control the town by poisoning the town's water supply. The main character, Amara, was unaware of Lord Malveron's sinister plan until she discovered a hidden note detailing his intentions. Now, Amara must confront her own fears and fight against her family's enemy.
Analyze the following text and provide an opinion on whether the author is arguing in favor or against the use of technology in education. While technology undoubtedly provides students with access to a wealth of knowledge and resources, it also has the potential to be highly distracting. Today's students are drawn to the constant stimuli provided by digital devices, making it difficult to remain focused. Teachers struggle to keep their students' attention on the task at hand. Time spent on educational technology could be better invested in traditional teaching methods, which provide direct interaction and engagement between teachers and students.
Rewrite the following paragraph in a more concise manner, preserving only the most important information. The small, rural community was comprised of just a few houses on a long, dusty road. Each one was a different color and had a unique mailbox with intricate designs. Everyone who lived there knew each other well, and they often had gatherings at the heart of the community. Children played outdoors during the day, and the adults would spend their evenings together, talking about life and sharing memories. They had a strong bond and truly enjoyed one another's company, creating an atmosphere of togetherness, love and support that made the entire community a special place for everyone involved.
Summarize an article about the current state of innovation in the financial services industry. The financial services industry has seen sustained growth and innovation over the last decade. This range of new investment opportunities has led to a surge in the number of venture capital and private equity funds targeting the sector. While these investments are often made in startups, the industry’s typical incumbents, from banks to insurers, are also investing heavily in new digital products or revamping existing ones. More focus has been placed on customer segmentation strategies and digital marketing strategies, as well as new data governance strategies and customer personalization tactics.
Read the given text, and step-by-step, think about how you would create multiple choice questions to test comprehension. The industrial revolution began in Britain around 1760 and marked a significant shift in production methods, as well as social structures. For the first time, goods were produced using machines powered by steam engines rather than by human labor alone. This led to huge increases in efficiency and enabled the mass production of items such as textiles, iron, and steel. The industrial revolution also brought about urbanization, as people moved from rural areas to work in factories in the cities.
Given an article, summarize the main points in 5 sentences. Article: AI chatbots are computer programs that use natural language processing (NLP), artificial intelligence, and machine learning (ML) technologies to simulate human conversations with customers. AI chatbots have become increasingly popular in the customer service industry, as they can quickly provide answers to customer questions, offer product and service advice, and guide customers through complex procedures. AI chatbots can also save businesses time and money by automating repetitive customer service tasks, such as account management and support, and helping reduce customer service costs.
Summarize the given text using five sentences. The World Wildlife Fund (WWF) is an international organization committed to conservation of the world's natural resources, wildlife and ecosystems. Established in 1961 as the International Union for Conservation of Nature and Natural Resources, WWF has since grown to become one of the largest conservation organizations in the world. Their mission seeks to conserve and protect nature and reduce the most pressing threats to the diversity of life and the planet. To protect the environment, WWF works on various levels, from grassroots and local initiatives to international agreements.
Summarize the given article about AI in healthcare. AI has enabled the healthcare sector to revolutionize patient care in a number of ways. AI technologies are capable of accelerating the process of diagnosing diseases and have the ability to detect all kinds of abnormalities which may have otherwise gone undetected. Through the use of AI and machine learning, healthcare providers are able to reduce costs and increase efficiency by streamlining processes like appointment scheduling or hospital equipment tracking. AI is also being used to aid healthcare professionals in performing complex surgeries.
Summarize the main points of the following article about social media and its impacts. Social media has become an incredibly powerful tool in the 21st century. It allows people to instantly share images and videos, connect with friends around the world and access news and updates from a variety of sources. While this can be a positive thing, there are some downsides, too. Social media can become a time-sink, taking time away from more productive activities, and can have a serious impact on a person’s mental health. It can also be used to spread false information and cyberbullying.
Summarize the article under the  title "Risks of childhood obesity". Childhood obesity is a growing concern around the world. There is an increase in the prevalence of chronic diseases like type 2 diabetes and hypertension. As a result, there is an increasing risk of physical and psychological problems, like sleep apnea and social isolation. Childhood obesity is also associated with a higher risk of developing cardiovascular disease and other serious health conditions. Furthermore, this growing health concern can lead to financial costs for families, as well as schools and the government.
Summarize this article about the history of the Internet in Japan. The Internet has had a long and convoluted history in Japan since its earliest beginnings in the early 1990s. After some initial experimentation with international networks, Japan opted to develop its own advanced research networks, allowing it to remain ahead of the curve in terms of knowledge and technological innovation. The development of the web played a large role in the growth of the industry, and the creation and adaptation of new technologies and services allowed the Internet to become an integral part of daily life in the nation.
List the main theme of the article. This article provides an exploration of the implications of noise, both externally-introduced to a system, and internally-generated within a system, on collective intelligence. It is argued that external noise can drive a precarious collective intelligence system towards either order or chaos via ‘phase transitions’. Decomposition of external noise in terms of regime control and critical control is discussed as possible mechanisms for phase transitions. Internal noise is discussed with respect to the question: “How resilient is a collective intelligence system against internal noise?” Distinguishing between noise-supported and noise-stabilized collective intelligence systems concludes the article.
Summarize the given text in 5-8 sentences. The German Health Interview and Examination Survey for Children and Adolescents (KiGGS) is one of the largest and most comprehensive surveys ever conducted on the health and lifestyle of young people in Germany. It started in 2003 and looks at the physical and mental health of German children and adolescents between the ages of 0 and 17. In addition to providing insight into the general health of the population, KiGGS also investigates other aspects of health such as physical activity, nutrition, and family environment, as well as socio-economic determinants of health.
Summarize the following article, highlight all major points in the summary. Limited partnerships (LPs) are a flexible form of business structure that provides owners of the business both limited liability and tax advantages. With a limited partnership, there are at least one general partner who is personally liable for the obligations of the business, and limited partners who nominally manage the business, but have limited involvement and do not have personal liability. This can help limit the exposure of individual partners in a business venture, ensuring that the losses experience by one partner is contained to the extent of their invested capital."""

text = text1 + " " + text2 + " " + text3 + " " + text4 + " " + text5 + " " + text6 + " " + text8

model.eval()


with torch.no_grad():

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)

    # inputs = tokenizer(text8, return_tensors="pt").to(model.device)
    # outputs = model(**inputs)

# print(len(collect_inputs))
# print(collect_inputs[1].shape)
# expert_idx = 0
# plot_inputs_one = collect_inputs[expert_idx]
# plot_inputs = torch.cat(collect_inputs, dim=0)
# print(plot_inputs.shape)
# ## plot the distribution of the inputs
# input_list = plot_inputs.flatten().to(torch.float32).numpy()
# ## clip all outliers, i.e., values greater than 100 and less than -100
# input_list = input_list[(input_list < 8) & (input_list > -8)]



# plt.hist(input_list, bins=200)
# plt.title("Distribution of Down Projection Inputs")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.savefig("down_proj_inputs_distribution_all.png")
# plt.close()
## plot the distribution of the inputs for the first layer



# expert_idx = 0
for expert_idx in range(20):
    plot_inputs_one = collect_inputs[expert_idx]
    input_list_one = plot_inputs_one.flatten().to(torch.float32).numpy()
    #input_list_one = input_list_one[(input_list_one < 4) & (input_list_one > -4)]
    ## 取绝对值
    input_list_one =  np.abs(input_list_one)
    print(f"max value: {np.max(input_list_one)}, ")

    # plt.hist(input_list_one, bins=200)
    import seaborn as sns
    # ## 设置宽高
    fig_w = 16/2.54 ## cm to inch
    fig_h = 8/2.54

    fontsize = 11
    # legendsize = 10
    plt.figure(figsize=(fig_w, fig_h))
    # sns.histplot(input_list_one, kde=True, bins=1000, color='darkblue')


    # 绘制直方图+KDE
    sns.histplot(
        input_list_one,
        bins=200, 
        color='darkblue',
        log_scale=True  # 等价于plt.xscale('log')，但更适配seaborn
    )

    mean_val = np.mean(input_list_one)
    median_val = np.median(input_list_one)
    q50 = np.percentile(input_list_one, 50)  # 中位数
    q25_val = np.percentile(input_list_one, 25)  # 1/4分位数
    q75_val = np.percentile(input_list_one, 75)  # 3/4分位数
    q99_val = np.percentile(input_list_one, 99)  # 99分位数

    # 添加统计参考线
    # plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')

    plt.axvline(q25_val, color='orange', linestyle='-.', label=f'25th Percentile: {q25_val:.4f}')
    plt.axvline(q50, color='green', linestyle='-.', label=f'50th Percentile: {q50:.4f}')
    plt.axvline(q75_val, color='red', linestyle='-.', label=f'75th Percentile: {q75_val:.4f}')
    # plt.axvline(q99_val, color='red', linestyle='-.', label=f'99th Percentile: {q99_val:.2f}')





    ## y轴设置科学计数法
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ## set log for x axis
    # plt.xscale('log')
    ## 设置titick label大小
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ## 设置更小的顶部边距
    plt.subplots_adjust(top=0.95, bottom=0.18, left=0.1, right=0.98)
    # plt.title("Distribution of Down Projection Inputs")
    plt.xlabel("Absolute Value of Activation", fontsize=fontsize)
    plt.ylabel("Frequency", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper left')  # 调整图例位置
    plt.savefig(f"down_proj_inputs_distribution_expert{expert_idx}.png")
    plt.close()


# 关键修复：更严格的数据清洗
# # 1. 移除NaN和无穷大值
# input_list_one = input_list_one[~np.isnan(input_list_one)]
# input_list_one = input_list_one[~np.isinf(input_list_one)]

# 2. 移除零值（避免log10(0)错误）
# input_list_one = input_list_one[input_list_one > 0]

# mean_val = np.mean(input_list_one)
# median_val = np.median(input_list_one)
# q25_val = np.percentile(input_list_one, 25)  # 1/4分位数
# q75_val = np.percentile(input_list_one, 75)  # 3/4分位数

# # 配置图形尺寸
# fig_w = 16/2.54  # 厘米转英寸
# fig_h = 8/2.54
# fontsize = 11

# plt.figure(figsize=(fig_w, fig_h))

# # 按对数间隔划分bins，适应对数刻度
# log_min = np.log10(input_list_one.min())
# log_max = np.log10(input_list_one.max())
# bins = np.logspace(log_min, log_max, 100)  # 生成100个对数间隔的bins

# # 绘制直方图和核密度估计
# sns.histplot(
#     input_list_one, 
#     kde=True, 
#     bins=bins, 
#     color='darkblue',
#     log_scale=True  # 使用对数刻度
# )

# # 添加统计参考线
# plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
# plt.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
# plt.axvline(q25_val, color='orange', linestyle='-', label=f'25th Percentile: {q25_val:.2f}')
# plt.axvline(q75_val, color='purple', linestyle='-', label=f'75th Percentile: {q75_val:.2f}')

# # 优化坐标轴和标签
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # y轴使用科学计数法
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.xlabel("Activation Value (log scale)", fontsize=fontsize)
# plt.ylabel("Frequency", fontsize=fontsize)
# plt.legend(fontsize=fontsize-1, loc='upper left')  # 调整图例位置

# # 调整布局，避免标签被截断
# plt.subplots_adjust(top=0.92, bottom=0.15, left=0.15, right=0.95)

# # 保存图像，设置较高的dpi以保证清晰度
# plt.savefig("down_proj_inputs_with_quantiles.png", dpi=300, bbox_inches='tight')
# plt.close()