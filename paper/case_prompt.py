import json
import numpy as np
from scipy.spatial.distance import cosine
import openai
openai.api_base='https://'
openai.api_key = "sk-"
import re
import random


def llm(content, print_string=True):
    if isinstance(content, str):
        messages = [{"role": "user", "content": content}]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )
        answer = []
        for chunk in response:

            if 'content' in chunk["choices"][0]['delta']:
                chunk_content = chunk["choices"][0]['delta']['content']
                if print_string:
                    print(chunk_content, end='')
                answer.append(chunk_content)

        res = ''.join(answer)
        return res
    elif isinstance(content, list):
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=content,
            stream=True
        )
        answer = []
        for chunk in response:
            if 'content' in chunk["choices"][0]['delta']:
                chunk_content = chunk["choices"][0]['delta']['content']
                if print_string:
                    print(chunk_content, end='')
                answer.append(chunk_content)
        res = ''.join(answer)
        return res


def get_embedding(x):
    if isinstance(x, str):
        return openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding']
    if isinstance(x, list):
        embeddings = openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data']
        return [i['embedding'] for i in embeddings]


def randomize_options(options):
    options = list(options.items())
    # 打乱选项的顺序
    random.shuffle(options)
    # 重新构建打乱后的选项
    randomized_options = {chr(65 + i): text for i, (_, text) in enumerate(options)}
    randomized_options_format = ''
    for k,v in randomized_options.items():
        randomized_options_format+=f'{k}:{v}\n'
    return randomized_options_format

def generate_chain_of_thought_and_answer(question, options):
    options = randomize_options(options)

    chain_of_thougth_prompt = f'''请谨慎思考,一步步的进行推理,最后得到答案,按照以下格式回答:
    -format-
    推理过程:
    1.xxx
    2.xxx
    3.xxx
    ...
    结论:xxx
    -end format-
    你需要回答的问题为:`
    {question}
    从以下中选择正确的选项
    {options}
    `

    '''

    chain_of_thougth_data = llm(chain_of_thougth_prompt)

    return chain_of_thougth_data


# 假设我们有方法来检查答案是否正确。
def is_answer_correct(chain_of_thought_data: str, answer: str):
    correct_answers_prompt = f'''请判断以下问题的回答是不是正确的:{chain_of_thought_data}
    这个问题的正确答案为:{answer}
    请先写出最终答案对比过程,然后给出答案,按照以下格式回答:
    -format-
    对比过程:
    正确答案为:xxx
    回答的答案为:xxx
    判断:
    是否正确:是/否
    -end format-

    '''
    correct_answers_data = llm(correct_answers_prompt)
    if re.search('是否正确.{1,2}是', correct_answers_data):
        return True
    else:
        return False


def get_best_chain_of_thought(question, options, answer):
    correct_data = []

    for _ in range(1):
        chain_of_thought_data = generate_chain_of_thought_and_answer(question, options)
        if is_answer_correct(chain_of_thought_data, answer):
            correct_data.append(chain_of_thought_data)
    if not correct_data:
        return False

    if len(correct_data) == 1:
        return correct_data[0]

    common_thought_prompt = f'以下是针对问题{question}的几条回答:'

    for inx, data in enumerate(correct_data):
        common_thought_prompt += f'\n第{inx + 1}个答案:{answer}\n\n'

    common_thought_prompt += '''请从以上答案中,按照以下步骤分析
    1.给每个答案从思想通用性的角度进行打分
    2.排序找到最通用的答案思路
    3.给出文档ID,格式为:`最好的答案为第[1,2,3,4,5]个`
    请从第一个步骤开始执行:
    '''
    common_thought_data = llm(common_thought_prompt)
    for inx in range(len(correct_data)):
        common_thought_best = re.search(f'第{inx+1}个答案', common_thought_data)
        if common_thought_best:

            return correct_data[inx]


development_data = [{"meta_info": "taiwanese_test_Q", "question": "39 这一位中年男性，最近出现如图a 之病变，其皮肤切片病理检查结果如图b，经免疫萤光直接法检查，可見IgG 在表皮细胞间沉积如图c，则最可能的诊断是：\n", "answer_idx": "C", "answer": "天疱疮（pemphigus vulgaris）", "options": {"A": "水疱型脓痂疹（bullous type impetigo）", "B": "類天疱疮（bullous pemphigoid）", "C": "天疱疮（pemphigus vulgaris）", "D": "后天性水疱性表皮松解症（epidermolysis bullosa acquisita）"}},
                    {"meta_info": "taiwanese_test_Q", "question": "下列药物可能引起急性肾脏损伤，而主要的作用是肾血管的影响，何者例外？", "answer_idx": "C", "answer": "含铂的抗癌制剂（如cisplatin）", "options": {"A": "非类固醇消炎剂（nonsteroidal anti-inflammatory drugs）", "B": "血管张力素阻断剂（angiotensin-converting enzyme inhibitors）", "C": "含铂的抗癌制剂（如cisplatin）", "D": "肾素抑制剂（renin inhibitors）"}},
                    {"meta_info": "taiwanese_test_Q", "question": "下列何者不是心脏复健运动所造成生理上的变化？", "answer_idx": "A", "answer": "增加预估最大心跳率（estimated maximum heart rate）", "options": {"A": "增加预估最大心跳率（estimated maximum heart rate）", "B": "增加最大耗氧量（VO max）", "C": "减少病人在运动时之最大心肌耗氧量（maximal myocardial oxygen capacity）", "D": "降低血管周边阻力（peripheral resistance）"}},
                    {"meta_info": "taiwanese_test_Q", "question": "有关肝炎病毒，下列之叙述何者错误？", "answer_idx": "C", "answer": "HAV、HBV、HCV 及 HDV 均是单股（single-stranded）RNA 病毒", "options": {"A": "HAV 及 HEV 是粪－口途径传播", "B": "HBV 及 HCV 可能经由血液传染", "C": "HAV、HBV、HCV 及 HDV 均是单股（single-stranded）RNA 病毒", "D": "HBV 是含有套膜（envelope）的病毒"}},
                    {"meta_info": "taiwanese_test_Q", "question": "临床上高度怀疑是肺癌的病人，同时呈现高血钙（hypercalcemia）之现象时，则其肺部肿瘤之病理切片结果最可能是：", "answer_idx": "D", "answer": "Squamous cell carcinoma", "options": {"A": "Adenocarcinoma", "B": "Pleomorphic carcinoma", "C": "Small cell carcinoma", "D": "Squamous cell carcinoma"}},
                    {"meta_info": "taiwanese_test_Q", "question": "关于遗传学原理应用于临床医学上之叙述，下列何者错误？", "answer_idx": "A", "answer": "粒线体基因突变造成之疾病为父系遗传（paternal transmission）", "options": {"A": "粒线体基因突变造成之疾病为父系遗传（paternal transmission）", "B": "对于单基因孟德尔式遗传疾病（monogenic Mendelian disorders），遗传模式（mode of", "C": "基因体印记（genomic imprinting）现象，会使某些疾病遗传模式不符单基因孟德尔式遗传模式", "D": "复杂性遗传疾病（complex genetic disorders），其临床表现易受环境因素所影响"}},
                    {"meta_info": "taiwanese_test_Q", "question": "在上臂（arm）处，臂神经丛发出的那兩条神经没有分支？", "answer_idx": "D", "answer": "正中与尺神经（median and ulnar nerves）", "options": {"A": "肌皮与正中神经（musculocutaneous and median nerves）", "B": "尺与桡神经（ulnar and radial nerves）", "C": "肌皮与桡神经（musculocutaneous and radial nerves）", "D": "正中与尺神经（median and ulnar nerves）"}},
                    {"meta_info": "taiwanese_test_Q", "question": "一位 70 岁男性，因为吸入性肺炎引发呼吸衰竭接受呼吸器治療，现因身体狀况改善，考虑脱離呼吸器。下列那样结果可预测其呼吸器脱離之失败率很高？", "answer_idx": "D", "answer": "RR（Respiratory rate）30/min 且 tidal volume 200 mL", "options": {"A": "血压 120/80 mmHg", "B": "最大吸气压（Maximal inspiratory pressure）为-30 cmH2O", "C": "动脉血 pH 7.35-7.40", "D": "RR（Respiratory rate）30/min 且 tidal volume 200 mL"}},
                    {"meta_info": "taiwanese_test_Q", "question": "活化的巨噬细胞会分泌：", "answer_idx": "C", "answer": "interleukin-4", "options": {"A": "interleukin-2", "B": "L-2)", "C": "interleukin-4", "D": "L-4)"}},
                    {"meta_info": "taiwanese_test_Q", "question": "48岁男性末期肾病病⼈，接受⻑期免疫抑制剂治疗，移植前无糖尿病病史，移植3个⽉后⾎中肌酸酐0.8 mg/dL，饭前⾎糖210 mg/dL，下列何种药物最有可能产⽣此种合并症？", "answer_idx": "B", "answer": "tacrolimus", "options": {"A": "mycophenolate mofetil", "B": "tacrolimus", "C": "sirolimus", "D": "everolimus"}}
]


# 预处理
embedding_vectors = []
chains_of_thought = []

# 预处理开发数据集中的每个问题。
for question, options, answer in development_data:
    chain_of_thought = get_best_chain_of_thought(question, options, answer)
    if chain_of_thought:
        embedding = get_embedding(question)
        embedding_vectors.append(embedding)
        chains_of_thought.append(chain_of_thought)



# 推理时间
# 计算测试问题的嵌入。
# 占位数据结构，用于开发数据和测试问题。
test_question = "下列何者由連接蛋白（connexin）所构成？"
test_options = {"A": "segment IV 和 segment VI", "B": "segment IV 和 segment V", "C": "segment V 和 segment VII",
                "D": "segment VI 和 segment VII"}

test_embedding = get_embedding(test_question)

# 计算余弦相似性的函数。
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - cosine(a, b)


# 基于余弦相似性找出最相似的5个问题。
similarities = [cosine_similarity(test_embedding, vec) for vec in embedding_vectors]
top_indices = np.argsort(similarities)[-5:]

# 检索最相似的5个例子。
similar_examples = [(development_data[i], chains_of_thought[i]) for i in top_indices]

# 将这些例子格式化为上下文。

prompt = '你将谨慎推理,一步步思考,并且回答用户的问题,请参考以下参考案例:'

for development_data, chains_of_thought in similar_examples:
    prompt += f'```问题:{development_data},解答过程:{chains_of_thought}```\n\n'

# 多次打乱并使用语言模型生成答案。
generated_chains_of_thought = []
generated_answers = []
for _ in range(5):
    random_options = randomize_options(test_options)
    prompt = f'''请谨慎思考,一步步的进行推理,最后得到答案,按照以下格式回答:
    -format-
    推理过程:
    1.xxx
    2.xxx
    3.xxx
    ...
    结论:xxx
    -end format-
    你需要回答的问题为:`{test_question},从{random_options}中选择正确的选项`
'''
    generated_chains_of_thought.append(llm(prompt))


print('最后跑完拉:',generated_chains_of_thought)
answer = []
for i in generated_chains_of_thought:
    answer.append(re.search('结论:.*',i).group())

answer = []
for i in generated_chains_of_thought:
    answer.append(re.search('结论:.*', i).group())

print(answer)

counter_prompt = f'以下是五个专家对某个问题的投票结果:\n'
for inx, ans in enumerate(answer):
    counter_prompt += f'第{inx + 1}位专家的结果:{ans}\n\n'

counter_prompt += '请统计专家结论票数,并给出哪几位专家的结论是最多的.'


print(llm(counter_prompt))