# Chat-llm
Create a chat robot to help you
## 项目背景

在学习和工作中，我们总是会面临各种各样的情感问题，而在当下的浮躁与快节奏中，我们很难独自消化，也不善于向外展现我们的脆弱。

基于此，我们开发了一款带RAG (Retrieval Augmented Generation, 检索增强生成) 的大模型聊天机器人，专注解答情感问题，希望能够帮助大家健康度地过我们的余生。

## 项目动机

当前社会，生活和工作节奏正在加快，同时，城市化将人们分割成了一个个完全独立的个体。这已经和几十年前的群居生活完全不同。由于「计划生育政策」，许多家庭只有一个小孩，情感需求已经变得越来越重要。在读书的人，周中是披星戴月的学习，周末是补不完的课外辅导班和写不完的作业；在工作的人，无论是在熟悉，或者陌生的城市里，大多都是一个人在生活。缺乏交流的他们，很容易在快节奏和高压环境下累计各种各样情感问题。

因此，我们开发了一款利用rag框架的LLM聊天机器人……

## Introduction
In this project, we use the openai or deepseek api to create a chat robot. 
Additionally, we use langchain to memorize the chat history.
## Usage
1. Install the dependencies
`pip install -r requirements.txt`
2. How to run
   - Run code in terminal  
   ``
   uvicorn main:app --reload
   ``
   - Run code in IDE  
   ``
   Run main.py directly
   ``
3. Chat with the robot
   - local  
   http://127.0.0.1:8000/docs
   - dev
   - uat

## 项目进展
注：由于本人对开发和大模型并不熟悉，无法做到一下做完，只能先实现「最小模型」，在此基础上进行迭代更新。因此，项目进展并不快。项目进展按日进行记录

### 1. 创建基于Fastapi的API，实现前端传入问题到后端

- 为后续将回答传给后端大模型做准备

### 2. 购买DeepSeek API key，并成功在本地连接DeepSeek

- 实现在Python项目中与DeepSeek交流

### 3. 整合1与2

- 实现「前端输入数据」→ 「数据发送后端DeepSeek」→ 「DeepSeek回答问题」的功能

### 4. 在3的基础上，增加消息返回功能

- 将DS的回答和SessionID发送给前端，实现在用户界面进行展示的功能

### 5. 创建SessionId 用于识别不同对话

- 考虑实际场景：不同会话需要用SessionId进行标识。同时在用户输入消息时判断：如果用户没有填写SessionId，则会自动新建一个SessionId，表示这是一个新的会话；如果用户填入了SessionId，表明该会话已经存在，不再自动生成。

### 6. 存储大模型对话记录

- 当Python服务重启后，即使输入已有SessionId，大模型也无法读取其历史记录，因此，需要持久化存储。
- 为方便可视化，创建conversations文件夹，利用Json存储每一次聊天记录；同时，基于SQLalchemy创建数据库存储对话数据。

### 7.  用LangChain重构大模型

- Langchain是最流行的大模型开发包之一，并且LC兼容DS。因此，利用LC重构已有框架。
- 在重构后，通过配置LC线程，增加记忆功能，让大模型能够记住历史对话内容。当用户在上一次对话完成后，询问「我上一次问了什么」时，大模型也能够准确回答。

### 8. 建立历史会话读取机制

- 实现在每一次开启会话时，会自动判断该对话是否有历史消息。若有，则将该会话下最新的一则消息与用户最新输入经过整合后，一并发送给大模型。
- 修改system prompt，使其明白历史消息的意义。

### 9. 更改启动设置，优化主函数

- 将terminal的启动方式变更为run main.py，在控制台能够看到完整的长文字输出
- 利用FastAPI里的路由，将main里的api剥离至router中，简化主函数内容
- 将历史会话读取机制封装为函数，简化主函数里的内容

### 10. 数据库方法迁移

- 将数据保存至数据库、调用数据库中历史消息记录等方法迁移至service文件夹中
- 修改db.py文件中，创建数据库的路径，确保查询数据库的路径正确

### 11. 创建数据库用于构建RAG

- RAG可能包括爬虫、分词、向量化等操作。需要理清楚后才能进行下一步
- 可以暂时存储一些假数据，代替「爬虫」

### 12. Embedding模型

- 利用embedding实现向量化嵌入操作
- 采用Qwen embedding模型

### 13. 将用户提问与embedding进行整合
- 为后续数据库存储向量做准备

### 14. 创建数据库存储向量

### 其他后续

- 融入Springboot服务
- 加入agent以及agent到hierarchy机制
- 设计更加精细化的后端
- 制作前端页面