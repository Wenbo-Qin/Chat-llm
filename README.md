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
3. How to use chunking, embedding, vector store
   Due to long time waiting for chunking, we highly recommend you to chunking before using rag.  
   - run faiss_store.py  
      - It will chunking, embedding data and store vectors in faiss  
   - waiting patiently for chunking  

4. Open website of Chat with the robot
   - local  
      [click here](http://localhost:8000/docs)
   - dev
   - uat
---

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
- 单独做了一个api，而非在askLLM里套用embedding
   - 该API已废弃，因为用RAG的技术中已经实现了embedding功能

### 14. MCP方法与 MCP Server集成
- 将MCP方法集成至MCP Server中

### 15. Team leader 搭建
- 创建team leader，能够根据用户意图选择合适的方法
   - 方法包括：llm_chat, llm_rag, llm_query

### 16. 建立RAG检索库，并完成chunking -> embedding -> store
- 知乎检索20篇文章作为语料库，并chunking, embedding, vector store
- 用faiss进行vector store

### 17. 在llm_rag中构建子workflow，实现RAG功能
- 已实现rag_graph，作为workflow graph的子图
- 在rag_graph中实现了对用户query的embedding和相似度查询
- 返回文档，交给大模型生成回答

### 18. 加入ReAct框架，可以代替team_leader
- 详见API /react-ask
- RAG仍采用graph的形式

### 19. ReAct框架加入session_id记忆机制
- 同时如果输入了session_id，则会调用load_history_conversation加载历史聊天记录
   - **注意：该历史聊天记录可能是原始文件，后续需要优化**

### 20. 使用RecursiveCharacterTextSplitter而非定长chunking
- chunking结果更加整洁

### 21. 采用OCR技术对pdf文档进行识别与chunking
- MinerU技术（[详情点击此处](https://mineru.net/apiManage/docs)）
- pdf文件较大，因此不上传了

### 22. 为RAG任务制作扩展查询功能
- 通过agent，将用户的query扩展成多个query

### 23. 新增思考-行动-观察细节，更加用户友好
- 详见 react_workflow.py

### 24. 表格获取问题 （以EOS6DⅡ说明书.pdf为例）
- RAG在获取表格内容时，会保留原始表格内容，如</tr> (pdf转markdown的结果)、|| (pdf转docx的结果)
   - document_processor_test.py 分析表格特征，做了一些探索
- 表格截断问题
   - 每一个chunk，先判断是不是表格，如果是，可以考虑和前一个chunk合并
- 表格跨页问题

### 25. RAG获取文档的质量问题
- 以 上海芯导...PDF为例，当查询“告诉我上海芯导公司 主营业务分行业、分产品、分地区、分销售模式情况”的时候，返回很多无关答案（大多无关回答都多次包含 上海芯导公司 这几个字），因为频率过高导致相似度过高
   - 采取 重排序、 chunk优化策略
### 其他后续
- 重排序机制设计
- 融入Springboot服务
- 加入agent以及agent到hierarchy机制
- 设计更加精细化的后端
- 制作前端页面