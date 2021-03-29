# 特征工程

## 处理数据


```python
# 全量信息处理,合并数据
all_data =  train_data.append(test_data)
all_data = all_data.merge(user_info, on="user_id", how="left")
all_data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>merchant_id</th>
      <th>label</th>
      <th>prob</th>
      <th>age_range</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34176</td>
      <td>3906</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34176</td>
      <td>121</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34176</td>
      <td>4356</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34176</td>
      <td>2217</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>230784</td>
      <td>4818</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>

```python
# wewewwe
del train_data, test_data, user_info
gc.collect()
```


    16896


```python
# 对用户日志按照时间排序
user_log = user_log.sort_values(by=["user_id", "time_stamp"])

# 构建表达式(统计用户的历史信息)
list_join_func = lambda x: " ".join(str(i) for i in x)

agg_dict = {
    "item_id": list_join_func,
    "cat_id": list_join_func,
    "seller_id": list_join_func,
    "brand_id": list_join_func,
    "time_stamp": list_join_func,
    "action_type": list_join_func
}

rename_dict = {
    "item_id": "item_path",
    "cat_id": "cat_path",
    "seller_id": "seller_path",
    "brand_id": "brand_path",
    "time_stamp": "time_stamp_path",
    "action_type": "action_type_path"
}
```


```python
# 聚合用户历史购买行为
# user_log.groupby("user_id").agg(agg_dict)
```


```python
def merge_list(df_ID, join_columns, df_data, agg_dict, rename_dict):
    """合并用户信息和日志信息"""
    
    # 构建用户历史记录
    df_data = df_data.groupby("user_id").agg(agg_dict).reset_index().rename(columns=rename_dict)
    
    df_ID = df_ID.merge(df_data, on=join_columns, how="left")
    
    return df_ID

all_date = merge_list(all_data, "user_id", user_log, agg_dict, rename_dict)
```


```python
# 回收内存和删除不必要的数据
# gc.collect()
# del user_log
```

## 定义特征统计函数

### 定义统计函数


```python
# 统计数据总数
def cnt_(x):
    
    try:
        return len(x.split(" "))
    except:
        return -1    
```


```python
# 统计唯一值数量
def nunique_(x):
    
    try:
        return len(set(x.split(" ")))
    except:
        return -1    
```


```python
# 统计数据最大值
def max_(x):
    
    try:
        return np.max([float(i) for i in x.split(" ")])
    except:
        return -1
```


```python
# 统计数据最小值
def min_(x):
    
    try:
        return np.min([float(i) for i in x.split(" ")])
    except:
        return -1
```


```python
# 统计数据标准差
def std_(x):
    
    try:
        return np.std([float(i) for i in x.split(" ")])
    except:
        return -1
```


```python
# 统计TopN的数据
def most_n(x, n):
    
    try:
        return Counter(x.split(" ")).most_common(n)[n-1][0] # 名称
    except:
        return -1
    
def most_n_cnt(x, n):
    
    try:
        return Counter(x.split(" ")).most_common(n)[n-1][1] # 数量
    except:
        return -1
```

### 定义调用函数


```python
def user_cnt(df_data, single_col, name):
    
    df_data[name] = df_data[single_col].apply(cnt_)
    
    return df_data
```


```python
def user_nunique(df_data, single_col, name):
    
    df_data[name] = df_data[single_col].apply(nunique_)
    
    return df_data
```


```python
def user_max(df_data, single_col, name):
    
    df_data[name] = df_data[single_col].apply(max_)
    
    return df_data
```


```python
def user_min(df_data, single_col, name):
    
    df_data[name] = df_data[single_col].apply(min_)
    
    return df_data
```


```python
def user_std(df_data, single_col, name):
    
    df_data[name] = df_data[single_col].apply(std_)
    
    return df_data
```


```python
def user_cnt(df_data, single_col, name):
    
    df_data[name] = df_data[single_col].apply(cnt_)
    
    return df_data
```


```python
def user_most_n(df_data, single_col, name, n=1):
    
    func = lambda x: most_n(x, n)
    df_data[name] = df_data[single_col].apply(func)
    
    return df_data
```


```python
def user_most_n_cnt(df_data, single_col, name, n=1):
    
    func = lambda x: most_n_cnt(x, n)
    df_data[name] = df_data[single_col].apply(func)
    
    return df_data
```

### 提取统计变量(用户特征构造)

#### 统计用户的多样性


```python
# 数据部分采样和全采样
# all_data_test = all_date.sample(2000)
all_data_test = all_date
```


```python
# 每一位用户所点击的商铺总次数(不去重)
all_data_test = user_cnt(all_data_test, "seller_path", "user_seller_cnt")

# 店铺个数(去重)
all_data_test = user_nunique(all_data_test, "seller_path", "user_seller_unique")

# 不同种类
all_data_test = user_nunique(all_data_test, "cat_path", "user_cat_unique")

# 不同商品
all_data_test = user_nunique(all_data_test, "item_path", "user_item_unique")

# 不同品牌
all_data_test = user_nunique(all_data_test, "brand_path", "user_brand_unique")

# 活跃天数
all_data_test = user_nunique(all_data_test, "time_stamp_path", "user_time_stamp_unique")

# 用户不同行为的种类数目
all_data_test = user_nunique(all_data_test, "action_type_path", "user_action_type_unique")
```


```python
# 查看构建特征对购买的影响情况
new_feature = ["user_seller_unique", "user_cat_unique", "user_item_unique"
               , "user_brand_unique", "user_time_stamp_unique", "user_action_type_unique"]

for feature in new_feature:
    print("Feature:",feature)
    print(all_data_test.groupby("label")[feature].agg(["mean", "std"]))
```

    Feature: user_seller_unique
                mean        std
    label                      
    0.0    35.447798  36.650101
    1.0    37.904965  38.137931
    Feature: user_cat_unique
                mean        std
    label                      
    0.0    23.878262  18.266488
    1.0    26.464142  19.018367
    Feature: user_item_unique
                mean         std
    label                       
    0.0    81.077354  110.089651
    1.0    97.890797  128.547510
    Feature: user_brand_unique
                mean        std
    label                      
    0.0    34.847149  34.815715
    1.0    37.365848  36.349934
    Feature: user_time_stamp_unique
                mean        std
    label                      
    0.0    17.391973  14.719989
    1.0    19.992289  15.992401
    Feature: user_action_type_unique
               mean       std
    label                    
    0.0    2.634301  0.551990
    1.0    2.667503  0.537855


#### 时间类信息获取


```python
# 最早时间
all_data_test = user_min(all_data_test, "time_stamp_path", "user_time_stamp_min")

# 最近时间
all_data_test = user_max(all_data_test, "time_stamp_path", "user_time_stamp_max")

# 用户点击时间间隔
all_data_test["time_stamp_range"] = all_data_test["user_time_stamp_max"] - all_data_test["user_time_stamp_min"]

# 活跃天数的方差,描述用户的活跃的波动情况
all_data_test = user_std(all_data_test, "time_stamp_path", "user_time_stamp_std")
```

#### 其他基本信息


```python
# 用户最喜欢的店铺
all_data_test = user_most_n(all_data_test, "seller_path", "user_seller_most_1", n=1)

# 用户最喜欢的类目
all_data_test = user_most_n(all_data_test, "cat_path", "user_cat_most_1", n=1)

# 用户最喜欢的品牌
all_data_test = user_most_n(all_data_test, "brand_path", "user_brand_most_1", n=1)

# 用户最常见的行为操作
all_data_test = user_most_n(all_data_test, "action_type_path", "user_action_type_1", n=1)

# 统计最喜欢的次数
# 用户最喜欢的店铺
all_data_test = user_most_n_cnt(all_data_test, "seller_path", "user_seller_most_1_cnt", n=1)

# 用户最喜欢的类目
all_data_test = user_most_n_cnt(all_data_test, "cat_path", "user_cat_most_1_cnt", n=1)

# 用户最喜欢的品牌
all_data_test = user_most_n_cnt(all_data_test, "brand_path", "user_brand_most_1_cnt", n=1)

# 用户最常见的行为操作
all_data_test = user_most_n_cnt(all_data_test, "action_type_path", "user_action_type_1_cnt", n=1)
```

#### 类别特征编码为哑变量


```python
age_range = pd.get_dummies(all_data_test["age_range"], prefix="age")
gender = pd.get_dummies(all_data_test["gender"], prefix="gender")
all_data_test = all_data_test.join(age_range)
all_data_test = all_data_test.join(gender)
```


```python
all_data_test.columns
```


    Index(['user_id', 'merchant_id', 'label', 'prob', 'age_range', 'gender',
           'item_path', 'cat_path', 'seller_path', 'brand_path', 'time_stamp_path',
           'action_type_path', 'user_seller_cnt', 'user_seller_unique',
           'user_cat_unique', 'user_item_unique', 'user_brand_unique',
           'user_time_stamp_unique', 'user_action_type_unique',
           'user_time_stamp_min', 'user_time_stamp_max', 'time_stamp_range',
           'user_time_stamp_std', 'user_seller_most_1', 'user_cat_most_1',
           'user_brand_most_1', 'user_action_type_1', 'user_seller_most_1_cnt',
           'user_cat_most_1_cnt', 'user_brand_most_1_cnt',
           'user_action_type_1_cnt', 'age_0.0', 'age_1.0', 'age_2.0', 'age_3.0',
           'age_4.0', 'age_5.0', 'age_6.0', 'age_7.0', 'age_8.0', 'gender_0.0',
           'gender_1.0', 'gender_2.0'],
          dtype='object')


```python
import copy

def col_cnt_(df_data, columns_list, action_type):
    """统计点击数量"""

    data_dict = {}
    try:
        col_list = copy.deepcopy(columns_list)
        
        if action_type != None:
            col_list += ["action_type_path"]
        
        for col in col_list:
            data_dict[col] = df_data[col].split(" ")
            
        path_len = len(data_dict[col])
        
        # {"sell_path": ["66", "55", ......]}
        data_out = []
        for i_ in range(path_len):
            data_txt = ""
            for col_ in columns_list:
                # 统计点击购买行为
                if data_dict["action_type_path"][i_]  == action_type:
                    data_txt += "_" + data_dict[col_][i_]
            data_out.append(data_txt)
            
        return len(data_out)        
    except:
        return -1
```

### 构建用户与商家的交互信息
例如：用户对商铺的点击行为特征，挖掘其是行为对点击情况的影响作用。

#### 用户的点击情况


```python
# 消除重复点击
def col_unique_(df_data, columns_list, action_type):

    data_dict = {}
    try:
        col_list = copy.deepcopy(columns_list)
        
        if action_type != None:
            col_list += ["action_type_path"]
        
        for col in col_list:
            data_dict[col] = df_data[col].split(" ")
            
        path_len = len(data_dict[col])
        
        # {"sell_path": ["66", "55", ......], "action_type_path":["1", "0", "2", "3", "0",......]}
        data_out = []

        for i_ in range(path_len):
            data_txt = ""
            for col_ in columns_list:
                # 统计点击购买行为
                if data_dict["action_type_path"][i_]  == action_type:
                    data_txt += "_" + data_dict[col_][i_]
            data_out.append(data_txt)
            
        return len(set(data_out))        
    except:

        return -1
```


```python
def user_col_cnt_(df_data, columns_list, action_type, name):
    
    df_data[name] = df_data.apply(lambda x:col_cnt_(x, columns_list, action_type), axis=1)
    
    return df_data

def user_col_unique_(df_data, columns_list, action_type, name):
    
    df_data[name] = df_data.apply(lambda x:col_unique_(x, columns_list, action_type), axis=1)
    
    return df_data
```


```python
# 不同店铺的点击次数
all_data_test = user_col_cnt_(all_data_test, ["seller_path"], "1", "user_cnt_1")
all_data_test = user_col_cnt_(all_data_test, ["seller_path"], "0", "user_cnt_0")
all_data_test = user_col_cnt_(all_data_test, ["seller_path"], "2", "user_cnt_2")
all_data_test = user_col_cnt_(all_data_test, ["seller_path"], "3", "user_cnt_3")
```


```python
# 店铺唯一值
all_data_test = user_col_unique_(all_data_test, ["seller_path"], "0", "user_unique_0")
all_data_test = user_col_unique_(all_data_test, ["seller_path"], "1", "user_unique_1")
all_data_test = user_col_unique_(all_data_test, ["seller_path"], "2", "user_unique_2")
all_data_test = user_col_unique_(all_data_test, ["seller_path"], "3", "user_unique_3")
```

#### 统计用户历史对该商品的评分


```python
def user_merchant_mark(df_data, merchant_id, seller_path, action_type_path, action_type):
    """统计历史用户的打分情况"""
    
    sell_len = len(df_data[seller_path].split(" "))
    data_dict = {}
    data_dict[seller_path] = df_data[seller_path].split(" ")
    data_dict[action_type_path] = df_data[action_type_path].split(" ")
    
    # 遍历历史商铺数据访问数据
    mark = 0
    for i in range(sell_len):
        if data_dict[seller_path][i] == str(df_data[merchant_id]):
            if data_dict[action_type_path][i] == action_type:
                mark += 1
    return 0
```


```python
def user_merchant_mark_all(user_data, merchant_id, seller_path, action_type_path, action_type, name):
    """统计所有用户的点击情况"""
    
    user_data[name + "_" + action_type] = user_data.apply(lambda x: user_merchant_mark(x, merchant_id, seller_path, action_type_path, action_type), axis=1)
    
    return user_data
```


```python
# 用户针对此商家有多少次 0、1、2、3动作 
all_data_test = user_merchant_mark_all(all_data_test,'merchant_id','seller_path','action_type_path','0','user_merchant_action')
all_data_test = user_merchant_mark_all(all_data_test,'merchant_id','seller_path','action_type_path','1','user_merchant_action')
all_data_test = user_merchant_mark_all(all_data_test,'merchant_id','seller_path','action_type_path','2','user_merchant_action')
all_data_test = user_merchant_mark_all(all_data_test,'merchant_id','seller_path','action_type_path','3','user_merchant_action')
```


```python
all_data_test.columns
```




    Index(['user_id', 'merchant_id', 'label', 'prob', 'age_range', 'gender',
           'item_path', 'cat_path', 'seller_path', 'brand_path', 'time_stamp_path',
           'action_type_path', 'user_seller_cnt', 'user_seller_unique',
           'user_cat_unique', 'user_item_unique', 'user_brand_unique',
           'user_time_stamp_unique', 'user_action_type_unique',
           'user_time_stamp_min', 'user_time_stamp_max', 'time_stamp_range',
           'user_time_stamp_std', 'user_seller_most_1', 'user_cat_most_1',
           'user_brand_most_1', 'user_action_type_1', 'user_seller_most_1_cnt',
           'user_cat_most_1_cnt', 'user_brand_most_1_cnt',
           'user_action_type_1_cnt', 'age_0.0', 'age_1.0', 'age_2.0', 'age_3.0',
           'age_4.0', 'age_5.0', 'age_6.0', 'age_7.0', 'age_8.0', 'gender_0.0',
           'gender_1.0', 'gender_2.0', 'user_cnt_1', 'user_cnt_0', 'user_cnt_2',
           'user_cnt_3', 'user_unique_0', 'user_unique_1', 'user_unique_2',
           'user_unique_3', 'user_merchant_action_0', 'user_merchant_action_1',
           'user_merchant_action_2', 'user_merchant_action_3'],
          dtype='object')

### 提取统计变量(商铺特征构造)


```python
# user_log_test = user_log.sample(2000)
user_log_test = user_log
```


```python
# 构建表达式(统计用户的历史信息)
list_join_func = lambda x: " ".join(str(i) for i in x)
agg_seller_dict = {
    "item_id": list_join_func,
    "cat_id": list_join_func,
    "brand_id": list_join_func,
    "user_id": list_join_func,
    "time_stamp": list_join_func,
    "action_type": list_join_func
}

rename_seller_dict = {
    "item_id": "item_path",
    "cat_id": "cat_path",
    "user_id": "user_path",
    "brand_id": "brand_path",
    "time_stamp": "time_stamp_path",
    "action_type": "action_type_path"
}
```

#### 商铺基本信息
例如：访问人数、商品种类数目、商铺中的品牌数目


```python
# 构建商品信息表
user_log_seller = user_log_test.groupby("seller_id").agg(agg_seller_dict).reset_index().rename(columns=rename_seller_dict)

# 统计商铺被用户点击的总次数
user_log_seller = user_cnt(user_log_seller, "user_path", "seller_user_cnt")

# 统计商铺有多少不同消费着（上一步去重）
user_log_seller = user_nunique(user_log_seller, "user_path", "seller_user_unique")

# 商铺有多少种类商品
user_log_seller = user_nunique(user_log_seller, "cat_path", "seller_cat_unique")

# 商铺有多少不同的商品
user_log_seller = user_nunique(user_log_seller, "item_path", "seller_item_unique")

# 商铺有多少不同的品牌
user_log_seller = user_nunique(user_log_seller, "brand_path", "seller_brand_unique")

# 商铺被点击的时间相差天数
user_log_seller = user_nunique(user_log_seller, "action_type_path", "seller_action_type_unique")
```

#### 商铺时间信息


```python
# 最早时间
user_log_seller = user_min(user_log_seller, "time_stamp_path", "seller_time_stamp_min")

# 最近时间
user_log_seller = user_max(user_log_seller, "time_stamp_path", "seller_time_stamp_max")

# 用户点击时间间隔
user_log_seller["seller_time_stamp_range"] = user_log_seller["seller_time_stamp_max"] - user_log_seller["seller_time_stamp_min"]

# 活跃天数的方差,描述商铺被访问时间的活跃的波动情况
user_log_seller = user_std(user_log_seller, "time_stamp_path", "seller_user_time_stamp_std")

# 商铺被点击的时间活跃天数(商铺的活跃度)
user_log_seller = user_nunique(user_log_seller, "time_stamp_path", "seller_time_stamp_unique")
```

#### 商铺被点击信息


```python
# 统计商品被点击情况
def user_action_cnt(df_data,col_action,action_type,name):
    
    func = lambda x: len([i for i in x.split(' ') if i == action_type])
    df_data[name+'_'+action_type] = df_data[col_action].apply(func) 
    return df_data
```


```python
user_log_seller = user_action_cnt(user_log_seller, "action_type_path", "1", "seller_cnt_1")
user_log_seller = user_action_cnt(user_log_seller, "action_type_path", "0", "seller_cnt_0")
user_log_seller = user_action_cnt(user_log_seller, "action_type_path", "2", "seller_cnt_2")
user_log_seller = user_action_cnt(user_log_seller, "action_type_path", "3", "seller_cnt_3")
```

## 合并数据


```python
user_log_seller = user_log_seller.rename(columns={"seller_id": "merchant_id"})
seller_features = [c for c in user_log_seller.columns if c not in 
                   ["item_path", "cat_path", "user_path", "brand_path", "time_stamp_path", "action_type_path"]]

user_log_seller  = user_log_seller[seller_features]

all_data_test = all_data_test.merge(user_log_seller, on="merchant_id", how="left")
```


```python
all_data_test.columns
```


    Index(['user_id', 'merchant_id', 'label', 'prob', 'age_range', 'gender',
           'item_path', 'cat_path', 'seller_path', 'brand_path', 'time_stamp_path',
           'action_type_path', 'user_seller_cnt', 'user_seller_unique',
           'user_cat_unique', 'user_item_unique', 'user_brand_unique',
           'user_time_stamp_unique', 'user_action_type_unique',
           'user_time_stamp_min', 'user_time_stamp_max', 'time_stamp_range',
           'user_time_stamp_std', 'user_seller_most_1', 'user_cat_most_1',
           'user_brand_most_1', 'user_action_type_1', 'user_seller_most_1_cnt',
           'user_cat_most_1_cnt', 'user_brand_most_1_cnt',
           'user_action_type_1_cnt', 'age_0.0', 'age_1.0', 'age_2.0', 'age_3.0',
           'age_4.0', 'age_5.0', 'age_6.0', 'age_7.0', 'age_8.0', 'gender_0.0',
           'gender_1.0', 'gender_2.0', 'user_cnt_1', 'user_cnt_0', 'user_cnt_2',
           'user_cnt_3', 'user_unique_0', 'user_unique_1', 'user_unique_2',
           'user_unique_3', 'user_merchant_action_0', 'user_merchant_action_1',
           'user_merchant_action_2', 'user_merchant_action_3', 'seller_user_cnt',
           'seller_user_unique', 'seller_cat_unique', 'seller_item_unique',
           'seller_brand_unique', 'seller_action_type_unique',
           'seller_time_stamp_min', 'seller_time_stamp_max',
           'seller_time_stamp_range', 'seller_user_time_stamp_std',
           'seller_time_stamp_unique', 'seller_cnt_1_1', 'seller_cnt_0_0',
           'seller_cnt_2_2', 'seller_cnt_3_3'],
          dtype='object')