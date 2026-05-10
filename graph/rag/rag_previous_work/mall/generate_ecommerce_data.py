"""电商商品数据生成脚本 — 生成真实风格的蓝牙耳机/手机等品类数据"""

import json
import random
import datetime
import os

from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

load_dotenv()

DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}"
    f"/{os.getenv('MYSQL_DATABASE')}"
)

engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()


# ================== 表定义 ==================

class Product(Base):
    """SPU 表 — 标准产品单位"""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)
    brand = Column(String(50))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.now)

    skus = relationship("ProductSKU", back_populates="product")
    reviews = relationship("ProductReview", back_populates="product")


class ProductSKU(Base):
    """SKU 表 — 库存量单位"""
    __tablename__ = "product_skus"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    sku_name = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)
    original_price = Column(Float)
    stock = Column(Integer, default=0)
    specs = Column(String(500))
    is_active = Column(Boolean, default=True)

    product = relationship("Product", back_populates="skus")


class ProductReview(Base):
    """商品评价表"""
    __tablename__ = "product_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    user_name = Column(String(50))
    rating = Column(Integer)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.now)

    product = relationship("Product", back_populates="reviews")


class CartItem(Base):
    """购物车表"""
    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    sku_id = Column(Integer, ForeignKey("product_skus.id"), nullable=False)
    quantity = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.datetime.now)


class EcommerceOrder(Base):
    """电商订单表"""
    __tablename__ = "ecommerce_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    sku_id = Column(Integer, ForeignKey("product_skus.id"), nullable=False)
    quantity = Column(Integer, default=1)
    total_price = Column(Float)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.datetime.now)


# ================== 商品数据 ==================

PRODUCTS = [
    {
        "name": "QCY T13 真无线蓝牙耳机",
        "category": "蓝牙耳机",
        "brand": "QCY",
        "description": "QCY T13 是一款入门级真无线蓝牙耳机，搭载蓝牙5.1芯片，连接稳定延迟低。采用13mm大动圈单元，音质饱满层次分明。单次续航约7小时，配合充电盒总续航可达30小时。支持触控操作、IPX5防水，适合日常通勤和运动使用。充电盒支持Type-C快充，30分钟可充至50%。耳机重量仅4.2g，佩戴舒适无感。",
        "skus": [
            {"name": "白色", "price": 89.0, "original_price": 129.0, "stock": 500, "specs": '{"颜色":"白色","续航":"30h","防水":"IPX5"}'},
            {"name": "黑色", "price": 89.0, "original_price": 129.0, "stock": 350, "specs": '{"颜色":"黑色","续航":"30h","防水":"IPX5"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "性价比超高！89块买到这个音质很惊喜，低音很足，续航也够一天用。充电很快，戴久了耳朵不疼。"},
            {"rating": 4, "content": "音质不错，连接稳定。就是白色容易脏，建议买黑色。续航确实能达到30小时左右。"},
            {"rating": 4, "content": "通勤路上用的，降噪一般但价格摆在这。触控操作很方便，切歌接电话都很流畅。"},
            {"rating": 3, "content": "用了两个月左耳偶尔断连，重启后恢复。其他方面都不错，对得起价格。"},
        ],
    },
    {
        "name": "漫步者 Lolli3 真无线蓝牙耳机",
        "category": "蓝牙耳机",
        "brand": "漫步者",
        "description": "漫步者 Lolli3 采用高通QCC3056芯片，支持aptX Adaptive高清解码，蓝牙5.3连接更稳定。13mm复合振膜动圈，三频均衡。单次续航8小时，总续航32小时。支持主动降噪(ANC)，降噪深度约25dB。IP54防尘防水，支持游戏模式(延迟低至60ms)。充电盒支持无线充电。",
        "skus": [
            {"name": "云白色", "price": 199.0, "original_price": 259.0, "stock": 280, "specs": '{"颜色":"云白色","降噪":"ANC 25dB","续航":"32h"}'},
            {"name": "雅黑色", "price": 199.0, "original_price": 259.0, "stock": 420, "specs": '{"颜色":"雅黑色","降噪":"ANC 25dB","续航":"32h"}'},
            {"name": "雾霾蓝", "price": 219.0, "original_price": 279.0, "stock": 150, "specs": '{"颜色":"雾霾蓝","降噪":"ANC 25dB","续航":"32h"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "音质真的很好！aptX解码听起来比同价位耳机清晰很多。降噪在地铁上够用，人声能隔绝大部分。"},
            {"rating": 5, "content": "游戏模式延迟很低，打王者基本无感延迟。续航也很顶，充满电能用好几天。"},
            {"rating": 4, "content": "佩戴舒适度不错，但降噪效果不如入耳式的。音质对得起200块的价格。"},
            {"rating": 3, "content": "无线充电有点慢，建议直接用线充。其他方面中规中矩，性价比一般。"},
        ],
    },
    {
        "name": "索尼 WF-1000XM5 旗舰降噪耳机",
        "category": "蓝牙耳机",
        "brand": "索尼",
        "description": "索尼 WF-1000XM5 是索尼最新旗舰真无线降噪耳机，搭载V2处理器+QN2e芯片，降噪性能较上代提升约40%。采用8.4mm碳纤维振膜单元，支持LDAC高码率传输。单次续航8小时(ANC开)，总续航24小时。支持360 Reality Audio空间音频、自适应声音控制、多点连接。IPX4防水，充电盒支持Qi无线充电。重量仅5.9g，比上代小25%。",
        "skus": [
            {"name": "黑色", "price": 1799.0, "original_price": 1999.0, "stock": 80, "specs": '{"颜色":"黑色","降噪":"V2+QN2e","续航":"24h","防水":"IPX4"}'},
            {"name": "银色", "price": 1799.0, "original_price": 1999.0, "stock": 60, "specs": '{"颜色":"银色","降噪":"V2+QN2e","续航":"24h","防水":"IPX4"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "降噪天花板！飞机上基本听不到引擎声。音质比XM4提升明显，LDAC模式下听无损很爽。佩戴比上代舒服很多。"},
            {"rating": 5, "content": "除了贵没有缺点。自适应降噪非常智能，能根据环境自动调节。360音效看电影很震撼。"},
            {"rating": 4, "content": "音质和降噪确实顶级，但1799的价格对大多数人来说还是贵了。续航比上一代略短。"},
        ],
    },
    {
        "name": "苹果 AirPods Pro 2 (USB-C)",
        "category": "蓝牙耳机",
        "brand": "苹果",
        "description": "Apple AirPods Pro 第二代 (USB-C版)，搭载H2芯片，主动降噪性能是上代的两倍。自适应通透模式可在降低噪音的同时保留环境声。支持个性化空间音频和动态头部追踪。单次续航6小时(ANC开)，总续航30小时。充电盒支持MagSafe、Apple Watch充电器和USB-C充电。IP54防尘防水，支持查找功能(充电盒自带扬声器)。",
        "skus": [
            {"name": "USB-C版", "price": 1699.0, "original_price": 1899.0, "stock": 120, "specs": '{"接口":"USB-C","降噪":"H2芯片","续航":"30h","防水":"IP54"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "苹果生态的最佳选择，降噪比上一代好太多了。自适应通透模式很实用，不用摘耳机也能听到周围人说话。"},
            {"rating": 5, "content": "空间音频看电影效果很棒，头部追踪很灵敏。USB-C接口终于统一了充电线。"},
            {"rating": 4, "content": "好用但性价比不高。和苹果设备配合完美，但安卓用户不建议买。"},
        ],
    },
    {
        "name": "Redmi Buds 5 Pro 降噪耳机",
        "category": "蓝牙耳机",
        "brand": "小米",
        "description": "Redmi Buds 5 Pro 搭载蓝牙5.3，支持46dB深度主动降噪，同价位降噪深度领先。11mm大动圈单元，支持LHDC 5.0高清音频解码。单次续航10小时，总续航38小时。支持三麦克风通话降噪、双设备连接、游戏模式(延迟低至49ms)。IP55防尘防水。充电盒支持快充，10分钟充电可使用3小时。",
        "skus": [
            {"name": "晴空白", "price": 299.0, "original_price": 399.0, "stock": 600, "specs": '{"颜色":"晴空白","降噪":"46dB","续航":"38h"}'},
            {"name": "极夜黑", "price": 299.0, "original_price": 399.0, "stock": 450, "specs": '{"颜色":"极夜黑","降噪":"46dB","续航":"38h"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "299块做到46dB降噪，小米真的卷。LHDC音质比AAC好很多，听流行歌很爽。续航38h真的够用一周。"},
            {"rating": 4, "content": "降噪效果不错但比不过AirPods Pro。性价比很高，适合预算有限的同学。"},
            {"rating": 4, "content": "游戏模式延迟低，吃鸡基本无感延迟。不过佩戴久了有点涨耳。"},
        ],
    },
    {
        "name": "Redmi Note 13 Pro",
        "category": "手机",
        "brand": "小米",
        "description": "Redmi Note 13 Pro 搭载骁龙7s Gen2处理器，6.67英寸1.5K AMOLED屏幕，120Hz刷新率。后置2亿像素主摄(三星HP3传感器，支持OIS光学防抖)，800万超广角+200万微距。前置1600万像素。5100mAh大电池+67W快充。支持NFC、红外遥控、3.5mm耳机孔。玻璃后盖+塑料中框，厚度7.98mm，重量187g。",
        "skus": [
            {"name": "8GB+128GB 子夜黑", "price": 1299.0, "original_price": 1499.0, "stock": 200, "specs": '{"内存":"8GB","存储":"128GB","颜色":"子夜黑"}'},
            {"name": "8GB+256GB 子夜黑", "price": 1499.0, "original_price": 1699.0, "stock": 180, "specs": '{"内存":"8GB","存储":"256GB","颜色":"子夜黑"}'},
            {"name": "12GB+256GB 镜瓷白", "price": 1699.0, "original_price": 1899.0, "stock": 150, "specs": '{"内存":"12GB","存储":"256GB","颜色":"镜瓷白"}'},
            {"name": "12GB+512GB 浅梦空间", "price": 1999.0, "original_price": 2199.0, "stock": 100, "specs": '{"内存":"12GB","存储":"512GB","颜色":"浅梦空间"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "2亿像素拍照真的清晰！夜景模式也很强。1.5K屏幕看着很舒服，67W充电够快。这个价位性价比很高。"},
            {"rating": 4, "content": "性能中规中矩，骁龙7s够日常使用。电池耐用，一天半没问题。拍照是最大的亮点。"},
            {"rating": 4, "content": "给父母买的，屏幕大字体清晰，续航好。就是有点重，187g对于手小的女生来说偏沉。"},
        ],
    },
    {
        "name": "iPhone 15",
        "category": "手机",
        "brand": "苹果",
        "description": "iPhone 15 搭载A16仿生芯片，6.1英寸Super Retina XDR显示屏，支持动态岛。后置4800万像素主摄+1200万像素超广角，支持2倍光学品质变焦。前置1200万像素TrueDepth摄像头。USB-C接口首次取代Lightning。支持车祸检测、卫星紧急SOS。续航最长20小时视频播放。航空级铝金属机身，IP68防水。",
        "skus": [
            {"name": "128GB 黑色", "price": 5399.0, "original_price": 5999.0, "stock": 90, "specs": '{"存储":"128GB","颜色":"黑色"}'},
            {"name": "128GB 粉色", "price": 5399.0, "original_price": 5999.0, "stock": 70, "specs": '{"存储":"128GB","颜色":"粉色"}'},
            {"name": "256GB 蓝色", "price": 6399.0, "original_price": 6999.0, "stock": 80, "specs": '{"存储":"256GB","颜色":"蓝色"}'},
            {"name": "512GB 绿色", "price": 8399.0, "original_price": 8999.0, "stock": 30, "specs": '{"存储":"512GB","颜色":"绿色"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "终于换成USB-C了！出门不用带两根线。动态岛很实用，拍照比上一代好很多。A16性能依然很强。"},
            {"rating": 4, "content": "中规中矩的升级，没有ProMotion高刷是遗憾。但日常使用很流畅，生态体验还是苹果最好。"},
            {"rating": 3, "content": "5399起步但只有128GB，安卓同价位已经256GB起步了。续航一般，重度使用撑不了一天。"},
        ],
    },
    {
        "name": "vivo X100",
        "category": "手机",
        "brand": "vivo",
        "description": "vivo X100 搭载天玑9300旗舰处理器，6.78英寸AMOLED曲面屏，120Hz刷新率，支持2160Hz高频PWM调光。后置蔡司联合调校影像系统：5000万像素主摄(索尼IMX920，OIS)+5000万超广角+5000万长焦(2x光变)。前置3200万像素。5000mAh电池+120W闪充，11分钟充至50%。支持NFC、红外、IP68防水。",
        "skus": [
            {"name": "12GB+256GB 星迹蓝", "price": 3699.0, "original_price": 3999.0, "stock": 120, "specs": '{"内存":"12GB","存储":"256GB","颜色":"星迹蓝"}'},
            {"name": "16GB+256GB 落日橙", "price": 3999.0, "original_price": 4299.0, "stock": 100, "specs": '{"内存":"16GB","存储":"256GB","颜色":"落日橙"}'},
            {"name": "16GB+512GB 白月光", "price": 4499.0, "original_price": 4799.0, "stock": 60, "specs": '{"内存":"16GB","存储":"512GB","颜色":"白月光"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "蔡司影像太强了！拍人像肤色自然，夜景噪点控制很好。天玑9300性能流畅，120W充电太快了。"},
            {"rating": 5, "content": "屏幕素质优秀，2160Hz PWM调光晚上看手机不伤眼。续航也不错，中度使用一天半。"},
            {"rating": 4, "content": "拍照确实好，但系统广告有点多。曲面屏不太习惯，怕摔。其他方面都很满意。"},
        ],
    },
    {
        "name": "小米充电宝 20000mAh 50W版",
        "category": "充电宝",
        "brand": "小米",
        "description": "小米 20000mAh 充电宝 50W版，支持50W MAX快充，可为笔记本电脑充电。双向快充，输入输出均支持PD/QC协议。三口输出(USB-A x2 + USB-C x1)，可同时为三台设备充电。20000mAh大容量，可为iPhone 15充电约4.5次。数显屏幕实时显示剩余电量。航空级铝合金外壳，支持登机携带。重量约405g。",
        "skus": [
            {"name": "黑色", "price": 169.0, "original_price": 199.0, "stock": 300, "specs": '{"容量":"20000mAh","功率":"50W","颜色":"黑色"}'},
            {"name": "银色", "price": 169.0, "original_price": 199.0, "stock": 250, "specs": '{"容量":"20000mAh","功率":"50W","颜色":"银色"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "能给笔记本充电太方便了！50W功率充MacBook Air没问题。数显很实用，不用猜还有多少电。"},
            {"rating": 4, "content": "容量大但确实有点重，405g放包里还是能感觉到。充电速度很快，三口同时输出也不发热。"},
            {"rating": 5, "content": "出差必备！一个充电宝搞定手机+平板+笔记本。上飞机没问题，安检直接过。"},
        ],
    },
    {
        "name": "Anker 安克 MagGo 10000mAh 磁吸充电宝",
        "category": "充电宝",
        "brand": "安克",
        "description": "Anker MagGo 磁吸充电宝，10000mAh容量，支持MagSafe磁吸无线充电(15W)和USB-C有线快充(20W)。可吸附iPhone 12及以上机型背面，边充边用不挡手。自带折叠支架，可当手机支架使用。LED指示灯显示电量状态。重量仅210g，超薄设计(12mm)。支持Qi无线充电协议，兼容安卓设备。",
        "skus": [
            {"name": "白色", "price": 199.0, "original_price": 249.0, "stock": 180, "specs": '{"容量":"10000mAh","功率":"15W无线+20W有线","颜色":"白色"}'},
            {"name": "蓝色", "price": 199.0, "original_price": 249.0, "stock": 150, "specs": '{"容量":"10000mAh","功率":"15W无线+20W有线","颜色":"蓝色"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "磁吸太方便了！吸在iPhone背面很牢固，边充电边刷视频不挡手。自带支架好评。"},
            {"rating": 4, "content": "无线充电速度一般，建议用线充更快。但便携性真的好，比带线充电宝方便多了。"},
            {"rating": 4, "content": "10000mAh够日常用，能给iPhone充2次左右。颜值高，送给女朋友她很喜欢。"},
        ],
    },
    {
        "name": "华为 Watch GT4 46mm",
        "category": "智能手表",
        "brand": "华为",
        "description": "华为 Watch GT4 46mm版本，搭载鸿蒙4.0系统。1.43英寸AMOLED高清屏幕，466x466分辨率。支持全天候血氧监测、心率监测、睡眠监测、压力监测。100+运动模式，支持GPS/北斗/GLONASS/Galileo四系统定位。蓝牙通话、NFC门禁/公交卡。续航最长14天(典型使用场景8天)。5ATM防水，支持游泳佩戴。不锈钢表壳+氟橡胶表带。",
        "skus": [
            {"name": "曜石黑", "price": 1488.0, "original_price": 1688.0, "stock": 100, "specs": '{"尺寸":"46mm","颜色":"曜石黑","续航":"14天"}'},
            {"name": "山茶棕", "price": 1588.0, "original_price": 1788.0, "stock": 80, "specs": '{"尺寸":"46mm","颜色":"山茶棕","续航":"14天"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "续航真的强！充满电用了10天还有15%。运动记录很准确，GPS定位快。睡眠监测数据很有参考价值。"},
            {"rating": 4, "content": "表盘很漂亮，46mm大小刚好。鸿蒙系统流畅但应用生态不如Apple Watch。运动功能很全面。"},
            {"rating": 5, "content": "NFC功能太实用了，门禁公交抬腕即刷。血氧和心率监测很灵敏，帮我发现了睡眠呼吸暂停问题。"},
        ],
    },
    {
        "name": "小米手环8 Pro",
        "category": "智能手表",
        "brand": "小米",
        "description": "小米手环8 Pro，1.74英寸AMOLED大屏，60Hz刷新率，支持息屏显示。全天候血氧/心率/睡眠监测，150+运动模式，支持独立GNSS定位(无需带手机)。14天超长续航，支持5ATM防水。NFC版支持门禁卡、公交卡。支持蓝牙通话、消息通知、天气、闹钟等。铝合金边框，金属质感表带快拆设计。重量仅26g。",
        "skus": [
            {"name": "标准版 黑色", "price": 299.0, "original_price": 349.0, "stock": 500, "specs": '{"版本":"标准版","颜色":"黑色","续航":"14天"}'},
            {"name": "NFC版 银色", "price": 349.0, "original_price": 399.0, "stock": 400, "specs": '{"版本":"NFC版","颜色":"银色","续航":"14天"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "299块买到这个配置超值！屏幕大清晰度高，独立GPS跑步不用带手机了。续航真的能到14天。"},
            {"rating": 4, "content": "性价比之王。运动数据记录准确，睡眠监测比上一代精确多了。就是不支持第三方应用。"},
            {"rating": 5, "content": "NFC版强烈推荐！刷门禁刷公交太方便了。金属质感比上一代好很多，不像手环更像手表了。"},
        ],
    },
    {
        "name": "iPad Air (M2芯片)",
        "category": "平板电脑",
        "brand": "苹果",
        "description": "iPad Air 搭载Apple M2芯片，性能较上一代M1提升约50%。11英寸Liquid Retina显示屏，支持P3广色域和True Tone。1200万像素后置摄像头+1200万像素超广角前置(横置)。支持Apple Pencil Pro和妙控键盘。Touch ID指纹识别。Wi-Fi 6E+蓝牙5.3。USB-C接口支持USB 3速度。续航最长10小时。支持5G蜂窝网络版本可选。",
        "skus": [
            {"name": "128GB Wi-Fi 星光色", "price": 4799.0, "original_price": 4999.0, "stock": 50, "specs": '{"存储":"128GB","网络":"Wi-Fi","颜色":"星光色"}'},
            {"name": "256GB Wi-Fi 深空灰", "price": 5599.0, "original_price": 5799.0, "stock": 40, "specs": '{"存储":"256GB","网络":"Wi-Fi","颜色":"深空灰"}'},
            {"name": "256GB 5G 蓝色", "price": 6799.0, "original_price": 6999.0, "stock": 20, "specs": '{"存储":"256GB","网络":"5G","颜色":"蓝色"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "M2性能太强了！剪4K视频毫无压力。配合Apple Pencil Pro画画手感很好。学生党记笔记神器。"},
            {"rating": 5, "content": "屏幕素质优秀，看PDF和论文很舒服。续航够用，带去图书馆一天不用充电。轻薄便携。"},
            {"rating": 4, "content": "128GB起步有点小，建议256GB起步。没有高刷是遗憾。除此之外没什么槽点。"},
        ],
    },
    {
        "name": "华为 MatePad Pro 13.2英寸",
        "category": "平板电脑",
        "brand": "华为",
        "description": "华为 MatePad Pro 13.2英寸，搭载麒麟9000S处理器。13.2英寸OLED柔性屏，2.8K分辨率，144Hz刷新率，支持1440Hz高频PWM调光。1300万像素后置+1600万像素前置。10100mAh超大电池+88W超级快充。支持华为M-Pencil(第三代)，4096级压感。鸿蒙4.0系统，支持多屏协同、超级终端。四扬声器+Histen 8.0音效。厚度仅5.5mm，重量580g。",
        "skus": [
            {"name": "12GB+256GB Wi-Fi 曜金黑", "price": 4999.0, "original_price": 5499.0, "stock": 40, "specs": '{"内存":"12GB","存储":"256GB","颜色":"曜金黑"}'},
            {"name": "16GB+512GB Wi-Fi 晶钻白", "price": 5999.0, "original_price": 6499.0, "stock": 30, "specs": '{"内存":"16GB","存储":"512GB","颜色":"晶钻白"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "13.2英寸大屏看论文太爽了！OLED屏幕色彩鲜艳，144Hz很流畅。88W充电真的快，20分钟充满一半。"},
            {"rating": 5, "content": "鸿蒙生态的多屏协同很好用，手机直接拖文件到平板。M-Pencil写字延迟很低，做笔记很舒服。"},
            {"rating": 4, "content": "屏幕和续航都很顶，但麒麟9000S性能不如M2。适合办公和学习，玩游戏不太行。"},
        ],
    },
    {
        "name": "罗技 MX Keys S 无线键盘",
        "category": "键盘",
        "brand": "罗技",
        "description": "罗技 MX Keys S 是一款高端无线薄膜键盘，支持蓝牙和Logi Bolt接收器双模连接，最多可配对3台设备并一键切换。球形凹面键帽设计，手感接近机械键盘。支持智能背光(手靠近自动亮起)、USB-C充电。支持Logi Options+自定义按键和Flow跨电脑复制粘贴。续航最长10天(背光开)或5个月(背光关)。全尺寸布局，铝合金机身。",
        "skus": [
            {"name": "黑色", "price": 599.0, "original_price": 699.0, "stock": 120, "specs": '{"颜色":"黑色","连接":"蓝牙+接收器","续航":"10天"}'},
            {"name": "白色", "price": 599.0, "original_price": 699.0, "stock": 80, "specs": '{"颜色":"白色","连接":"蓝牙+接收器","续航":"10天"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "办公键盘的天花板！键程适中手感很好，背光智能不浪费电。多设备切换很流畅，Flow跨电脑复制粘贴太好用了。"},
            {"rating": 4, "content": "手感确实好但价格偏贵。薄膜键盘不如机械耐用，但便携性和静音效果优秀。"},
            {"rating": 5, "content": "程序员必备！自定义按键映射很实用，三台电脑无缝切换。充电一次用好久。"},
        ],
    },
    {
        "name": "VGN V98Pro V2 机械键盘",
        "category": "键盘",
        "brand": "VGN",
        "description": "VGN V98Pro V2 是一款98配列三模机械键盘(有线/2.4G/蓝牙5.0)，搭载极地狐轴V2(线性轴，手感轻润)。全键热插拔，支持3/5针轴座。Gasket结构+Poron夹心棉+IXPE轴下垫，消音效果优秀。1.2mm厚PCB，RGB背光。4000mAh大电池，无线续航约200小时(灯光关)。支持驱动自定义按键和灯光。PBT双色注塑键帽，耐磨不打油。重量约950g。",
        "skus": [
            {"name": "极地狐轴V2 奶白色", "price": 299.0, "original_price": 399.0, "stock": 350, "specs": '{"轴体":"极地狐轴V2","颜色":"奶白色","连接":"三模"}'},
            {"name": "极地狐轴V2 深空灰", "price": 299.0, "original_price": 399.0, "stock": 280, "specs": '{"轴体":"极地狐轴V2","颜色":"深空灰","连接":"三模"}'},
            {"name": "冰淇淋Pro轴 白色", "price": 349.0, "original_price": 449.0, "stock": 200, "specs": '{"轴体":"冰淇淋Pro轴","颜色":"白色","连接":"三模"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "299块的Gasket键盘！声音很好听，麻将音。极地狐轴手感顺滑，打字不累。热插拔方便换轴。"},
            {"rating": 5, "content": "性价比爆表！这个配置其他品牌要400+。消音做得很好，办公室用不吵人。RGB灯效也不错。"},
            {"rating": 4, "content": "无线模式延迟略高，打游戏建议用2.4G。其他方面都很满意，299买到这个品质很值。"},
        ],
    },
    {
        "name": "罗技 GPW3 (G Pro X Superlight 2)",
        "category": "鼠标",
        "brand": "罗技",
        "description": "罗技 GPW3 是 GPW 系列的第三代旗舰无线游戏鼠标，重量仅60g。搭载HERO 2传感器，支持最高32000 DPI。LIGHTSPEED无线技术，延迟低至1ms。LIGHTFORCE混合光学微动，点击寿命约1亿次。续航最长95小时。USB-C充电。5个可编程按键，支持G HUB自定义。支持POWERPLAY无线充电系统。",
        "skus": [
            {"name": "黑色", "price": 899.0, "original_price": 999.0, "stock": 60, "specs": '{"颜色":"黑色","重量":"60g","DPI":"32000"}'},
            {"name": "白色", "price": 899.0, "original_price": 999.0, "stock": 50, "specs": '{"颜色":"白色","重量":"60g","DPI":"32000"}'},
            {"name": "粉色", "price": 949.0, "original_price": 1049.0, "stock": 30, "specs": '{"颜色":"粉色","重量":"60g","DPI":"32000"}'},
        ],
        "reviews": [
            {"rating": 5, "content": "60g真的太轻了！用了就回不去。无线延迟和有线一样，打FPS完全无感延迟。续航也很顶。"},
            {"rating": 5, "content": "GPW系列一代比一代强。微动升级后点击更清脆，传感器精度更高。除了贵没毛病。"},
            {"rating": 4, "content": "手感很好但899确实贵。适合重度游戏玩家，轻度用户买GPW2性价比更高。"},
        ],
    },
]


def generate_data():
    Base.metadata.create_all(engine)
    session = Session()

    review_templates = [
        "张同学", "李同学", "王同学", "刘同学", "陈同学",
        "赵同学", "杨同学", "周同学", "吴同学", "黄同学",
    ]

    for prod_data in PRODUCTS:
        product = Product(
            name=prod_data["name"],
            category=prod_data["category"],
            brand=prod_data["brand"],
            description=prod_data["description"],
        )
        session.add(product)
        session.flush()

        for sku_data in prod_data["skus"]:
            sku = ProductSKU(
                product_id=product.id,
                sku_name=sku_data["name"],
                price=sku_data["price"],
                original_price=sku_data["original_price"],
                stock=sku_data["stock"],
                specs=sku_data["specs"],
            )
            session.add(sku)

        for review_data in prod_data["reviews"]:
            review = ProductReview(
                product_id=product.id,
                user_name=random.choice(review_templates),
                rating=review_data["rating"],
                content=review_data["content"],
                created_at=datetime.datetime.now() - datetime.timedelta(
                    days=random.randint(1, 60)
                ),
            )
            session.add(review)

    session.commit()
    session.close()
    total_skus = sum(len(p["skus"]) for p in PRODUCTS)
    total_reviews = sum(len(p["reviews"]) for p in PRODUCTS)
    print(f"已生成 {len(PRODUCTS)} 个SPU, {total_skus} 个SKU, {total_reviews} 条评价")


if __name__ == "__main__":
    generate_data()
